#include <numeric>
#include <mutex>

#include <pcl/common/transforms.h>
#include <pcl/common/io.h>

#include <opencv2/core/utility.hpp>

#include <glog/logging.h>

#include "CaliErrDetector.hpp"
#include "Utils.hpp"
#include "Misc.hpp"
#include "CovisGraph.hpp"
#include "CovisDataPool.hpp"
#include "ManualPhotoErr.hpp"
#include "Rand.hpp"
#include "StopWatch.hpp"

using phc::CaliErrDetector;

namespace {
    const float kDeg2Rad = M_PI / 180.0f;
}

class CaliErrDetector::Impl {
public:
	Impl(const ErrDetectorConf& config) : conf_(config){}

	void GenerateExtriGrid();
    void RemoveDataOutOfWindow();
    float SingleExtrinsicsCost(const Eigen::Isometry3f &extri, 
        PtCloudXYZI_t::Ptr cloud, const std::vector<PtId_t> &poi);
    float SingleExtrinsicsCost(const Eigen::Isometry3f& extri, const std::unordered_set<PtId_t> &mask);
    void SampleExtriGrid(std::vector<Eigen::Isometry3f> &samples, int sample_num);
    void GenerateCovisData(const PtCloudXYZI_t &cloud, const std::vector<PtId_t>& poi);

	std::vector<Eigen::Isometry3f> grid_;

    std::vector<DataFrame> data_window_;

    bool extri_dirty_flag_ = false;
    Eigen::Isometry3f extri_; // T_cl

    float trans_step_size_ = 0.05f;  // Meters
    float rot_step_size_ = 0.5f;  // Degrees
    int step_num_ = 2;

    // int sample_num = 30;

    bool dist_thresh_dirty_flag_ = false;
    float dist_thresh_ = 0.0f;
    CamIntri intri_;

    ErrDetectorConf conf_;

    CovisDataPool data_pool_;
    CovisGraph cg_;
};

CaliErrDetector::CaliErrDetector(const ErrDetectorConf& config) : impl_(new Impl(config)) {}

CaliErrDetector::~CaliErrDetector() {}


void CaliErrDetector::Impl::GenerateExtriGrid() {
    static constexpr float kDim = 6;

    CHECK(extri_dirty_flag_);

    // step = 1 -> [-1, 0, 1]
    const int num_per_dimension = 1 + 2 * step_num_;
    const int start_num = 0 - step_num_;
    const int grid_size = std::pow(num_per_dimension, kDim);

    LOG(INFO) << "[CaliErrDetector] Step per dimension: " << num_per_dimension;
    LOG(INFO) << "[CaliErrDetector] Grid size: " << grid_size;

    grid_.clear();
    grid_.reserve(grid_size);

    std::vector<int> loc(num_per_dimension);
    std::iota(loc.begin(), loc.end(), start_num);

    Eigen::Vector3f init_trans(extri_.translation());
    Eigen::Vector3f init_rot(extri_.rotation().eulerAngles(0, 1, 2));

    float rot_step_size_rad = rot_step_size_ * kDeg2Rad;

    for (int tran_x = 0; tran_x < loc.size(); ++tran_x) {
        for (int tran_y = 0; tran_y < loc.size(); ++tran_y) {
            for (int tran_z = 0; tran_z < loc.size(); ++tran_z) {
                for (int rot_x = 0; rot_x < loc.size(); ++rot_x) {
                    for (int rot_y = 0; rot_y < loc.size(); ++rot_y) {
                        for (int rot_z = 0; rot_z < loc.size(); ++rot_z) {

                            // If all zero, equal to center
                            if (loc[tran_x] == loc[tran_y] && loc[tran_x] == loc[tran_z] && 
                                loc[tran_x] == loc[rot_x] && loc[tran_x] == loc[rot_y] &&
                                loc[tran_x] == loc[rot_z] && loc[tran_x] == 0) {
                                continue;
                            }
                            Eigen::Vector3f trans(static_cast<float>(init_trans.x() + loc[tran_x] * trans_step_size_),
                                static_cast<float>(init_trans.y() + loc[tran_y] * trans_step_size_),
                                static_cast<float>(init_trans.z() + loc[tran_z] * trans_step_size_));
                            Eigen::Vector3f rot(static_cast<float>(init_rot.x() + loc[rot_x] * rot_step_size_rad),
                                static_cast<float>(init_rot.y() + loc[rot_y] * rot_step_size_rad),
                                static_cast<float>(init_rot.z() + loc[rot_z] * rot_step_size_rad));

                            Eigen::Isometry3f iso(utils::EulerToQuat(rot.x(), rot.y(), rot.z()));
                            iso.pretranslate(trans);

                            grid_.push_back(iso);
                        }
                    }
                }
            }
        }
    }

    extri_dirty_flag_ = false;
}

void CaliErrDetector::Impl::SampleExtriGrid(std::vector<Eigen::Isometry3f>& samples, int sample_num) {
    CHECK_LE(sample_num, grid_.size()) << "Grid size: " << grid_.size();
    
    samples.reserve(sample_num);
    std::sample(grid_.begin(), grid_.end(), std::back_inserter(samples), sample_num, utils::Rand::GetMt19937());
}

void CaliErrDetector::Impl::RemoveDataOutOfWindow() {
    while (data_window_.size() > conf_.window_size) {
        data_window_.erase(data_window_.begin());
    }
}

float CaliErrDetector::Impl::SingleExtrinsicsCost(const Eigen::Isometry3f& extri, 
    PtCloudXYZI_t::Ptr cloud, const std::vector<PtId_t>& poi) {

    // Covis graph based on given extrinsics
    CovisGraph cg;
    misc::ComputeCovisInfo(cg, data_window_,
        *cloud, poi, intri_, extri,
        conf_.covis_conf.max_view_range,
        conf_.covis_conf.score_thresh,
        conf_.covis_conf.edge_discard,
        conf_.pixel_val_lower_lim, 
        conf_.pixel_val_upper_lim);
    cg.EraseLessObservedPt(conf_.obs_thresh);

    // Prepare data pool
    CovisGraph cg_new;
    CovisDataPool dp;
    dp.Add(cg, *cloud, data_window_, extri, cg_new);

    // Weight computation
    misc::ComputeWeightsOnGraph(cg_new, dp, dist_thresh_, dist_thresh_ * conf_.trans_thresh_ratio);

    // Mask generate
    std::unordered_set<PtId_t> mask;
    cg_new.GenTransWeightMask(mask);

    // ManualPhotoErr only store reference for CovisDataPool & CovisGraph
    // Make sure those instances are alive
    ManualPhotoErr man_err(dp, cg_new, intri_);
    man_err.SetExtri(extri);
    man_err.SetPyramidLevel(conf_.pyramid_lvl);
    man_err.SetTransMask(mask);

    float cost;
    man_err.Compute(cost);

    return cost;
}

float CaliErrDetector::Impl::SingleExtrinsicsCost(const Eigen::Isometry3f& extri, const std::unordered_set<PtId_t>& mask) {
    // ManualPhotoErr only store reference for CovisDataPool & CovisGraph
    // Make sure those instances can out-live ManualPhotoErr instance
    ManualPhotoErr man_err(data_pool_, cg_, intri_);
    man_err.SetExtri(extri);
    man_err.SetPyramidLevel(conf_.pyramid_lvl);
    // man_err.SetPyramidLevel(1);
    man_err.SetTransMask(mask);

    float cost;
    man_err.Compute(cost);

    return cost;
}

void CaliErrDetector::Impl::GenerateCovisData(const PtCloudXYZI_t& cloud, const std::vector<PtId_t>& poi) {
    // Covis graph based on init extrinsics
    CovisGraph cg;
    misc::ComputeCovisInfo(cg, data_window_,
        cloud, poi, intri_, extri_,
        conf_.covis_conf.max_view_range,
        conf_.covis_conf.score_thresh,
        conf_.covis_conf.edge_discard,
        conf_.pixel_val_lower_lim,
        conf_.pixel_val_upper_lim);
    cg.EraseLessObservedPt(conf_.obs_thresh);

    // Prepare data pool
    CovisGraph cg_new;
    data_pool_.Add(cg, cloud, data_window_, extri_, cg_new);

    cg_.Merge(cg_new);
}

void CaliErrDetector::SetExtri(const Eigen::Isometry3f& extri) {
    impl_->extri_ = extri;
    impl_->extri_dirty_flag_ = true;
}

void CaliErrDetector::Detect(float& center_cost, std::vector<float>& grid_costs) {
    Impl& impl = *impl_;

    utils::StopWatch sw;

    if (impl.extri_dirty_flag_) {
        impl.GenerateExtriGrid();
    }

    std::vector<Eigen::Isometry3f> extri_samples;
    impl.SampleExtriGrid(extri_samples, impl.conf_.extri_sample_num);

    // Weight computation
    misc::ComputeWeightsOnGraph(impl.cg_, impl.data_pool_, impl.dist_thresh_, impl.dist_thresh_ * impl.conf_.trans_thresh_ratio);

    // Mask generate
    std::unordered_set<PtId_t> mask;
    impl.cg_.GenTransWeightMask(mask);

    //int cnt = 0;
    //grid_costs.reserve(impl.grid_.size());
    //for (const Eigen::Isometry3f &err_extri : extri_samples) {
    //    LOG_IF(WARNING, poi.size() < 10000) << "[ErrDetector] Points of interest < 10000, may not be sufficient";

    //    float cost = impl.SingleExtrinsicsCost(err_extri, cloud, poi);
    //    grid_costs.push_back(cost);
    //    LOG(INFO) << "[ErrDetector] Computing " << cnt++ 
    //        << " out of " << extri_samples.size()
    //        << " cost: " << cost;
    //}

    std::mutex mtx;
    grid_costs.reserve(impl.grid_.size());
    cv::parallel_for_(cv::Range(0, extri_samples.size()),
        [&](const cv::Range& range) {
            for (int r = range.start; r < range.end; ++r) {
                const Eigen::Isometry3f& err_extri = extri_samples.at(r);

                // float cost = impl.SingleExtrinsicsCost(err_extri, cloud, poi);
                float cost = impl.SingleExtrinsicsCost(err_extri, mask);

                std::lock_guard<std::mutex> lck(mtx);
                grid_costs.push_back(cost);
            }
        });

    // center_cost = impl.SingleExtrinsicsCost(impl.extri_, cloud, poi);
    center_cost = impl.SingleExtrinsicsCost(impl.extri_, mask);
    LOG(INFO) << "[ErrDetector] Center cost: " << center_cost;

    LOG(INFO) << "[ErrDetector] Detection time: " << sw.GetTimeElapse();
}

void CaliErrDetector::Detect(float& center_cost) {
    Impl& impl = *impl_;

    utils::StopWatch sw;

    // Weight computation
    misc::ComputeWeightsOnGraph(impl.cg_, impl.data_pool_, impl.dist_thresh_, impl.dist_thresh_ * impl.conf_.trans_thresh_ratio);

    // Mask generate
    std::unordered_set<PtId_t> mask;
    impl.cg_.GenTransWeightMask(mask);

    center_cost = impl.SingleExtrinsicsCost(impl.extri_, mask);
    LOG(INFO) << "[ErrDetector] Center cost: " << center_cost;

    LOG(INFO) << "[ErrDetector] Detection time: " << sw.GetTimeElapse();
}

void CaliErrDetector::AddDataFrame(const PtCloudXYZI_t::Ptr& ptcloud_ptr, const cv::Mat& img, const Eigen::Isometry3f& T_wl) {
    CHECK_NE(ptcloud_ptr, nullptr);
    CHECK(!img.empty());

    DataFrame f;
    f.ptcloud = ptcloud_ptr;
    f.img = img;
    f.T_wl = T_wl;

    Impl& impl = *impl_;

    // Compute thresh
    if (impl.dist_thresh_dirty_flag_) {
        impl.dist_thresh_ = 
            misc::DistThreshCompute(impl.intri_, img.cols, img.rows, 
                impl.conf_.err_tolr_x, impl.conf_.err_tolr_y);
        impl.dist_thresh_dirty_flag_ = false;
    }

    // Ptcloud clip
    utils::RemoveCloudPtsOutOfRange(*f.ptcloud, impl.conf_.ptcloud_clip_min, impl.conf_.ptcloud_clip_max);

    // Add to window
    impl.data_window_.push_back(f);

    // Return if window not fully filled
    if (impl.data_window_.size() < impl.conf_.window_size) {
        //LOG(INFO) << "[ErrorDetector] Data window not filled, current: "
        //    << impl.data_window_.size()
        //    << ", window size: " << impl.conf_.window_size;
        return;
    }

    // Generate local cloud
    PtCloudXYZI_t::Ptr cloud(new PtCloudXYZI_t);
    misc::ConcatPointClouds(*cloud, impl.data_window_);

    std::vector<PtId_t> poi; // Points of interest
    utils::RandSampleInterestPts(poi, *cloud, impl.conf_.sample_ratio);

    LOG(INFO) << "[ErrDetector] Points of interest: " << poi.size();

    impl.GenerateCovisData(*cloud, poi);

    impl.data_window_.clear();
}

void CaliErrDetector::SetIntri(float fx, float fy, float cx, float cy) {
    impl_->intri_ = CamIntri{fx, fy, cx, cy};
    impl_->dist_thresh_dirty_flag_ = true;
}

float CaliErrDetector::WorsePercentage(const float& center_cost, const std::vector<float>& grid_costs) {
    int worse_cnt = 0;
    for (float c : grid_costs) {
        if (center_cost < c) {
            worse_cnt++;
        }
    }

    return (float)worse_cnt / (float)grid_costs.size();
}

void CaliErrDetector::ClearDataPool() {
    impl_->cg_.Clear();
    impl_->data_pool_.Clear();
}