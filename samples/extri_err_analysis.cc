#include <random>

#include <glog/logging.h>

#include <pcl/io/pcd_io.h>

#include "kitti_loader/KittiLoader.hpp"

#include "PhotoCali.hpp"
#include "Config.hpp"
#include "StopWatch.hpp"
#include "DataRecorder.hpp"

void SamplePts(std::unordered_set<phc::PtId_t>& pts, size_t start_idx, size_t end_idx, int num);
float ComputeWorsePercent(phc::utils::DataRecorder::Ptr err_recorder, phc::utils::DataRecorder::Ptr origin_recorder);
void DumpInfoIfNotWorse(phc::utils::DataRecorder::Ptr err_recorder, phc::utils::DataRecorder::Ptr origin_recorder);

int main(int argc, char const* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: ./kitti_loader <path-to-config-file>" << std::endl;
        return EXIT_FAILURE;
    }

    FLAGS_logtostderr = 1;
    google::InitGoogleLogging(argv[0]);
    LOG(INFO) << "GLOG inited";

    std::string config_path(argv[1]);
    if (!phc::utils::Config::SetFile(config_path)) {
        LOG(FATAL) << "Fail to load config file.";
    }

    // Init dataset loader
    bool if_success = false;
    kitti::KittiLoader kitti_dataset(phc::utils::Config::Get<std::string>("dataset_path"), if_success);
    CHECK(if_success);

    // Perturbation for extriinsics
    Eigen::Isometry3f extri_ptb = Eigen::Isometry3f(Eigen::AngleAxisf(M_PI / 180 * 1.0, Eigen::Vector3f(1, 0, 1)));
    extri_ptb.pretranslate(Eigen::Vector3f(0.05, 0.05, 0.0));

    // Calibrator instance
    phc::PhotoCali calibrator_origin;
    phc::PhotoCali calibrator_err;

    // Set dataset info
    Eigen::Isometry3f origin_extri = kitti_dataset.GetExtrinsics();
    Eigen::Isometry3f err_extri = extri_ptb * origin_extri;
    kitti::Intrinsics kitti_intri = kitti_dataset.GetLeftCamIntrinsics();

    calibrator_origin.SetCamIntri(kitti_intri.fx, kitti_intri.fy, kitti_intri.cx, kitti_intri.cy);
    calibrator_origin.SetExtri(origin_extri);

    calibrator_err.SetCamIntri(kitti_intri.fx, kitti_intri.fy, kitti_intri.cx, kitti_intri.cy);
    calibrator_err.SetExtri(err_extri);

    int start_f_idx = phc::utils::Config::Get<int>("start_frame_idx");
    int f_num = phc::utils::Config::Get<int>("frame_num");
    for (int i = start_f_idx; i < start_f_idx + f_num; ++i) {
        kitti::Frame f = kitti_dataset[i];
        Eigen::Isometry3f T_wl_orgin = f.gt_pose * origin_extri;
        Eigen::Isometry3f T_wl_err = f.gt_pose * err_extri;

        calibrator_origin.AddDataFrame(f.ptcloud, f.left_img, T_wl_orgin);
        calibrator_err.AddDataFrame(f.ptcloud, f.left_img, T_wl_err);
    }

    // Set recorder for original extrinsics
    phc::utils::DataRecorder::Ptr origin_recorder(new phc::utils::DataRecorder);
    calibrator_origin.SetRecorder(origin_recorder);

    // Set sample ratio
    calibrator_origin.SetRandSampleRatio(0.0001f);

    // Compute cost
    LOG(INFO) << "Computing with original extrinscis";
    phc::utils::StopWatch sw1;
    float origin_cost = calibrator_origin.ComputeCost();
    LOG(INFO) << "Time used: " << sw1.GetTimeElapse();
    LOG(INFO) << "Total cost: " << origin_cost;

    // Set recorder for err extrinsics
    phc::utils::DataRecorder::Ptr err_recorder(new phc::utils::DataRecorder);
    calibrator_err.SetRecorder(err_recorder);

    // Set interest points
    std::vector<pcl::index_t> pt_interest;
    origin_recorder->GetRecordedPtIdx(pt_interest);
    calibrator_err.SetInterestPtindices(pt_interest);

    // Compute cost (with err)
    LOG(INFO) << "Computing with err extrinscis";
    phc::utils::StopWatch sw2;
    float err_cost = calibrator_err.ComputeCost();
    LOG(INFO) << "Time used: " << sw2.GetTimeElapse();
    LOG(INFO) << "Total cost: " << err_cost;

    // Compute worse percent (higher is better)
    float worse_pct = ComputeWorsePercent(err_recorder, origin_recorder);
    LOG(INFO) << "-----";
    LOG(INFO) << "Worse percentage: " << worse_pct * 100 << "%";

    // Dump data to disk
    origin_recorder->SetDumpPath(phc::utils::Config::Get<std::string>("recorder.dump_path"));
    err_recorder->SetDumpPath(phc::utils::Config::Get<std::string>("recorder.dump_path_err"));

    DumpInfoIfNotWorse(err_recorder, origin_recorder);

    origin_recorder->DumpVisCloud();
    err_recorder->DumpVisCloud();

    // origin_recorder->DumpInfo();
    // err_recorder->DumpInfo();

    return EXIT_SUCCESS;
}

void SamplePts(std::unordered_set<phc::PtId_t> &pts, size_t start_idx, size_t end_idx, int num) {
    CHECK(start_idx < end_idx);
    CHECK(num < end_idx - start_idx);

    std::vector<phc::FrameId_t> numbers(end_idx - start_idx);
    std::iota(numbers.begin(), numbers.end(), start_idx);

    std::shuffle(numbers.begin(), numbers.end(), std::mt19937{ std::random_device{}() });

    pts.clear();
    for (int i = 0; i < num; ++i) {
        pts.insert(numbers[i]);
    }
}

float ComputeWorsePercent(phc::utils::DataRecorder::Ptr err_recorder, phc::utils::DataRecorder::Ptr origin_recorder) {
    std::vector<pcl::index_t> pt_indices;
    err_recorder->GetRecordedPtIdx(pt_indices);

    int worse_cnt = 0;
    for (const pcl::index_t &idx : pt_indices) {
        phc::utils::SinglePtInfo info_origin;
        phc::utils::SinglePtInfo info_err;
        origin_recorder->GetInfoWithGlobCloudIdx(info_origin, idx);
        err_recorder->GetInfoWithGlobCloudIdx(info_err, idx);
        if (info_origin.cost < info_err.cost) {
            worse_cnt += 1;
        }
    }

    return static_cast<float>(worse_cnt) / static_cast<float>(pt_indices.size());
}

// Dump info if the extrinsics with error does not result in higher cost
void DumpInfoIfNotWorse(phc::utils::DataRecorder::Ptr err_recorder, phc::utils::DataRecorder::Ptr origin_recorder) {
    CHECK(!err_recorder->GetDumpPath().empty());
    CHECK(!origin_recorder->GetDumpPath().empty());

    std::vector<pcl::index_t> pt_indices;
    err_recorder->GetRecordedPtIdx(pt_indices);

    for (const pcl::index_t& idx : pt_indices) {
        phc::utils::SinglePtInfo info_origin;
        phc::utils::SinglePtInfo info_err;
        origin_recorder->GetInfoWithGlobCloudIdx(info_origin, idx);
        err_recorder->GetInfoWithGlobCloudIdx(info_err, idx);
        if (info_origin.cost > info_err.cost) {
            origin_recorder->DumpInfoSinglePt(info_origin);
            err_recorder->DumpInfoSinglePt(info_err);
        }
    }
}