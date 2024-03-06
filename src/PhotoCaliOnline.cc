#include <glog/logging.h>

#include <pcl/common/transforms.h>
#include <pcl/common/io.h>

#include <opencv2/imgproc.hpp>

#include "vis_check/VisCheck.hpp"

#include "CovisGraph.hpp"
#include "CovisDataPool.hpp"
#include "PhotoCaliOnline.hpp"
#include "Utils.hpp"
#include "Rand.hpp"
#include "ManualPhotoErr.hpp"
#include "Optimizer.hpp"
#include "StopWatch.hpp"
#include "Misc.hpp"
#include "DataBag.hpp"
#include "ImgPyramid.hpp"

using phc::PhotoCaliOnline;

class PhotoCaliOnline::Impl {
public:
	Impl() : cloud_local_(new PtCloudXYZI_t){}
	Impl(const OnlineCaliConf &conf) : cloud_local_(new PtCloudXYZI_t), conf_(conf) {}

	OnlineCaliConf conf_;

	CamIntri intri_;

	Eigen::Isometry3f T_cl_;

	float dist_thresh_ = -1.0f;

	std::vector<DataFrame> data_window_;

	PtCloudXYZI_t::Ptr cloud_local_;
	std::vector<PtId_t> interest_pt_id_;

	CovisDataPool cd_pool_;
	CovisGraph global_cg_;

	// If detected error on extrinsics
	bool err_detected_ = false;

	// To be removed
	utils::DataRecorder::Ptr recorder_ = nullptr;

	// Point id mask for translation optimization
	std::unordered_set<PtId_t> trans_mask_;

	// void ConcatCloudInWindow(PtCloudXYZI_t& cloud_out);

	// poi: points of interest
	void RandSampleInterestPts(std::vector<PtId_t>& poi, const PtCloudXYZI_t& cloud, const float& sample_ratio);

	void AddCovisDataPool(const CovisGraph& cg);

	void ComputeCovisInfo(CovisGraph& cg);

	void ComputeWeightsOnGraph();

	void DistThreshCompute(int img_width, int img_height);

	visc::PtIndices InterestPtId2ViscPtIndices();

	void GridSearchExtri(Eigen::Isometry3f &extri, const std::vector<Eigen::Isometry3f> &grid);
};

//void PhotoCaliOnline::Impl::ConcatCloudInWindow(PtCloudXYZI_t& cloud_out) {
//	for (const auto& f : data_window_) {
//		const Eigen::Isometry3f& T_wl = f.T_wl;
//		PtCloudXYZI_t cloud_cp;
//		pcl::copyPointCloud(*f.ptcloud, cloud_cp);
//
//		// Transform to world frame and concat
//		pcl::transformPointCloud(cloud_cp, cloud_cp, T_wl.matrix());
//		cloud_out += cloud_cp;
//	}
//}

void PhotoCaliOnline::Impl::GridSearchExtri(Eigen::Isometry3f& optimal_extri, const std::vector<Eigen::Isometry3f>& grid) {
	std::vector<float> costs(grid.size());
	
	std::mutex mtx;
	cv::parallel_for_(cv::Range(0, grid.size()),
		[&](const cv::Range& range) {
			for (int r = range.start; r < range.end; ++r) {
				const Eigen::Isometry3f& grid_extri = grid.at(r);

				ManualPhotoErr man_err(cd_pool_, global_cg_, intri_);
				man_err.SetExtri(grid_extri);
				man_err.SetPyramidLevel(0);
				man_err.SetTransMask(trans_mask_);

				float cost;
				man_err.Compute(cost);

				std::lock_guard<std::mutex> lck(mtx);
				costs[r] = cost;
			}
		});

	int min_idx = std::min_element(costs.begin(), costs.end()) - costs.begin();

	optimal_extri = grid.at(min_idx);
}

void PhotoCaliOnline::Impl::RandSampleInterestPts(std::vector<PtId_t>& poi, const PtCloudXYZI_t& cloud, const float& sample_ratio) {
	int min = 0;
	int max = cloud.size() - 1;
	int sample_num = cloud.size() * sample_ratio;

	poi = utils::Rand::UniformDistInt(min, max, sample_num);
}

void PhotoCaliOnline::Impl::AddCovisDataPool(const CovisGraph& cg) {
	CHECK(!cloud_local_->empty());
	CHECK(!data_window_.empty());

	CovisGraph cg_new; // New covis graph with new index pointing to data in the global data pool
	cd_pool_.Add(cg, *cloud_local_, data_window_, T_cl_, cg_new);
	global_cg_.Merge(cg_new);
}

void PhotoCaliOnline::Impl::ComputeCovisInfo(CovisGraph& cg) {

	misc::ComputeCovisInfo(cg, data_window_, 
		*cloud_local_, interest_pt_id_, intri_, T_cl_, 
		conf_.covis_conf.max_view_range,
		conf_.covis_conf.score_thresh,
		conf_.covis_conf.edge_discard,
		conf_.pixel_val_lower_lim,
		conf_.pixel_val_upper_lim);
}

void PhotoCaliOnline::Impl::ComputeWeightsOnGraph() {
	CHECK_GT(dist_thresh_, 0.0f);
	CHECK(!global_cg_.Empty());
	CHECK(!cd_pool_.Empty());

	float rot_thresh = dist_thresh_;
	float trans_thresh = dist_thresh_ * conf_.trans_thresh_ratio;

	LOG(INFO) << "[PhotoCaliOnline] Rotation threshold: " << rot_thresh;
	LOG(INFO) << "[PhotoCaliOnline] Translation threshold: " << trans_thresh;

	std::vector<FrameId_t> f_ids;
	global_cg_.GetAllFrames(f_ids);
	for (const auto &f_id : f_ids) {
		std::unordered_set<PtId_t> pt_ids;
		global_cg_.FindAllVisiblePts(f_id, pt_ids);

		// Img, T_cw
		const std::pair<cv::Mat, Eigen::Isometry3f>& frame = cd_pool_.GetImgWithPose(f_id);
		for (const auto &pt_id : pt_ids) {
			const PtXYZI_t& pt = cd_pool_.GetPt(pt_id); // In world coordinate

			// Point in camera
			Eigen::Vector3f pt_cam = frame.second * Eigen::Vector3f(pt.x, pt.y, pt.z);

			float dist = pt_cam.norm();
			float rot_weight = dist < rot_thresh ? dist : rot_thresh;
			float trans_weight = dist < trans_thresh ? 1.0f : 0.0f;
			// float trans_weight = dist < trans_thresh ? trans_thresh : 0.0f;

			global_cg_.SetRotWeight(f_id, pt_id, rot_weight);
			global_cg_.SetTransWeight(f_id, pt_id, trans_weight);
		}
	}
}

void PhotoCaliOnline::Impl::DistThreshCompute(int img_width, int img_height) {
	LOG_IF(WARNING, dist_thresh_ > 0.0f) << "[PhotoCaliOnline] Distance threshold already computed, check code.";

	//float fov_x, fov_y;
	//float res_x, res_y;
	//float thresh_x, thresh_y;

	//fov_x = 2 * atan2f(img_width, (2 * intri_.fx));
	//fov_y = 2 * atan2f(img_height, (2 * intri_.fy));

	//res_x = fov_x / img_width;
	//res_y = fov_y / img_height;

	//thresh_x = conf_.err_tolr_x / res_x;
	//thresh_y = conf_.err_tolr_y / res_y;

	//dist_thresh_ = thresh_x > thresh_y ? thresh_x : thresh_y;

	dist_thresh_ = misc::DistThreshCompute(intri_, img_width, img_height, conf_.err_tolr_x, conf_.err_tolr_y);

	LOG(INFO) << "[PhotoCaliOnline] Distance threshold set to " << dist_thresh_;
}

visc::PtIndices PhotoCaliOnline::Impl::InterestPtId2ViscPtIndices() {
	CHECK(!interest_pt_id_.empty());

	visc::PtIndices res;
	res.indices.reserve(interest_pt_id_.size());

	for (const PtId_t& id : interest_pt_id_) {
		res.indices.push_back(id);
	}

	return res;
}

PhotoCaliOnline::PhotoCaliOnline() : impl_ (new Impl) {
}

PhotoCaliOnline::PhotoCaliOnline(const OnlineCaliConf& conf) : impl_(new Impl(conf)) {

}

PhotoCaliOnline::~PhotoCaliOnline(){
}

void PhotoCaliOnline::SetInitExtri(const Eigen::Isometry3f& T_cl) {
	impl_->T_cl_ = T_cl;
}

void PhotoCaliOnline::SetCamIntrinsics(float fx, float fy, float cx, float cy) {
	CamIntri& intri = impl_->intri_;

	intri.fx = fx;
	intri.fy = fy;
	intri.cx = cx;
	intri.cy = cy;
}

void PhotoCaliOnline::AddDataFrame(const PtCloudXYZI_t::Ptr& ptcloud_ptr, const cv::Mat& img, const Eigen::Isometry3f& T_wl) {
	CHECK_NE(ptcloud_ptr, nullptr);
	CHECK(!img.empty());

	DataFrame f;
	f.ptcloud = ptcloud_ptr;
	f.img = img;
	f.T_wl = T_wl;

	// Avoid dereferencing same pointer multiple times
	Impl& impl = *impl_;

	// If not set, compute & set thresh
	if (impl.dist_thresh_ < 0.0f) {
		impl.DistThreshCompute(img.cols, img.rows);
	}

	// Ptcloud clip
	utils::RemoveCloudPtsOutOfRange(*f.ptcloud, impl.conf_.ptcloud_clip_min, impl.conf_.ptcloud_clip_max);

	// Add to window
	impl.data_window_.push_back(f);

	// Return if window not fully filled
	if (impl.data_window_.size() < impl.conf_.window_size) {
		LOG(INFO) << "[PhotoCaliOnline] Current data number: " << impl.data_window_.size()
			<< " Window size: " << impl.conf_.window_size;
		return;
	}

	LOG(INFO) << "[PhotoCaliOnline] Adding data to global data pool";

	// Clear local cloud and points of interest since they are temporary
	impl.cloud_local_->clear();
	impl.interest_pt_id_.clear();

	// Build current global cloud
	// impl.ConcatCloudInWindow(*impl.cloud_local_);
	misc::ConcatPointClouds(*impl.cloud_local_, impl.data_window_);

	// Random sampling
	// impl.RandSampleInterestPts(impl.interest_pt_id_, *impl.cloud_local_, impl.conf_.sample_ratio);
	utils::RandSampleInterestPts(impl.interest_pt_id_, *impl.cloud_local_, impl.conf_.sample_ratio);

	// Visibility rejection
	CovisGraph cg_local;
	impl.ComputeCovisInfo(cg_local);
	cg_local.EraseLessObservedPt(impl.conf_.obs_thresh); // Remove points less observed

	CHECK(!cg_local.Empty());

	// Add to global data pool
	impl.AddCovisDataPool(cg_local);

	LOG(INFO) << "[PhotoCaliOnline] Data pool status: ptcloud size = " << impl.cd_pool_.CloudSize()
		<< ", image num = " << impl.cd_pool_.ImgNum();

	// Clear window after data transfered to global data pool
	impl.data_window_.clear();

	// Compute weights for covis graph
	impl.ComputeWeightsOnGraph();

	// Generate new point id mask
	impl.global_cg_.GenTransWeightMask(impl.trans_mask_);

#if 0
	// Optimization start criteria
	if (impl.cd_pool_.ImgNum() < impl.conf_.opt_conf.start_frame_num) {
		return;
	}

	if (impl.conf_.mode == OnlineCaliMode::kManualCost) {
		LOG(INFO) << "[PhotoCaliOnline] Manual cost computation start";

		ManualPhotoErr man_err(impl.cd_pool_, impl.global_cg_, impl.intri_);
		man_err.SetExtri(impl.T_cl_);
		man_err.SetPyramidLevel(impl.conf_.opt_conf.pyramid_lvl);
		man_err.SetTransMask(impl.trans_mask_);

		std::vector<float> costs_trans;
		std::vector<float> costs_rot;
		man_err.Compute(costs_rot, costs_trans);

		if (impl.recorder_) {
			impl.recorder_->costs_rot = costs_rot;
			impl.recorder_->costs_trans = costs_trans;
		}
	}
	else if (impl.conf_.mode == OnlineCaliMode::kOptimize) {
		LOG(INFO) << "[PhotoCaliOnline] Optimization start";

		utils::StopWatch sw;

		PhotoOptimzerPyr optimizer(impl.cd_pool_, impl.global_cg_, 
			impl.intri_, impl.conf_.opt_conf, impl.T_cl_, impl.trans_mask_);

		optimizer.Optimize(impl.T_cl_);

		LOG(INFO) << "[PhotoCaliOnline] Optimization time: " << sw.GetTimeElapse();
	}
#endif // 0
}

void PhotoCaliOnline::Calibrate() {
	Impl& impl = *impl_;

	// LOG(INFO) << "[PhotoCaliOnline] Grid search start";
	utils::StopWatch sw;

	// std::vector<Eigen::Isometry3f> extri_grid;
	// utils::GenGridIsometry3f(extri_grid, impl.T_cl_, 1, 0.5f, 0.05f);

	// LOG(INFO) << "[PhotoCaliOnline] Grid size: " << extri_grid.size();

	// impl.GridSearchExtri(impl.T_cl_, extri_grid);

	// LOG(INFO) << "[PhotoCaliOnline] Grid search done, time used: " << sw.GetTimeElapse();

	LOG(INFO) << "[PhotoCaliOnline] Optimization start";

	PhotoOptimzerPyr optimizer(impl.cd_pool_, impl.global_cg_,
	impl.intri_, impl.conf_.opt_conf, impl.T_cl_, impl.trans_mask_);

	optimizer.Optimize(impl.T_cl_);

	LOG(INFO) << "[PhotoCaliOnline] Optimization time: " << sw.GetTimeElapse();
}

phc::PtCloudXYZI_t::Ptr PhotoCaliOnline::GetVisPtCloud() {
	return impl_->cd_pool_.GetCloud();
}

void PhotoCaliOnline::SetDataRecorder(utils::DataRecorder::Ptr recorder_ptr) {
	CHECK(recorder_ptr);
	impl_->recorder_ = recorder_ptr;
}

void PhotoCaliOnline::GetExtri(Eigen::Isometry3f& T_cl) const {
	T_cl = impl_->T_cl_;
}

void PhotoCaliOnline::GetCamIntri(Eigen::Matrix3f& intri_mat) const {
	intri_mat = impl_->intri_.AsMat();
}

void PhotoCaliOnline::GenCovisDataBag(const std::string& name) const{
	using utils::CovisDataBag;
	using Eigen::Vector3f;
	using Eigen::Vector2f;
	using Eigen::Matrix3f;
	using Eigen::Isometry3f;

	CovisDataBag::Ptr bag(new CovisDataBag(name));
	m2d::DataManager::Add(bag);

	const CovisDataPool& pool = impl_->cd_pool_;
	const CovisGraph& graph = impl_->global_cg_;
	const Matrix3f& intri_mat = impl_->intri_.AsMat();

	bag->SetPtcloud(pool.GetCloud());

	std::vector<FrameId_t> f_ids;
	graph.GetAllFrames(f_ids);
	std::sort(f_ids.begin(), f_ids.end());
	for (const FrameId_t &f_id : f_ids) {
		std::unordered_set<PtId_t> vis_pts;
		graph.FindAllVisiblePts(f_id, vis_pts);

		ImgWithPose_t img_pose = pool.GetImgWithPose(f_id);
		cv::Mat &img = img_pose.first;
		Isometry3f T_cw = img_pose.second;

		cv::Mat img_vis;
		cv::cvtColor(img_pose.first, img_vis, cv::COLOR_GRAY2BGR);
		for (const PtId_t &p_id : vis_pts) {
			PtXYZI_t pt = pool.GetPt(p_id);

			Vector2f pixel;
			utils::ProjectPoint(Vector3f(pt.x, pt.y, pt.z), pixel, T_cw, intri_mat);
			float val = utils::GetSubPixelValBilinear(img, pixel);
			utils::ImgMarkPixel(img_vis, pixel, std::to_string(val));
		}

		bag->AddImg(img_vis);
	}

	std::vector<PtId_t> pt_ids;
	std::vector<PtId_t> pt_ids_sample;
	graph.GetAllPts(pt_ids);
	std::sample(pt_ids.begin(), pt_ids.end(), std::back_inserter(pt_ids_sample), 10, utils::Rand::GetMt19937());
	for (const PtId_t &pt_id : pt_ids_sample) {
		std::vector<FrameId_t> vis_frames;
		graph.FindAllVisibleFrames(pt_id, vis_frames);

		PtXYZI_t pt = pool.GetPt(pt_id);

		std::vector<cv::Vec2i> pixel_loc;
		std::vector<float> pixel_val;
		std::vector<cv::Mat> vis_imgs;

		for (const FrameId_t& f_id : vis_frames) {
			ImgWithPose_t img_pose = pool.GetImgWithPose(f_id);
			cv::Mat& img = img_pose.first;
			Isometry3f T_cw = img_pose.second;

			cv::Mat img_vis;
			cv::cvtColor(img_pose.first, img_vis, cv::COLOR_GRAY2BGR);

			Vector2f pixel;
			utils::ProjectPoint(Vector3f(pt.x, pt.y, pt.z), pixel, T_cw, intri_mat);
			float val = utils::GetSubPixelValBilinear(img, pixel);
			utils::ImgMarkPixel(img_vis, pixel, "");

			pixel_loc.push_back(cv::Vec2i(pixel.x(), pixel.y()));
			pixel_val.push_back(val);
			vis_imgs.push_back(img_vis);
		}

		bag->AddPtDetail(pixel_loc, pixel_val, vis_imgs);
	}
}

void PhotoCaliOnline::GenPyrGradDataBag() const {
	using utils::PyrGradientDataBag;
	using Eigen::Matrix3f;
	using Eigen::Vector2f;
	using Eigen::Vector3f;
	using Eigen::Isometry3f;

	PyrGradientDataBag::Ptr bag(new PyrGradientDataBag("pyr_grad_data"));
	m2d::DataManager::Add(bag);

	const CovisDataPool& pool = impl_->cd_pool_;
	const CovisGraph& graph = impl_->global_cg_;
	const Matrix3f& intri_mat = impl_->intri_.AsMat();

	std::vector<FrameId_t> f_ids;
	graph.GetAllFrames(f_ids);

	std::vector<FrameId_t> sample_f_ids;
	std::sample(f_ids.begin(), f_ids.end(), 
		std::back_inserter(sample_f_ids), 20, utils::Rand::GetMt19937());
	
	// Each sample frame
	for (const FrameId_t& f_id : sample_f_ids) {
		std::unordered_set<PtId_t> vis_pts;
		graph.FindAllVisiblePts(f_id, vis_pts);

		ImgWithPose_t img_pose = pool.GetImgWithPose(f_id);
		Isometry3f T_cw = img_pose.second;

		PyrGradientDataBag::PyrGradDataType pyr_grad;
		pyr_grad.reserve(impl_->conf_.opt_conf.pyramid_lvl);

		PyrGradientDataBag::PyrImgDataType pyr_imgs;
		pyr_imgs.reserve(impl_->conf_.opt_conf.pyramid_lvl);

		// Each pyramid level
		for (int lvl = 0; lvl <= impl_->conf_.opt_conf.pyramid_lvl; ++lvl) {
			cv::Mat lvl_img;
			ImgPyramid::ImgPyrDown(img_pose.first, lvl_img, lvl);

			Matrix3f lvl_intri;
			ImgPyramid::IntriPyrDown(intri_mat, lvl_intri, lvl);

			cv::Mat lvl_img_vis;
			cv::resize(lvl_img, lvl_img_vis, img_pose.first.size());
			cv::cvtColor(lvl_img_vis, lvl_img_vis, cv::COLOR_GRAY2BGR);

			std::vector<float> grad_norm;
			grad_norm.reserve(vis_pts.size());

			// Each point
			for (const PtId_t& p_id : vis_pts) {
				PtXYZI_t pt = pool.GetPt(p_id);

				Vector2f pixel;
				utils::ProjectPoint(Vector3f(pt.x, pt.y, pt.z), pixel, T_cw, lvl_intri);

				Vector2f gradient = utils::SobelGradientAtSubPixel(lvl_img, pixel);

				grad_norm.push_back(gradient.norm());

				Vector2f pixel_raw; // Pixel on raw image (level 0)
				utils::ProjectPoint(Vector3f(pt.x, pt.y, pt.z), pixel_raw, T_cw, intri_mat);

				utils::ImgMarkPixel(lvl_img_vis, pixel_raw, "");
			}

			pyr_grad.push_back(grad_norm);
			pyr_imgs.push_back(lvl_img_vis);
		}

		bag->AddData(pyr_imgs, pyr_grad);
	}
}

void PhotoCaliOnline::GenErrCompDataBag(const std::string& name, const Eigen::Isometry3f& err_T_cl) const {
	using utils::ErrCompDataBag;
	using Eigen::Vector3f;
	using Eigen::Vector2f;
	using Eigen::Matrix3f;
	using Eigen::Isometry3f;

	ErrCompDataBag::Ptr bag(new ErrCompDataBag(name));
	m2d::DataManager::Add(bag);

	const CovisDataPool& pool = impl_->cd_pool_;
	const CovisGraph& graph = impl_->global_cg_;
	const Matrix3f& intri_mat = impl_->intri_.AsMat();

	std::vector<PtId_t> pt_ids;
	std::vector<PtId_t> pt_ids_sample;
	graph.GetAllPts(pt_ids);
	std::sample(pt_ids.begin(), pt_ids.end(), std::back_inserter(pt_ids_sample), 20, utils::Rand::GetMt19937());
	std::sort(pt_ids_sample.begin(), pt_ids_sample.end());

	// Origin detail
	for (const PtId_t& pt_id : pt_ids_sample) {
		std::vector<FrameId_t> vis_frames;
		graph.FindAllVisibleFrames(pt_id, vis_frames);
		std::sort(vis_frames.begin(), vis_frames.end());

		PtXYZI_t pt = pool.GetPt(pt_id);

		std::vector<cv::Vec2i> pixel_loc;
		std::vector<float> pixel_val;
		std::vector<cv::Mat> vis_imgs;

		for (const FrameId_t& f_id : vis_frames) {
			ImgWithPose_t img_pose = pool.GetImgWithPose(f_id);
			cv::Mat& img = img_pose.first;
			Isometry3f T_cw = img_pose.second;

			cv::Mat img_vis;
			cv::cvtColor(img_pose.first, img_vis, cv::COLOR_GRAY2BGR);

			Vector2f pixel;
			utils::ProjectPoint(Vector3f(pt.x, pt.y, pt.z), pixel, T_cw, intri_mat);
			float val = utils::GetSubPixelValBilinear(img, pixel);
			// utils::ImgMarkPixel(img_vis, pixel, "");

			pixel_loc.push_back(cv::Vec2i(pixel.x(), pixel.y()));
			pixel_val.push_back(val);
			vis_imgs.push_back(img_vis);
		}

		bag->AddOriginDetail(pixel_loc, pixel_val, vis_imgs);
	}

	// Error detail
	for (const PtId_t& pt_id : pt_ids_sample) {
		std::vector<FrameId_t> vis_frames;
		graph.FindAllVisibleFrames(pt_id, vis_frames);
		std::sort(vis_frames.begin(), vis_frames.end());

		PtXYZI_t pt = pool.GetPt(pt_id);

		std::vector<cv::Vec2i> pixel_loc;
		std::vector<float> pixel_val;
		std::vector<cv::Mat> vis_imgs;

		for (const FrameId_t& f_id : vis_frames) {
			ImgWithPose_t img_pose = pool.GetImgWithPose(f_id);
			cv::Mat& img = img_pose.first;
			// T_lc * T_cw = T_lw
			Isometry3f T_cw = err_T_cl * impl_->T_cl_.inverse() * img_pose.second;

			cv::Mat img_vis;
			cv::cvtColor(img_pose.first, img_vis, cv::COLOR_GRAY2BGR);

			Vector2f pixel;
			utils::ProjectPoint(Vector3f(pt.x, pt.y, pt.z), pixel, T_cw, intri_mat);
			float val = utils::GetSubPixelValBilinear(img, pixel);
			// utils::ImgMarkPixel(img_vis, pixel, "");

			pixel_loc.push_back(cv::Vec2i(pixel.x(), pixel.y()));
			pixel_val.push_back(val);
			vis_imgs.push_back(img_vis);
		}

		bag->AddErrDetail(pixel_loc, pixel_val, vis_imgs);
	}
}