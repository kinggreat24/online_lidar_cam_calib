#include <random>

#include <glog/logging.h>

#include <pcl/common/transforms.h>
#include <pcl/common/io.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>

#include "vis_check/VisCheck.hpp"

#include "PhotoCali.hpp"
#include "StopWatch.hpp"
#include "Utils.hpp"
#include "Config.hpp"
#include "Optimizer.hpp"
#include "ImgPyramid.hpp"

using phc::PhotoCali;
using phc::DataFrame;
using phc::PtCloudXYZI_t;
using phc::PtCloudXYZRGB_t;
using phc::CovisGraph;
using phc::CamIntri;
using phc::utils::Config;

PhotoCali::PhotoCali() {
	global_cloud_ = PtCloudXYZI_t::Ptr(new PtCloudXYZI_t);
}

void PhotoCali::AddDataFrame(const PtCloudXYZI_t::Ptr& ptcloud_ptr, const cv::Mat& img, const Eigen::Isometry3f& T_wl) {
	CHECK(ptcloud_ptr != nullptr);
	CHECK(!img.empty());

	frames_.push_back(DataFrame{ ptcloud_ptr, img, T_wl });
}

void PhotoCali::GetConcatPtcloud(PtCloudXYZI_t& cloud_out) {
	cloud_out.clear();
	for (const auto &f : frames_) {
		const Eigen::Isometry3f& T_wl = f.T_wl;
		PtCloudXYZI_t cloud_cp;
		pcl::copyPointCloud(*f.ptcloud, cloud_cp);

		// Transform to world frame and concat
		pcl::transformPointCloud(cloud_cp, cloud_cp, T_wl.matrix());
		cloud_out += cloud_cp;
	}
}

void PhotoCali::SetInterestPtindices(const std::vector<pcl::index_t>& pts_indices) {
	for (const auto &idx : pts_indices) {
		interest_pts_.insert(idx);
	}
	LOG(INFO) << "Interest points set, RandSampleRatio will be ignored";
}

void PhotoCali::SetRandSampleRatio(const float& r) {
	float tmp = r;
	if (r > 1.0f) {
		LOG(WARNING) << "Desired sample ratio > 1.0f, turncated to 1.0f";
		tmp = 1.0f;
	}
	else if (r < 0.0f) {
		LOG(WARNING) << "Desired sample ratio < 0.0f, no sampling will be performed";
		tmp = -1.0f;
	}

	rand_sample_ratio_ = tmp;
	LOG(INFO) << "Sample ratio set to " << rand_sample_ratio_;
}

void PhotoCali::SetCamIntri(float fx, float fy, float cx, float cy) {
	intri_.fx = fx;
	intri_.fy = fy;
	intri_.cx = cx;
	intri_.cy = cy;
}

void PhotoCali::SetExtri(const Eigen::Isometry3f& T_cl) {
	T_cl_ = T_cl;
}

PtCloudXYZI_t::ConstPtr PhotoCali::GetGlobCloud() const {
	CHECK(global_cloud_ != nullptr);
	return global_cloud_;
}

void PhotoCali::RangeFilterOnAllPtcloud(float min, float max) {
	utils::StopWatch sw;

	for (auto &f : frames_) {
		utils::RemoveCloudPtsOutOfRange(*f.ptcloud, min, max);
	}

	LOG(INFO) << "RangeFilterOnAllPtcloud time used: " << sw.GetTimeElapse();
}

void PhotoCali::ComputeCovisInfo(CovisGraph& covis_g) {
	using pcl::PointCloud;
	using pcl::PointXYZ;

	// Clear all info
	covis_g.Clear();

	PointCloud<PointXYZ>::Ptr cloud_xyz(new PointCloud<PointXYZ>);
	pcl::copyPointCloud(*global_cloud_, *cloud_xyz);
	LOG(INFO) << "Point cloud size for VisCheck: " << cloud_xyz->size();

	utils::StopWatch sw;

	visc::VisCheck vis_checker;
	vis_checker.SetInputCloud(cloud_xyz);
	vis_checker.SetMaxViewRange(Config::Get<float>("covis_check.max_view_range"));
	vis_checker.SetVisScoreThreshMeanShift(Config::Get<float>("covis_check.mean_shift"));
	vis_checker.SetVisScoreThresh(Config::Get<float>("covis_check.score_thresh"));
	vis_checker.SetDiscardEdgeSize(Config::Get<int>("covis_check.edge_size_discard"));

	visc::CamIntrinsics intri{intri_.fx, intri_.fy, intri_.cx, intri_.cy};

	for (size_t i = 0; i < frames_.size(); ++i) {
		const DataFrame& f = frames_.at(i);

		LOG(INFO) << "Computing visibility info for frame " << i;

		Eigen::Isometry3f T_cw = T_cl_ * f.T_wl.inverse();
		vis_checker.SetCamera(intri, T_cw, f.img.cols, f.img.rows);

		visc::PtIndices res;
		vis_checker.ComputeVisibility(res);

		for (const pcl::index_t &idx : res.indices) {
			covis_g.Insert(i, idx);
		}
	}

	LOG(INFO) << "ComputeCovisInfo time used: " << sw.GetTimeElapse();
}

float PhotoCali::ComputeCost() {
	if (global_cloud_->empty()) {
		// Range filtering on all ptcloud and then concat
		RangeFilterOnAllPtcloud(Config::Get<float>("cloud_range_filter.min_range"), Config::Get<float>("cloud_range_filter.max_range"));
		GetConcatPtcloud(*global_cloud_);
		CloudFiltering(global_cloud_);
	}

	// Compute covisibility for frames and points
	ComputeCovisInfo(cov_graph_);
	if (recorder_) { // Record visible ptcloud if a recorder is attached
		std::vector<PtCloudXYZRGB_t::Ptr> vis_clouds;
		GetVisibleClouds(vis_clouds);
		recorder_->AddVisibleCloudForFrames(vis_clouds);
	}

	// Erase points with less visible frame number
	cov_graph_.EraseLessObservedPt(Config::Get<int>("covis_graph.obs_thresh"));

	// Sample points if desired
	std::vector<PtId_t> pts;
	cov_graph_.GetAllPts(pts);
	DownSamplePts(pts);

	// Compute cost
	utils::StopWatch sw;
	float total_cost = 0.0f;
	for (const auto &pt : pts) {
		if (recorder_) { recorder_->StartNewPtSession(); }

		std::unordered_set<FrameId_t> vis_frames;
		cov_graph_.FindAllVisibleFrames(pt, vis_frames);

		float single_cost = SinglePtCost(pt, vis_frames);
		total_cost += single_cost;

		if (recorder_) {
			recorder_->PtSessionSetGlobPtIdx(pt);
			recorder_->PtSessionSetCost(single_cost);
		}
	}

	LOG(INFO) << "Total cost computing time: " << sw.GetTimeElapse();
	LOG(INFO) << "Number of points computed: " << pts.size();

	return total_cost / static_cast<float>(pts.size());
}

Eigen::Vector2f PhotoCali::ComputeCostWeighted() {
	if (global_cloud_->empty()) {
		// Range filtering on all ptcloud and then concat
		RangeFilterOnAllPtcloud(Config::Get<float>("cloud_range_filter.min_range"), Config::Get<float>("cloud_range_filter.max_range"));
		GetConcatPtcloud(*global_cloud_);
		CloudFiltering(global_cloud_);
	}

	// Compute covisibility for frames and points
	ComputeCovisInfo(cov_graph_);

	// Erase points with less visible frame number
	cov_graph_.EraseLessObservedPt(Config::Get<int>("covis_graph.obs_thresh"));

	// Compute distance threshold
	int img_width = frames_.front().img.cols;
	int img_height = frames_.front().img.rows;
	float thresh = DistThreshCompute(img_width, img_height);

	LOG(INFO) << "Rotation thresh: " << thresh;
	LOG(INFO) << "Translation thresh: " << thresh / 2.0f;

	std::vector<PtId_t> pts;
	cov_graph_.GetAllPts(pts);
	// Compute cost
	utils::StopWatch sw;
	Eigen::Vector2f total_cost = Eigen::Vector2f::Zero();
	for (const auto& pt : pts) {

		std::unordered_set<FrameId_t> vis_frames;
		cov_graph_.FindAllVisibleFrames(pt, vis_frames);

		Eigen::Vector2f single_cost = SinglePtCostWeighted(pt, vis_frames, thresh);
		total_cost += single_cost;
	}

	LOG(INFO) << "Total cost computing time: " << sw.GetTimeElapse();
	LOG(INFO) << "Number of points computed: " << pts.size();

	return total_cost / static_cast<float>(pts.size());
}

float PhotoCali::SinglePtCost(const PtId_t& pt_id, const std::unordered_set<FrameId_t>& vis_frames) {
	Eigen::Vector3f pt_world(
		global_cloud_->at(pt_id).x,
		global_cloud_->at(pt_id).y,
		global_cloud_->at(pt_id).z);

	Eigen::VectorXf ph_vals(vis_frames.size());
	int num_cnt = 0;
	for (const auto &f_id : vis_frames) {
		Eigen::Isometry3f T_cw = T_cl_ * frames_[f_id].T_wl.inverse();
		Eigen::Vector2f pixel;
		utils::ProjectPoint(pt_world, pixel, T_cw, intri_.AsMat());

		// Photometric value
		float ph_val = utils::GetSubPixelValBilinear(frames_[f_id].img, pixel);
		ph_vals(num_cnt) = ph_val;
		num_cnt += 1;

		if (recorder_) {
			utils::ProjInfo p_info;
			p_info.img = frames_[f_id].img;
			p_info.sp_val_interp = ph_val;
			p_info.subpixel = pixel;
			p_info.frame_id = f_id;
			recorder_->PtSessionAddProjInfo(p_info);
		}
	}

	return utils::VarianceCompute(ph_vals);
}

Eigen::Vector2f PhotoCali::SinglePtCostWeighted(const PtId_t& pt_id, const std::unordered_set<FrameId_t>& vis_frames, float thresh) {
	Eigen::Vector3f pt_world(
		global_cloud_->at(pt_id).x,
		global_cloud_->at(pt_id).y,
		global_cloud_->at(pt_id).z);

	Eigen::Matrix3f intri_mat = intri_.AsMat();

	Eigen::VectorXf ph_vals(vis_frames.size());
	Eigen::VectorXf weights_rot(vis_frames.size());
	Eigen::VectorXf weights_trans(vis_frames.size());
	int num_cnt = 0;
	for (const auto& f_id : vis_frames) {
		Eigen::Isometry3f T_cw = T_cl_ * frames_[f_id].T_wl.inverse();
		Eigen::Vector3f pt_cam = T_cw * pt_world;
		Eigen::Vector2f pixel;
		utils::ProjectPoint(pt_cam, pixel, Eigen::Isometry3f::Identity(), intri_mat);

		// Weight computing
		float dist = pt_cam.norm(); // Eucliean distance to optical center
		weights_rot(num_cnt) = dist < thresh ? dist : thresh;
		weights_trans(num_cnt) = (dist < thresh / 2.0f) ? 1.0f : 0.0f;

		// Photometric value
		float ph_val = utils::GetSubPixelValBilinear(frames_[f_id].img, pixel);
		ph_vals(num_cnt) = ph_val;
		num_cnt += 1;

		if (recorder_) {
			utils::ProjInfo p_info;
			p_info.img = frames_[f_id].img;
			p_info.sp_val_interp = ph_val;
			p_info.subpixel = pixel;
			p_info.frame_id = f_id;
			recorder_->PtSessionAddProjInfo(p_info);
		}
	}

	return utils::VarianceComputeWeighted(ph_vals, weights_rot, weights_trans);
}

float PhotoCali::DistThreshCompute(int img_width, int img_height) {
	float fov_x, fov_y;
	float res_x, res_y;
	float thresh_x, thresh_y;

	fov_x = 2 * atan2f(img_width, (2 * intri_.fx));
	fov_y = 2 * atan2f(img_height, (2 * intri_.fy));

	res_x = fov_x / img_width;
	res_y = fov_y / img_height;

	thresh_x = err_tolerance_x_ / res_x;
	thresh_y = err_tolerance_y_ / res_y;

	return (thresh_x > thresh_y ? thresh_x : thresh_y);
}

void PhotoCali::DownSamplePts(std::vector<PtId_t>& pt_indices) {
	std::vector<PtId_t> sampled_pts;
	if (!interest_pts_.empty()) {
		LOG(INFO) << "Downsample points with given interest points, size: " << interest_pts_.size();
		for (const auto& idx : pt_indices) {
			if (interest_pts_.find(idx) != interest_pts_.end()) {
				sampled_pts.push_back(idx);
			}
		}
		pt_indices = sampled_pts;
	}
	else if (rand_sample_ratio_ > 0.0f) {
		int sampled_size = static_cast<int>(pt_indices.size() * rand_sample_ratio_);
		std::sample(pt_indices.begin(), pt_indices.end(), std::back_inserter(sampled_pts), sampled_size, std::mt19937{ std::random_device{}() });
		LOG(INFO) << "Downsample points with given random sample ratio: " << rand_sample_ratio_ << " Sampled points number: " << sampled_size;

		pt_indices = sampled_pts;
	}
	else {
		LOG(INFO) << "No downsample performed";
	}
}

PtCloudXYZRGB_t::Ptr PhotoCali::GetVisibleCloudForFrame(FrameId_t fid) const {
	CHECK(fid >= 0 && fid < frames_.size());
	CHECK(!cov_graph_.Empty());
	CHECK(global_cloud_);

	PtCloudXYZRGB_t::Ptr vis_cloud(new PtCloudXYZRGB_t);
	pcl::copyPointCloud(*global_cloud_, *vis_cloud);

	std::unordered_set<PtId_t> visible_pts;
	cov_graph_.FindAllVisiblePts(fid, visible_pts);

	pcl::Indices pt_indices;
	pt_indices.reserve(visible_pts.size());
	for (const PtId_t &pt : visible_pts) {
		pt_indices.push_back(pt);
	}

	utils::PtcloudSetColor(*vis_cloud, 255, 0, 0); // First color all point cloud in red
	utils::PtcloudSetColor(*vis_cloud, pt_indices, 0, 255, 0); // Then color visible point in green

	return vis_cloud;
}

void PhotoCali::GetVisibleClouds(std::vector<PtCloudXYZRGB_t::Ptr>& ptrs) const {
	ptrs.clear();
	ptrs.reserve(frames_.size());
	for (size_t fid = 0; fid < frames_.size(); ++fid) {
		ptrs.push_back(GetVisibleCloudForFrame(fid));
	}
}

void PhotoCali::Optimize(Eigen::Isometry3f& result) {
	global_cloud_->clear();
	RangeFilterOnAllPtcloud(Config::Get<float>("cloud_range_filter.min_range"), Config::Get<float>("cloud_range_filter.max_range"));
	GetConcatPtcloud(*global_cloud_);

	cov_graph_.Clear();
	ComputeCovisInfo(cov_graph_);
	cov_graph_.EraseLessObservedPt(Config::Get<int>("covis_graph.obs_thresh"));

	PhotometricOptimizer ph_opter;
	ph_opter.SetCameraIntri(intri_);
	ph_opter.SetInitExtri(T_cl_);
	
	for (const DataFrame &frame : frames_) {
		ph_opter.AddImgWithPose(frame.img, frame.T_wl.inverse());
	}

	std::vector<PtId_t> pts;
	cov_graph_.GetAllPts(pts);
	for (const auto &pt : pts) {
		std::unordered_set<FrameId_t> vis_frames;
		std::vector<FrameId_t> fv;
		cov_graph_.FindAllVisibleFrames(pt, vis_frames);
		fv.insert(fv.end(), vis_frames.begin(), vis_frames.end());

		pcl::PointXYZI pt_xyzi = global_cloud_->at(pt);
		Eigen::Vector3f pt_world(pt_xyzi.x, pt_xyzi.y, pt_xyzi.z);
		ph_opter.AddResidual(pt_world, fv);
	}

	// Remove residual too large
	std::vector<std::pair<ceres::ResidualBlockId, double>> id_res_pair;
	std::vector<ceres::ResidualBlockId> blk_to_remove;
	ph_opter.EvalAllResidualBlk(id_res_pair);
	GetBlkIdRemoval(blk_to_remove, id_res_pair, resblk_remove_percent_);
	ph_opter.RemoveResidualBlk(blk_to_remove);

	if (recorder_) {
		std::vector<double> residual;
		ph_opter.EvalAllResidualBlk(residual);
		recorder_->SetResidualBeforeOpt(residual);
	}

	ph_opter.Optimize(Config::Get<int>("optimizer.max_iter"));

	ph_opter.GetExtri(result);

	if (recorder_) {
		std::vector<double> residual;
		ph_opter.EvalAllResidualBlk(residual);
		recorder_->SetResidualAfterOpt(residual);
	}
}

void PhotoCali::OptimizeSingleLvlPyr(const Eigen::Isometry3f& init, Eigen::Isometry3f& result, unsigned int level) {
	if (global_cloud_->empty()) {
		RangeFilterOnAllPtcloud(Config::Get<float>("cloud_range_filter.min_range"), Config::Get<float>("cloud_range_filter.max_range"));
		GetConcatPtcloud(*global_cloud_);
		CloudFiltering(global_cloud_);
	}

	if (cov_graph_.Empty()) {
		ComputeCovisInfo(cov_graph_);
		cov_graph_.EraseLessObservedPt(Config::Get<int>("covis_graph.obs_thresh"));
	}

	LOG(INFO) << "Optimizing on pyramid level " << level;
	LOG(INFO) << "Image size: " << frames_.front().img.rows / pow(2, level) << "x" << frames_.front().img.cols / pow(2, level);

	// Compute intrinsics on this level
	Eigen::Matrix3f level_intri;
	ImgPyramid::IntriPyrDown(intri_.AsMat(), level_intri, level);

	// Construct optimizer
	PhotometricOptimizer ph_opter;
	ph_opter.SetCameraIntri(CamIntri::FromMat(level_intri));
	ph_opter.SetInitExtri(init);

	// Populate images with pose
	for (const DataFrame& frame : frames_) {
		cv::Mat level_img;
		ImgPyramid::ImgPyrDown(frame.img, level_img, level);
		ph_opter.AddImgWithPose(level_img, frame.T_wl.inverse());
	}

	// Populate covisibility info
	std::vector<PtId_t> pts;
	cov_graph_.GetAllPts(pts);
	for (const auto& pt : pts) {
		std::unordered_set<FrameId_t> vis_frames;
		std::vector<FrameId_t> fv;
		cov_graph_.FindAllVisibleFrames(pt, vis_frames);
		fv.insert(fv.end(), vis_frames.begin(), vis_frames.end());

		pcl::PointXYZI pt_xyzi = global_cloud_->at(pt);
		Eigen::Vector3f pt_world(pt_xyzi.x, pt_xyzi.y, pt_xyzi.z);
		ph_opter.AddResidual(pt_world, fv);
	}

	// Remove residual too large
	std::vector<std::pair<ceres::ResidualBlockId, double>> id_res_pair;
	std::vector<ceres::ResidualBlockId> blk_to_remove;
	ph_opter.EvalAllResidualBlk(id_res_pair);
	GetBlkIdRemoval(blk_to_remove, id_res_pair, resblk_remove_percent_);
	ph_opter.RemoveResidualBlk(blk_to_remove);

	if (recorder_) {
		std::vector<double> residual;
		ph_opter.EvalAllResidualBlk(residual);
		recorder_->SetResidualBeforeOpt(residual);
	}

	ph_opter.Optimize(Config::Get<int>("optimizer.max_iter"));

	ph_opter.GetExtri(result);

	if (recorder_) {
		std::vector<double> residual;
		ph_opter.EvalAllResidualBlk(residual);
		recorder_->SetResidualAfterOpt(residual);
	}
}

void PhotoCali::OptimizeSingleLvlPyr(Eigen::Isometry3f& result, unsigned int level) {
	OptimizeSingleLvlPyr(T_cl_, result, level);
}

void PhotoCali::OptimizeMultiLvlPyr(Eigen::Isometry3f& result, unsigned int pyr_depth) {
	if (global_cloud_->empty()) {
		RangeFilterOnAllPtcloud(Config::Get<float>("cloud_range_filter.min_range"), Config::Get<float>("cloud_range_filter.max_range"));
		GetConcatPtcloud(*global_cloud_);
	}

	if (cov_graph_.Empty()) {
		ComputeCovisInfo(cov_graph_);
		cov_graph_.EraseLessObservedPt(Config::Get<int>("covis_graph.obs_thresh"));
	}

	Eigen::Isometry3f temp = T_cl_;
	for (int lvl = pyr_depth - 1; lvl >= 0; --lvl) {
		Eigen::Isometry3f res;
		OptimizeSingleLvlPyr(temp, res, lvl);

		temp = res;
	}

	result = temp;
}

void PhotoCali::GetBlkIdRemoval(std::vector<ceres::ResidualBlockId>& removal_ids, const std::vector<std::pair<ceres::ResidualBlockId, double>>& id_res_pairs, double percent) {
	using pair_type = std::vector<std::pair<ceres::ResidualBlockId, double>>::value_type;

	CHECK(percent > 0.0 && percent <= 1.0);

	LOG(INFO) << "Residual block remove percent: " << percent;
	LOG(INFO) << "Residual block numeber before removing: " << id_res_pairs.size();

	std::vector<std::pair<ceres::ResidualBlockId, double>> sort_pairs = id_res_pairs;
	std::sort(sort_pairs.begin(), sort_pairs.end(),
		[](const pair_type &p1, const pair_type &p2) {
			return p1.second < p2.second;
		});

	int thresh_idx = static_cast<int>(sort_pairs.size() * percent);
	LOG(INFO) << "Residual removing thresh: " << sort_pairs[thresh_idx].second;

	removal_ids.reserve(thresh_idx);
	for (int i = thresh_idx; i < sort_pairs.size(); ++i) {
		removal_ids.push_back(sort_pairs[i].first);
	}

	LOG(INFO) << "Residual block to remove: " << removal_ids.size();
}

void PhotoCali::CloudFiltering(PtCloudXYZI_t::Ptr cloud) {

	LOG(INFO) << "Ptcloud before filtering: " << cloud->size();

	PtCloudXYZI_t::Ptr filtered_cloud(new PtCloudXYZI_t);

	pcl::StatisticalOutlierRemoval<pcl::PointXYZI> sor;
	sor.setInputCloud(cloud);
	sor.setMeanK(50);
	sor.setStddevMulThresh(1.0);
	sor.filter(*filtered_cloud);

	pcl::VoxelGrid<pcl::PointXYZI> vgf;
	vgf.setInputCloud(filtered_cloud);
	vgf.setLeafSize(0.1f, 0.1f, 0.1f);
	vgf.filter(*filtered_cloud);

	LOG(INFO) << "Ptcloud after filtering: " << filtered_cloud->size();

	cloud->swap(*filtered_cloud);
}

void PhotoCali::OptimizeSingleLvlPyrWeightedVar(Eigen::Isometry3f& result, unsigned int level) {
	OptimizeSingleLvlPyrWeightedVar(T_cl_, result, level);
}

void PhotoCali::OptimizeSingleLvlPyrWeightedVar(const Eigen::Isometry3f& init, Eigen::Isometry3f& result, unsigned int level) {
	if (global_cloud_->empty()) {
		RangeFilterOnAllPtcloud(Config::Get<float>("cloud_range_filter.min_range"), Config::Get<float>("cloud_range_filter.max_range"));
		GetConcatPtcloud(*global_cloud_);
		CloudFiltering(global_cloud_);
	}

	if (cov_graph_.Empty()) {
		ComputeCovisInfo(cov_graph_);
		cov_graph_.EraseLessObservedPt(Config::Get<int>("covis_graph.obs_thresh"));
	}

	LOG(INFO) << "Optimizing on pyramid level " << level;
	LOG(INFO) << "Image size: " << frames_.front().img.rows / pow(2, level) << "x" << frames_.front().img.cols / pow(2, level);

	// Compute intrinsics on this level
	Eigen::Matrix3f level_intri;
	ImgPyramid::IntriPyrDown(intri_.AsMat(), level_intri, level);

	// Construct optimizer
	PhotometricOptimizerTransRotSepWeightedVar ph_opter;
	float trans_thresh_factor = Config::Get<float>("optimizer.trans_thresh_factor");
	ph_opter.SetWeightThresh(DistThreshCompute(frames_.front().img.cols, frames_.front().img.rows), trans_thresh_factor);
	ph_opter.SetCameraIntri(CamIntri::FromMat(level_intri));
	ph_opter.SetInitExtri(init);

	// Populate images with pose
	for (const DataFrame& frame : frames_) {
		cv::Mat level_img;
		ImgPyramid::ImgPyrDown(frame.img, level_img, level);
		ph_opter.AddImgWithPose(level_img, frame.T_wl.inverse());
	}

	// Populate covisibility info
	std::vector<PtId_t> pts;
	cov_graph_.GetAllPts(pts);
	for (const auto& pt : pts) {
		std::unordered_set<FrameId_t> vis_frames;
		std::vector<FrameId_t> fv;
		cov_graph_.FindAllVisibleFrames(pt, vis_frames);
		fv.insert(fv.end(), vis_frames.begin(), vis_frames.end());

		pcl::PointXYZI pt_xyzi = global_cloud_->at(pt);
		Eigen::Vector3f pt_world(pt_xyzi.x, pt_xyzi.y, pt_xyzi.z);
		ph_opter.AddResidual(pt_world, fv);
	}

	// Remove residual too large
	// std::vector<std::pair<ceres::ResidualBlockId, double>> id_res_pair;
	// std::vector<ceres::ResidualBlockId> blk_to_remove;
	// ph_opter.EvalAllResidualBlk(id_res_pair);
	// GetBlkIdRemoval(blk_to_remove, id_res_pair, resblk_remove_percent_);
	// ph_opter.RemoveResidualBlk(blk_to_remove);

	// if (recorder_) {
	// 	std::vector<double> res_rot, res_trans;
	// 	ph_opter.EvalRotResidualBlks(res_rot);
	// 	ph_opter.EvalTransResidualBlks(res_trans);
	// 	recorder_->SetNewResidualInfo(res_rot, "Rotation residual, before optimization");
	// 	recorder_->SetNewResidualInfo(res_trans, "Translation residual, before optimization");
	// }

	ph_opter.Optimize(Config::Get<int>("optimizer.max_iter"));

	// if (recorder_) {
	// 	std::vector<double> res_rot, res_trans;
	// 	ph_opter.EvalRotResidualBlks(res_rot);
	// 	ph_opter.EvalTransResidualBlks(res_trans);
	// 	recorder_->SetNewResidualInfo(res_rot, "Rotation residual, after optimization");
	// 	recorder_->SetNewResidualInfo(res_trans, "Translation residual, after optimization");
	// }

	ph_opter.GetExtri(result);
}

void PhotoCali::OptimizeSingleLvlPyrRepeated(Eigen::Isometry3f& result, unsigned int level) {
	OptimizeSingleLvlPyrRepeated(T_cl_, result, level);
}

void PhotoCali::OptimizeSingleLvlPyrRepeated(const Eigen::Isometry3f& init, Eigen::Isometry3f& result, unsigned int level) {
	if (global_cloud_->empty()) {
		RangeFilterOnAllPtcloud(Config::Get<float>("cloud_range_filter.min_range"), Config::Get<float>("cloud_range_filter.max_range"));
		GetConcatPtcloud(*global_cloud_);
		CloudFiltering(global_cloud_);
	}

	if (cov_graph_.Empty()) {
		ComputeCovisInfo(cov_graph_);
		cov_graph_.EraseLessObservedPt(Config::Get<int>("covis_graph.obs_thresh"));
	}

	LOG(INFO) << "Optimizing on pyramid level " << level;
	LOG(INFO) << "Image size: " << frames_.front().img.rows / pow(2, level) << "x" << frames_.front().img.cols / pow(2, level);

	// Compute intrinsics on this level
	Eigen::Matrix3f level_intri;
	ImgPyramid::IntriPyrDown(intri_.AsMat(), level_intri, level);

	// Construct optimizer
	PhOptTransRotSepRepeated ph_opter;
	float trans_thresh_factor = Config::Get<float>("optimizer.trans_thresh_factor");
	ph_opter.SetWeightThresh(DistThreshCompute(frames_.front().img.cols, frames_.front().img.rows), trans_thresh_factor);
	ph_opter.SetCameraIntri(CamIntri::FromMat(level_intri));
	ph_opter.SetInitExtri(init);

	// Populate images with pose
	for (const DataFrame& frame : frames_) {
		cv::Mat level_img;
		ImgPyramid::ImgPyrDown(frame.img, level_img, level);
		ph_opter.AddImgWithPose(level_img, frame.T_wl.inverse());
	}

	// ph_opter.OptimizeRotation(20, cov_graph_, *global_cloud_);
	ph_opter.OptimizeTranslation(20, cov_graph_, *global_cloud_);

	// if (recorder_) {
	// 	std::vector<double> res_rot, res_trans;
	// 	ph_opter.EvalRotResidualBlks(res_rot);
	// 	ph_opter.EvalTransResidualBlks(res_trans);
	// 	recorder_->SetNewResidualInfo(res_rot, "Rotation residual, after optimization");
	// 	recorder_->SetNewResidualInfo(res_trans, "Translation residual, after optimization");
	// }

	ph_opter.GetExtri(result);
}