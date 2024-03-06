#include <thread>

#include <glog/logging.h>

#include "ImgPyramid.hpp"
#include "Optimizer.hpp"
#include "BASolver.hpp"

using phc::CovisGraph;
using phc::PhotometricOptimizer;
using phc::PhotometricOptimizerTransRotSepWeightedVar;
using phc::PhOptTransRotSepRepeated;
using phc::PhotometricError;
using phc::PhotometricErrorRotOnly;
using phc::PhotometricErrorTransOnly;
using phc::PhotoOptimzerPyr;

PhotometricOptimizer::PhotometricOptimizer() {
	PhotometricError::Clear();
}

void PhotometricOptimizer::GetExtri(Eigen::Isometry3f& extri) const {
	extri = Eigen::Isometry3f(extri_rot_.cast<float>());
	extri.pretranslate(extri_trans_.cast<float>());
}

void PhotometricOptimizer::SetCameraIntri(const CamIntri& intri) {
	PhotometricError::SetSharedCamIntri(intri.fx, intri.fy, intri.cx, intri.cy);
}

void PhotometricOptimizer::AddImgWithPose(const cv::Mat& img, const Eigen::Isometry3f& T_lw) {
	PhotometricError::AddImgWithPose(img, T_lw);
}

void PhotometricOptimizer::SetInitExtri(const Eigen::Isometry3f& extri) {
	Eigen::Isometry3d extri_d = extri.cast<double>();

	extri_rot_ = Eigen::Quaterniond(extri_d.rotation());
	extri_rot_.normalize();  // Get unit quat

	extri_trans_ = extri_d.translation();
}

void PhotometricOptimizer::AddResidual(const Eigen::Vector3f& pt_world, const std::vector<size_t>& visible_idx) {
	ceres::CostFunction* cost_func = PhotometricError::Create(pt_world, visible_idx);
	ceres::ResidualBlockId r_id = opt_prob_.AddResidualBlock(cost_func, nullptr, extri_rot_.coeffs().data(), extri_trans_.data());
	res_ids_.insert(r_id);
}

void PhotometricOptimizer::EvalAllResidualBlk(std::unordered_map<ceres::ResidualBlockId, double>& result) {
	result.clear();
	for (const ceres::ResidualBlockId &id : res_ids_) {
		double residual;
		opt_prob_.EvaluateResidualBlock(id, false, nullptr, &residual, nullptr);
		result[id] = residual;
	}
}

void PhotometricOptimizer::EvalAllResidualBlk(std::vector<double>& result) {
	result.clear();

	std::unordered_map<ceres::ResidualBlockId, double> eval;
	EvalAllResidualBlk(eval);
	result.reserve(eval.size());

	for (const auto &kv : eval) {
		result.push_back(kv.second);
	}
}

void PhotometricOptimizer::EvalAllResidualBlk(std::vector<std::pair<ceres::ResidualBlockId, double>>& result) {
	result.clear();

	std::unordered_map<ceres::ResidualBlockId, double> eval;
	EvalAllResidualBlk(eval);
	result.reserve(eval.size());

	for (const auto& kv : eval) {
		result.push_back(kv);
	}
}

void PhotometricOptimizer::RemoveResidualBlk(const ceres::ResidualBlockId& blk_id) {
	opt_prob_.RemoveResidualBlock(blk_id);
	res_ids_.erase(blk_id);
}

void PhotometricOptimizer::RemoveResidualBlk(const std::vector<ceres::ResidualBlockId>& blk_ids) {
	for (const ceres::ResidualBlockId &blk_id : blk_ids) {
		RemoveResidualBlk(blk_id);
	}
}

void PhotometricOptimizer::Optimize(int max_iter) {
	opt_prob_.AddParameterBlock(extri_rot_.coeffs().data(), 4, new ceres::EigenQuaternionParameterization());
	opt_prob_.AddParameterBlock(extri_trans_.data(), 3);

	ceres::Solver::Options options;
	options.num_threads = 8;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	options.max_num_iterations = max_iter;
	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &opt_prob_, &summary);

	LOG(INFO) << summary.FullReport();
}

PhotometricOptimizerTransRotSepWeightedVar::PhotometricOptimizerTransRotSepWeightedVar() {
	PhotometricErrorRotOnly::ClearBAImgInfo();
	PhotometricErrorTransOnly::ClearBAImgInfo();
}

void PhotometricOptimizerTransRotSepWeightedVar::SetInitExtri(const Eigen::Isometry3f& extri) {
	Eigen::Isometry3d extri_d = extri.cast<double>();

	extri_rot_ = Eigen::Quaterniond(extri_d.rotation());
	extri_rot_.normalize();  // Get unit quat

	extri_trans_ = extri_d.translation();
}

void PhotometricOptimizerTransRotSepWeightedVar::SetCameraIntri(const CamIntri& intri) {
	PhotometricErrorRotOnly::MutableBAImgInfo().SetCameraIntri(intri);
	PhotometricErrorTransOnly::MutableBAImgInfo().SetCameraIntri(intri);
}

void PhotometricOptimizerTransRotSepWeightedVar::AddImgWithPose(const cv::Mat& img, const Eigen::Isometry3f& T_lw) {
	PhotometricErrorRotOnly::MutableBAImgInfo().AddImgWithPose(img, T_lw);
	PhotometricErrorTransOnly::MutableBAImgInfo().AddImgWithPose(img, T_lw);
}

void PhotometricOptimizerTransRotSepWeightedVar::AddResidual(const Eigen::Vector3f& pt_world, const std::vector<size_t>& visible_idx) {
	ceres::CostFunction* cost_func_rot = PhotometricErrorRotOnly::Create(pt_world, visible_idx, extri_trans_.cast<float>());
	ceres::CostFunction* cost_func_trans = PhotometricErrorTransOnly::Create(pt_world, visible_idx, extri_rot_.cast<float>());

	ceres::ResidualBlockId r_id_rot = opt_prob_.AddResidualBlock(cost_func_rot, nullptr, extri_rot_.coeffs().data());
	ceres::ResidualBlockId r_id_trans = opt_prob_.AddResidualBlock(cost_func_trans, nullptr, extri_trans_.data());

	res_ids_rot_.insert(r_id_rot);
	res_ids_trans_.insert(r_id_trans);
}

void PhotometricOptimizerTransRotSepWeightedVar::Optimize(int max_iter){
	opt_prob_.AddParameterBlock(extri_rot_.coeffs().data(), 4, new ceres::EigenQuaternionParameterization());
	opt_prob_.AddParameterBlock(extri_trans_.data(), 3);

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	options.max_num_iterations = max_iter;
	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &opt_prob_, &summary);

	LOG(INFO) << summary.FullReport();
}

void PhotometricOptimizerTransRotSepWeightedVar::GetExtri(Eigen::Isometry3f& extri) const{
	extri = Eigen::Isometry3f(extri_rot_.cast<float>());
	extri.pretranslate(extri_trans_.cast<float>());
}

void PhotometricOptimizerTransRotSepWeightedVar::EvalResidualBlks(std::unordered_map<ceres::ResidualBlockId, double>& result, const std::unordered_set<ceres::ResidualBlockId>& res_ids) {
	result.clear();
	result.reserve(res_ids.size());
	for (const ceres::ResidualBlockId &id : res_ids) {
		double residual;
		opt_prob_.EvaluateResidualBlock(id, false, nullptr, &residual, nullptr);
		result[id] = residual;
	}
}

void PhotometricOptimizerTransRotSepWeightedVar::EvalRotResidualBlks(std::vector<double>& result) {
	std::unordered_map<ceres::ResidualBlockId, double> eval;
	EvalResidualBlks(eval, res_ids_rot_);
	result.reserve(eval.size());

	for (const auto& kv : eval) {
		result.push_back(kv.second);
	}
}

void PhotometricOptimizerTransRotSepWeightedVar::EvalTransResidualBlks(std::vector<double>& result) {
	result.clear();

	std::unordered_map<ceres::ResidualBlockId, double> eval;
	EvalResidualBlks(eval, res_ids_trans_);
	result.reserve(eval.size());

	for (const auto& kv : eval) {
		result.push_back(kv.second);
	}
}

void PhotometricOptimizerTransRotSepWeightedVar::SetWeightThresh(const float& thresh, const float& trans_factor) {
	PhotometricErrorRotOnly::MutableBAImgInfo().SetRotWeightThresh(thresh);
	PhotometricErrorTransOnly::MutableBAImgInfo().SetTransWeightThresh(thresh * trans_factor);

	LOG(INFO) << "RotWeightThresh: " << thresh;
	LOG(INFO) << "TransWeightThresh: " << thresh * trans_factor;
}

void PhotometricOptimizerTransRotSepWeightedVar::EvalRotResidualBlks(std::vector<std::pair<ceres::ResidualBlockId, double>>& result) {
	result.clear();

	std::unordered_map<ceres::ResidualBlockId, double> eval;
	EvalResidualBlks(eval, res_ids_rot_);
	result.reserve(eval.size());

	for (const auto& kv : eval) {
		result.push_back(kv);
	}
}

PhOptTransRotSepRepeated::PhOptTransRotSepRepeated() {
	PhotometricErrorRotOnly::ClearBAImgInfo();
	PhotometricErrorTransOnly::ClearBAImgInfo();
}

void PhOptTransRotSepRepeated::SetCameraIntri(const CamIntri& intri) {
	PhotometricErrorRotOnly::MutableBAImgInfo().SetCameraIntri(intri);
	PhotometricErrorTransOnly::MutableBAImgInfo().SetCameraIntri(intri);
}

void PhOptTransRotSepRepeated::SetInitExtri(const Eigen::Isometry3f& extri) {
	Eigen::Isometry3d extri_d = extri.cast<double>();

	extri_rot_ = Eigen::Quaterniond(extri_d.rotation());
	extri_rot_.normalize();  // Get unit quat

	extri_trans_ = extri_d.translation();
}

void PhOptTransRotSepRepeated::AddImgWithPose(const cv::Mat& img, const Eigen::Isometry3f& T_lw) {
	PhotometricErrorRotOnly::MutableBAImgInfo().AddImgWithPose(img, T_lw);
	PhotometricErrorTransOnly::MutableBAImgInfo().AddImgWithPose(img, T_lw);
}

void PhOptTransRotSepRepeated::SetWeightThresh(const float& thresh, const float& trans_factor) {
	PhotometricErrorRotOnly::MutableBAImgInfo().SetRotWeightThresh(thresh);
	PhotometricErrorTransOnly::MutableBAImgInfo().SetTransWeightThresh(thresh * trans_factor);

	LOG(INFO) << "RotWeightThresh: " << thresh;
	LOG(INFO) << "TransWeightThresh: " << thresh * trans_factor;
}

void PhOptTransRotSepRepeated::OptimizeRotation(int max_iter, const CovisGraph& cg, const PtCloudXYZI_t& ptcloud) {
	ceres::Problem prob;

	prob.AddParameterBlock(extri_rot_.coeffs().data(), 4, new ceres::EigenQuaternionParameterization());

	std::vector<PtId_t> pts;
	cg.GetAllPts(pts);

	for (const auto &pt : pts) {
		std::unordered_set<FrameId_t> frames;
		cg.FindAllVisibleFrames(pt, frames);

		// Set to vector
		std::vector<FrameId_t> frames_vec;
		frames_vec.reserve(frames.size());
		for (const auto &fid : frames) {
			frames_vec.push_back(fid);
		}
		
		const pcl::PointXYZI& pt_xyzi = ptcloud.at(pt);
		ceres::CostFunction* cfunc_rot = PhotometricErrorRotOnly::Create(Eigen::Vector3f(pt_xyzi.x, pt_xyzi.y, pt_xyzi.z), frames_vec, extri_trans_.cast<float>());
		ceres::ResidualBlockId rid_rot = prob.AddResidualBlock(cfunc_rot, new ceres::HuberLoss(20.0), extri_rot_.coeffs().data());

		res_ids_rot_.insert(rid_rot);
	}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	options.max_num_iterations = max_iter;
	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &prob, &summary);

	LOG(INFO) << summary.FullReport();
}

void PhOptTransRotSepRepeated::OptimizeTranslation(int max_iter, const CovisGraph& cg, const PtCloudXYZI_t& ptcloud) {
	ceres::Problem prob;

	prob.AddParameterBlock(extri_trans_.data(), 3);

	std::vector<PtId_t> pts;
	cg.GetAllPts(pts);

	for (const auto& pt : pts) {
		std::unordered_set<FrameId_t> frames;
		cg.FindAllVisibleFrames(pt, frames);

		// Set to vector
		std::vector<FrameId_t> frames_vec;
		frames_vec.reserve(frames.size());
		for (const auto& fid : frames) {
			frames_vec.push_back(fid);
		}

		const pcl::PointXYZI& pt_xyzi = ptcloud.at(pt);
		ceres::CostFunction* cfunc_trans = PhotometricErrorTransOnly::Create(Eigen::Vector3f(pt_xyzi.x, pt_xyzi.y, pt_xyzi.z), frames_vec, extri_rot_.cast<float>());
		ceres::ResidualBlockId rid_trans = prob.AddResidualBlock(cfunc_trans, new ceres::HuberLoss(20.0), extri_trans_.data());

		res_ids_trans_.insert(rid_trans);
	}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	options.max_num_iterations = max_iter;
	options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;

	ceres::Solver::Summary summary;
	ceres::Solve(options, &prob, &summary);

	LOG(INFO) << summary.FullReport();
}

void PhOptTransRotSepRepeated::GetExtri(Eigen::Isometry3f& extri) const{
	extri = Eigen::Isometry3f(extri_rot_.cast<float>());
	extri.pretranslate(extri_trans_.cast<float>());
}

PhotoOptimzerPyr::PhotoOptimzerPyr(const CovisDataPool& dp, const CovisGraph& cg, const CamIntri& intri, 
	const OnlineOptimizerConf& conf, const Eigen::Isometry3f &init_extri, const std::unordered_set<PtId_t>& trans_mask)
	: dp_(dp), cg_(cg), intri_(intri), conf_(conf), init_extri_(init_extri), trans_pt_mask_(trans_mask){

	CHECK(cg.Weighted()) << "[Optimizer] Need a weighted covis graph.";

	// Set init extrinsics (in T_cl)
	SetExtri(init_extri);
	
}

void PhotoOptimzerPyr::SetExtri(const Eigen::Isometry3f& extri) {
	Eigen::Isometry3d extri_d = extri.cast<double>();

	extri_rot_ = Eigen::Quaterniond(extri_d.rotation());
	extri_rot_.normalize();  // Get unit quat

	extri_trans_ = extri_d.translation();
}

void PhotoOptimzerPyr::Optimize(Eigen::Isometry3f& result) {

	for (int lvl = conf_.pyramid_lvl; lvl >= 0; --lvl) {
		LOG(INFO) << "[Optimizer] Optimizing on pyramid level " << lvl;

		// Optimize (record residuals before optimization)
		OptimizeSinglePyrLvl(lvl);
	}

	GetExtri(result);

	// Maybe another optimization run until converge?
}

void PhotoOptimzerPyr::OptimizeSinglePyrLvl(int lvl) {
	// -- Prepare data
	BAImgInfo& rot_ba_info = PhotoErrRotWeighted::MutableBAImgInfo();
	BAImgInfo& trans_ba_info = PhotoErrTransWeighted::MutableBAImgInfo();

	rot_ba_info.Clear();
	trans_ba_info.Clear();

	// Set intrinsics
	Eigen::Matrix3f lvl_intri;
	ImgPyramid::IntriPyrDown(intri_.AsMat(), lvl_intri, lvl);
	rot_ba_info.SetCameraIntri(CamIntri::FromMat(lvl_intri));
	trans_ba_info.SetCameraIntri(CamIntri::FromMat(lvl_intri));

	LOG(INFO) << "[Optimizer] Intrinsics on level: " << lvl_intri;

	// Set images and poses
	const std::vector<ImgWithPose_t> &imgs_poses_all = dp_.AllImgsWithPose();
	for (const ImgWithPose_t& img_pose : imgs_poses_all) {
		const Eigen::Isometry3f& T_cw = img_pose.second;
		const Eigen::Isometry3f& T_lw = init_extri_.inverse() * T_cw;

		cv::Mat lvl_img;
		ImgPyramid::ImgPyrDown(img_pose.first, lvl_img, lvl);

		rot_ba_info.AddImgWithPose(lvl_img, T_lw);
		trans_ba_info.AddImgWithPose(lvl_img, T_lw);
	}

	LOG(INFO) << "[Optimizer] Total image number: " << imgs_poses_all.size();

	// -- Optimize rotation
	ceres::Problem rot_prob;
	ResBlkId_PtId_Map_t rot_id_map;
	ConstructProblem(rot_prob, true, rot_id_map);

	// ResBlkResidualBatch_t rot_eval_before;
	// EvalAllResdualBlks(rot_prob, rot_eval_before);
	// EraseHighResidualBlks(rot_prob, rot_eval_before, conf_.residual_reserve_percent);

	SolveAndReport(rot_prob, true);

	// Outlier removal (points that results in a higher residual after optimization)
	// ResBlkResidualBatch_t rot_eval_after;
	// EvalAllResdualBlks(rot_prob, rot_eval_after);
	// JudgeOutliers(rot_eval_before, rot_eval_after, rot_id_map);

	// -- Optimize translation
	// ceres::Problem trans_prob;
	// ResBlkId_PtId_Map_t trans_id_map;
	// ConstructProblem(trans_prob, false, trans_id_map);

	// ResBlkResidualBatch_t trans_eval_before;
	// EvalAllResdualBlks(trans_prob, trans_eval_before);
	// EraseHighResidualBlks(trans_prob, trans_eval_before, conf_.residual_reserve_percent);

	// SolveAndReport(trans_prob, false);

	// Outlier removal (points that results in a higher residual after optimization)
	// ResBlkResidualBatch_t trans_eval_after;
	// EvalAllResdualBlks(trans_prob, trans_eval_after);
	// JudgeOutliers(trans_eval_before, trans_eval_after, trans_id_map);

	// LOG(INFO) << "[Optimizer] Total outliers: " << outlier_pts_.size();

	// Clear after optimization done
	rot_ba_info.Clear();
	trans_ba_info.Clear();

}

void PhotoOptimzerPyr::SolveAndReport(ceres::Problem& prob, bool is_rotation_solver) {
	ceres::Solver::Options opt;
	opt.linear_solver_type = ceres::DENSE_SCHUR;
	opt.minimizer_progress_to_stdout = false;
	opt.max_num_iterations = conf_.max_iter;
	opt.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
	opt.num_threads = std::thread::hardware_concurrency();

	ceres::Solver::Summary summary;
	ceres::Solve(opt, &prob, &summary);

	// LOG(INFO) << summary.FullReport();
}

void PhotoOptimzerPyr::ConstructProblem(ceres::Problem& problem, bool is_rotation_problem, ResBlkId_PtId_Map_t &id_map) {
	std::vector<PtId_t> pt_ids;
	cg_.GetAllPts(pt_ids);

	if (is_rotation_problem) { // Rotation problem
		problem.AddParameterBlock(extri_rot_.coeffs().data(), 4, new ceres::EigenQuaternionParameterization());

		for (const PtId_t& pt_id : pt_ids) {
			// Continue if marked as outlier
			if (outlier_pts_.find(pt_id) != outlier_pts_.end()) {
				continue;
			}

			std::vector<FrameId_t> vis_frames;
			std::vector<float> weights;
			cg_.QueryWeightRotPoint(pt_id, vis_frames, weights);

			const PtXYZI_t& pt = dp_.GetPt(pt_id); // In world coordinate
			ceres::CostFunction* cf_rot =
				PhotoErrRotWeighted::Create(Eigen::Vector3f(pt.x, pt.y, pt.z), vis_frames, weights, extri_trans_.cast<float>());
			ceres::ResidualBlockId rid_rot = problem.AddResidualBlock(cf_rot, nullptr, extri_rot_.coeffs().data());

			// Record block -> pt id
			id_map.insert(std::make_pair(rid_rot, pt_id));
		}
	}
	else { // Translation problem
		problem.AddParameterBlock(extri_trans_.data(), 3);
		for (const PtId_t& pt_id : pt_ids) {
			// Continue if the point is in mask
			if (trans_pt_mask_.find(pt_id) != trans_pt_mask_.end()) {
				continue;
			}
			// Continue if marked as outlier
			if (outlier_pts_.find(pt_id) != outlier_pts_.end()) {
				continue;
			}
			std::vector<FrameId_t> vis_frames;
			std::vector<float> weights;
			cg_.QueryWeightTransPoint(pt_id, vis_frames, weights);

			const PtXYZI_t& pt = dp_.GetPt(pt_id); // In world coordinate
			ceres::CostFunction* cf_trans =
				PhotoErrTransWeighted::Create(Eigen::Vector3f(pt.x, pt.y, pt.z), vis_frames, weights, extri_rot_.cast<float>());
			ceres::ResidualBlockId rid_trans = problem.AddResidualBlock(cf_trans, nullptr, extri_trans_.data());

			// Record block -> pt id
			id_map.insert(std::make_pair(rid_trans, pt_id));
		}
	}
}

void PhotoOptimzerPyr::EvalAllResdualBlks(const ceres::Problem& problem, ResBlkResidualBatch_t& result) {
	result.clear();
	result.reserve(problem.NumResidualBlocks());

	std::vector<ceres::ResidualBlockId> blk_ids;
	problem.GetResidualBlocks(&blk_ids);

	for (const auto& id : blk_ids) {
		double residual;
		problem.EvaluateResidualBlock(id, false, nullptr, &residual, nullptr);

		result.push_back(std::make_pair(id, residual));
	}

	LOG(INFO) << "[Optimizer] Evaluate blocks: " << result.size();

}

void PhotoOptimzerPyr::EraseHighResidualBlks(ceres::Problem& problem, ResBlkResidualBatch_t& eval_result, float percent_reserve) {
	using pair_type = std::vector<std::pair<ceres::ResidualBlockId, double>>::value_type;

	CHECK(percent_reserve > 0.0f && percent_reserve < 100.0f);
	CHECK(!eval_result.empty());

	// Copy
	ResBlkResidualBatch_t blk_residuals = eval_result;

	// Sort in non-decreasing order
	std::sort(blk_residuals.begin(), blk_residuals.end(), 
		[](const pair_type& p1, const pair_type& p2) {
			return p1.second < p2.second;
		});

	// Remove and memorize outlier point
	int thresh_idx = static_cast<int>(blk_residuals.size() * percent_reserve);
	std::unordered_set<ceres::ResidualBlockId> blks_removed;
	blks_removed.reserve(blk_residuals.size() - thresh_idx);
	for (int i = thresh_idx; i < blk_residuals.size(); ++i) {
		// outlier_pts_.insert(blk_residuals[i].second);
		problem.RemoveResidualBlock(blk_residuals[i].first);
		blks_removed.insert(blk_residuals[i].first);
	}

	// Remove data in eval_result
	eval_result.erase(std::remove_if(eval_result.begin(), eval_result.end(),
		[&blks_removed](const pair_type& p) {
			if (blks_removed.find(p.first) != blks_removed.end()) {
				return true;
			}
			return false;
		}), eval_result.end());

	LOG(INFO) << "[Optimizer] Total residual blocks: " << blk_residuals.size()
		<< ", remove number: " << blks_removed.size()
		<< ", remove threshold: " << blk_residuals[thresh_idx].second;
}

void PhotoOptimzerPyr::JudgeOutliers(const ResBlkResidualBatch_t& before, const ResBlkResidualBatch_t& after, const ResBlkId_PtId_Map_t& id_map) {
	CHECK(!id_map.empty());
	
	ResBlkId_Res_Map_t before_map, after_map;
	BatchToMap(before, before_map);
	BatchToMap(after, after_map);

	int cnt = 0;
	for (const auto &id_res : after_map) {
		const auto& res = before_map.find(id_res.first);
		if (res != before_map.end()) {
			const double &before_val = res->second;
			if (before_val < id_res.second) {
				outlier_pts_.insert(id_map.at(id_res.first));
				cnt += 1;
			}
		}
	}

	LOG(INFO) << "[Optimizer] Outliers after optimization: " << cnt;
}

void PhotoOptimzerPyr::BatchToMap(const ResBlkResidualBatch_t& batch, ResBlkId_Res_Map_t& map) {
	map.clear();
	map.reserve(batch.size());
	for (const auto &id_res : batch) {
		map.insert(id_res);
	}
}

void PhotoOptimzerPyr::GetExtri(Eigen::Isometry3f& extri) {
	extri = Eigen::Isometry3f(extri_rot_.cast<float>());
	extri.pretranslate(extri_trans_.cast<float>());
}