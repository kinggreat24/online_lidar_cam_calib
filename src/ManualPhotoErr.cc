#include <glog/logging.h>

#include "progressbar.hpp"

#include "ManualPhotoErr.hpp"
#include "Utils.hpp"
#include "ImgPyramid.hpp"

using phc::ManualPhotoErr;
using phc::ImgPyramid;

ManualPhotoErr::ManualPhotoErr(const CovisDataPool& dp, const CovisGraph& cg, const CamIntri& intri) : dp_(dp), cg_(cg), intri_(intri){

}

void ManualPhotoErr::Compute(std::vector<float>& costs_rot, std::vector<float>& costs_trans) {
	costs_rot.clear();
	costs_trans.clear();

	std::vector<pcl::index_t> pt_ids;
	cg_.GetAllPts(pt_ids);

	costs_rot.reserve(pt_ids.size());
	costs_trans.reserve(pt_ids.size() - trans_mask_.size());

	for (const pcl::index_t& pt_id : pt_ids) {
		
		float cost_rot = SinglePtCost(pt_id, kComputeRotationCost);
		costs_rot.push_back(cost_rot);

		if (!PtInTransMask(pt_id)) {
			float cost_trans = SinglePtCost(pt_id, kComputeTranslationCost);
			costs_trans.push_back(cost_trans);
		}
	}
}

void ManualPhotoErr::Compute(float& cost) {
	std::vector<float> costs_rot;
	std::vector<float> costs_trans;

	Compute(costs_rot, costs_trans);

	Eigen::VectorXf costs_rot_vec = Eigen::Map<Eigen::VectorXf>(costs_rot.data(), costs_rot.size());
	Eigen::VectorXf costs_trans_vec = Eigen::Map<Eigen::VectorXf>(costs_trans.data(), costs_trans.size());

	Eigen::Vector2f cost_vec = Eigen::Vector2f(costs_rot_vec.mean(), costs_trans_vec.mean());

	cost = cost_vec.norm();
}

float ManualPhotoErr::SinglePtCost(const pcl::index_t &pt_id, CostComputeOption opt) {
	// All visible frames
	std::vector<FrameId_t> frame_ids;
	cg_.FindAllVisibleFrames(pt_id, frame_ids);

	// Get actual point
	// Points are in world frame
	const PtXYZI_t& pt = dp_.GetPt(pt_id);
	Eigen::Vector3f pt_world(pt.x, pt.y, pt.z);
	Eigen::Isometry3f dp_extri = dp_.GetExtri();  // T_cl

	// Intri on pyramid
	Eigen::Matrix3f intri_mat;
	ImgPyramid::IntriPyrDown(intri_.AsMat(), intri_mat, pyr_lvl_);

	Eigen::VectorXf ph_vals(frame_ids.size());
	Eigen::VectorXf weights(frame_ids.size());

	int num_cnt = 0;
	for (const auto &fid : frame_ids) {
		// Frame data, pose in T_cw
		// Note that we maynot use the same extri as dp_
		ImgWithPose_t img_pose = dp_.GetImgWithPose(fid);
		const Eigen::Isometry3f T_lw = dp_extri.inverse() * img_pose.second;  // T_lc * T_cw

		// Pyramid
		cv::Mat img_pyr;
		ImgPyramid::ImgPyrDown(img_pose.first, img_pyr, pyr_lvl_);

		// Project to pixel with given extrinsics T_cl_
		const Eigen::Vector3f pt_cam = T_cl_ * T_lw * pt_world;
		Eigen::Vector2f pixel;
		utils::ProjectPoint(pt_cam, pixel, Eigen::Isometry3f::Identity(), intri_mat);

		// Query weight
		if (opt == kComputeRotationCost) {
			weights(num_cnt) = cg_.QueryWeightRot(fid, pt_id);
		}
		else if(opt == kComputeTranslationCost){
			weights(num_cnt) = cg_.QueryWeightTrans(fid, pt_id);
		}
		else {
			LOG(FATAL) << "[ManualPhotoErr] Cost compute option not set.";
		}
		
		// Photometric val
		float ph_val = utils::GetSubPixelValBilinear(img_pyr, pixel);
		// float ph_val = utils::GetSubPixelRelValBilinear(img_pyr, pixel);
		ph_vals(num_cnt) = ph_val;

		num_cnt += 1;
	}

	return utils::VarianceComputeWeighted(ph_vals, weights);
}

bool ManualPhotoErr::PtInTransMask(const PtId_t& ptid) {
	return (trans_mask_.find(ptid) != trans_mask_.end());
}