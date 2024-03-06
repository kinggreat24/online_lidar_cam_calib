#pragma once

#include <ceres/ceres.h>

#include "CovisGraph.hpp"
#include "CovisDataPool.hpp"
#include "Types.hpp"

namespace phc {
	class PhotometricOptimizer {
	public:
		PhotometricOptimizer();

		void GetExtri(Eigen::Isometry3f &extri) const;

		void SetCameraIntri(const CamIntri& intri);
		void SetInitExtri(const Eigen::Isometry3f& extri);
		void AddImgWithPose(const cv::Mat &img, const Eigen::Isometry3f &T_lw);

		// visible_idx follows the sequence of AddImgWithPose
		void AddResidual(const Eigen::Vector3f &pt_world, const std::vector<size_t> &visible_idx);

		void EvalAllResidualBlk(std::unordered_map<ceres::ResidualBlockId, double> &result);
		void EvalAllResidualBlk(std::vector<double> &result);
		void EvalAllResidualBlk(std::vector<std::pair<ceres::ResidualBlockId, double>> &result);
		void RemoveResidualBlk(const ceres::ResidualBlockId &blk_id);
		void RemoveResidualBlk(const std::vector<ceres::ResidualBlockId> &blk_ids);

		void Optimize(int max_iter);

	private:
		Eigen::Quaterniond extri_rot_;
		Eigen::Vector3d extri_trans_;

		ceres::Problem opt_prob_;

		std::unordered_set<ceres::ResidualBlockId> res_ids_;
	};

	// Optimize translation and rotation seperately
	class PhotometricOptimizerTransRotSepWeightedVar {
	public:
		PhotometricOptimizerTransRotSepWeightedVar();

		void GetExtri(Eigen::Isometry3f& extri) const;

		void SetInitExtri(const Eigen::Isometry3f& extri);
		void SetCameraIntri(const CamIntri& intri);
		void SetWeightThresh(const float &thresh, const float &trans_factor);
		void AddImgWithPose(const cv::Mat& img, const Eigen::Isometry3f& T_lw);

		// visible_idx follows the sequence of AddImgWithPose
		void AddResidual(const Eigen::Vector3f& pt_world, const std::vector<size_t>& visible_idx);

		void EvalResidualBlks(std::unordered_map<ceres::ResidualBlockId, double>& result, const std::unordered_set<ceres::ResidualBlockId> &res_ids);
		void EvalRotResidualBlks(std::vector<double>& result);
		void EvalTransResidualBlks(std::vector<double>& result);
		void EvalRotResidualBlks(std::vector<std::pair<ceres::ResidualBlockId, double>>& result);

		void Optimize(int max_iter);
	private:
		Eigen::Quaterniond extri_rot_;
		Eigen::Vector3d extri_trans_;

		ceres::Problem opt_prob_;

		std::unordered_set<ceres::ResidualBlockId> res_ids_rot_;
		std::unordered_set<ceres::ResidualBlockId> res_ids_trans_;
	};

	// Optimize translation and rotation seperately, repeated step till converge
	class PhOptTransRotSepRepeated {
	public:
		PhOptTransRotSepRepeated();

		void GetExtri(Eigen::Isometry3f& extri) const;

		void SetCameraIntri(const CamIntri& intri);
		void SetInitExtri(const Eigen::Isometry3f& extri);
		void AddImgWithPose(const cv::Mat& img, const Eigen::Isometry3f& T_lw);
		void SetWeightThresh(const float& thresh, const float& trans_factor);

		void OptimizeRotation(int max_iter, const CovisGraph& cg, const PtCloudXYZI_t &ptcloud);
		void OptimizeTranslation(int max_iter, const CovisGraph& cg, const PtCloudXYZI_t& ptcloud);

	private:
		Eigen::Quaterniond extri_rot_;
		Eigen::Vector3d extri_trans_;

		std::unordered_set<ceres::ResidualBlockId> res_ids_rot_;
		std::unordered_set<ceres::ResidualBlockId> res_ids_trans_;
	};

	class PhotoOptimzerPyr {
	public:
		// The constructor only takes reference to CovisDataPool & CovisGraph
		// Make sure CovisDataPool & CovisGraph are alive when using PhotoOptimzerPyr
		// init_extri in T_cl
		PhotoOptimzerPyr(const CovisDataPool& dp, const CovisGraph& cg, const CamIntri& intri, 
			const OnlineOptimizerConf& conf, const Eigen::Isometry3f& init_extri, const std::unordered_set<PtId_t> &trans_mask);

		void Optimize(Eigen::Isometry3f &result);
	private:
		using ResBlkResidual_t = std::pair<ceres::ResidualBlockId, double>;
		using ResBlkResidualBatch_t = std::vector<ResBlkResidual_t>;
		using ResBlkId_PtId_Map_t = std::unordered_map<ceres::ResidualBlockId, PtId_t>;
		using ResBlkId_Res_Map_t = std::unordered_map<ceres::ResidualBlockId, double>;

		void SetExtri(const Eigen::Isometry3f& extri);
		void GetExtri(Eigen::Isometry3f& extri);

		void OptimizeSinglePyrLvl(int lvl);

		void SolveAndReport(ceres::Problem &prob, bool is_rotation_solver);

		void ConstructProblem(ceres::Problem &problem, bool is_rotation_problem, ResBlkId_PtId_Map_t &id_map);

		void EvalAllResdualBlks(const ceres::Problem& problem, ResBlkResidualBatch_t &result);

		// Erase residual blocks with high residuals
		// percent_reserve = 0.95 will erase the top 5% highest residual block
		void EraseHighResidualBlks(ceres::Problem &problem, ResBlkResidualBatch_t &eval_result, float percent_reserve);

		void JudgeOutliers(const ResBlkResidualBatch_t &before, const ResBlkResidualBatch_t &after, const ResBlkId_PtId_Map_t &id_map);

		void BatchToMap(const ResBlkResidualBatch_t &batch, ResBlkId_Res_Map_t &map);

		OnlineOptimizerConf conf_;

		CamIntri intri_;

		const CovisDataPool& dp_;
		const CovisGraph& cg_;

		std::unordered_set<PtId_t> trans_pt_mask_;
		std::unordered_set<PtId_t> outlier_pts_;

		// float, in T_cl
		Eigen::Isometry3f init_extri_;

		// Ceres use double
		Eigen::Quaterniond extri_rot_;
		Eigen::Vector3d extri_trans_;
	};
}