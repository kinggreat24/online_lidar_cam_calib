#pragma once

#include <vector>

#include <Eigen/Core>

#include <opencv2/core/mat.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <ceres/ceres.h>

#include "CovisGraph.hpp"
#include "DataRecorder.hpp"
#include "Types.hpp"

namespace phc {

	class PhotoCali {
	public:

		PhotoCali();

		inline void SetRecorder(const utils::DataRecorder::Ptr& recorder_ptr) { recorder_ = recorder_ptr; }
		inline void SetResBlkRemovePercent(const double& percent) { resblk_remove_percent_ = percent; }
		inline void SetGlobCloud(PtCloudXYZI_t::ConstPtr cloud) { *global_cloud_ = *cloud; }
		inline void SetErrTolerance(float tx, float ty) { err_tolerance_x_ = tx; err_tolerance_y_ = ty; }
		
		inline Eigen::Matrix3f GetIntriMat() { return intri_.AsMat(); }

		void SetInterestPtindices(const std::vector<pcl::index_t>&pts_indices);

		void SetRandSampleRatio(const float& r);
		void AddDataFrame(const PtCloudXYZI_t::Ptr& ptcloud_ptr, const cv::Mat &img,const Eigen::Isometry3f& T_wl);
		void SetCamIntri(float fx, float fy, float cx, float cy);
		void SetExtri(const Eigen::Isometry3f &T_cl);

		float ComputeCost();
		Eigen::Vector2f ComputeCostWeighted();
		void Optimize(Eigen::Isometry3f &result);
		void OptimizeSingleLvlPyr(Eigen::Isometry3f& result, unsigned int level);
		void OptimizeMultiLvlPyr(Eigen::Isometry3f& result, unsigned int pyr_depth);
		void OptimizeSingleLvlPyrWeightedVar(Eigen::Isometry3f& result, unsigned int level);
		void OptimizeSingleLvlPyrRepeated(Eigen::Isometry3f& result, unsigned int level);

		void GetConcatPtcloud(PtCloudXYZI_t& cloud_out);

		PtCloudXYZI_t::ConstPtr GetGlobCloud() const;

		PtCloudXYZRGB_t::Ptr GetVisibleCloudForFrame(FrameId_t fid) const;
		void GetVisibleClouds(std::vector<PtCloudXYZRGB_t::Ptr> &ptrs) const;

	private:
		void RangeFilterOnAllPtcloud(float min, float max);
		void CloudFiltering(PtCloudXYZI_t::Ptr cloud);
		void DownSamplePts(std::vector<PtId_t> &pt_indices);
		void ComputeCovisInfo(CovisGraph &covis_g);

		void OptimizeSingleLvlPyr(const Eigen::Isometry3f &init, Eigen::Isometry3f& result, unsigned int level);
		void OptimizeSingleLvlPyrWeightedVar(const Eigen::Isometry3f& init, Eigen::Isometry3f& result, unsigned int level);
		void OptimizeSingleLvlPyrRepeated(const Eigen::Isometry3f& init, Eigen::Isometry3f& result, unsigned int level);
		
		float SinglePtCost(const PtId_t &pt_id, const std::unordered_set<FrameId_t> &vis_frames);
		Eigen::Vector2f SinglePtCostWeighted(const PtId_t& pt_id, const std::unordered_set<FrameId_t>& vis_frames, float thresh);

		// Distance threshold compute
		float DistThreshCompute(int img_width, int img_height);

		// Only preserve (perceent * 100)% most smaller residual block
		void GetBlkIdRemoval(std::vector<ceres::ResidualBlockId> &removal_ids, const std::vector<std::pair<ceres::ResidualBlockId, double>> &id_res_pairs, double percent);

		// Error torlerance for weight conputation
		float err_tolerance_x_ = 0.05f;
		float err_tolerance_y_ = 0.05f;

		float rand_sample_ratio_ = -1.0f;
		double resblk_remove_percent_ = 0.5;

		std::vector<DataFrame> frames_;

		Eigen::Isometry3f T_cl_;

		CamIntri intri_;

		PtCloudXYZI_t::Ptr global_cloud_ = nullptr;  // In world frame

		CovisGraph cov_graph_;

		std::unordered_set<PtId_t> interest_pts_;

		utils::DataRecorder::Ptr recorder_ = nullptr;
	};
}