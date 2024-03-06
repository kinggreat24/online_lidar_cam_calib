#pragma once

#include <Eigen/Core>

#include <pcl/types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/core/mat.hpp>

#include <ceres/ceres.h>

namespace phc {
	using FrameId_t = size_t;
	using PtId_t = pcl::index_t;

	using PtCloudXYZI_t = pcl::PointCloud<pcl::PointXYZI>;
	using PtCloudXYZRGB_t = pcl::PointCloud<pcl::PointXYZRGB>;

	using PtXYZI_t = pcl::PointXYZI;

	using ImgWithPose_t = std::pair<cv::Mat, Eigen::Isometry3f>;

	using Vector6f = Eigen::Matrix<float, 6, 1>;

	struct DataFrame {
		PtCloudXYZI_t::Ptr ptcloud;
		cv::Mat img;
		Eigen::Isometry3f T_wl;
	};

	struct CamIntri {
		float fx = 0.0f;
		float fy = 0.0f;
		float cx = 0.0f;
		float cy = 0.0f;

		Eigen::Matrix3f AsMat() const;

		void Clear();

		static CamIntri FromMat(const Eigen::Matrix3f& mat);
	};

	struct ResidualEvalInfo {
		std::unordered_map<ceres::ResidualBlockId, double> blkid_cost_map;
		std::unordered_map<ceres::ResidualBlockId, PtId_t> blkid_ptid_map;
	};

	struct CovisCheckConf {
		float max_view_range = 50.0f; // In meter
		float score_thresh = 0.98f;
		int edge_discard = 5; // In pixel
	};

	struct OnlineOptimizerConf {
		int start_frame_num = 100;
		int pyramid_lvl = 4;
		int max_iter = 50;
		float residual_reserve_percent = 0.95;
	};

	struct ErrDetectorConf {
		float ptcloud_clip_min = 5.0f;
		float ptcloud_clip_max = 80.0f;

		float sample_ratio = 0.3f;

		float err_tolr_x = 0.05; // Meter
		float err_tolr_y = 0.05; // Meter
		float trans_thresh_ratio = 0.5; // Ratio for translation threshold

		int obs_thresh = 2;

		int window_size = 10;

		int pyramid_lvl = 0;

		int extri_sample_num = 100;

		float pixel_val_lower_lim = 0.0f;
		float pixel_val_upper_lim = 255.0f;

		CovisCheckConf covis_conf;
	};

	enum class OnlineCaliMode {
		kManualCost = 0,
		kOptimize = 1
	};

	struct OnlineCaliConf {
		float ptcloud_clip_min = 5.0f;
		float ptcloud_clip_max = 80.0f;

		float sample_ratio = 0.1f;

		float err_tolr_x = 0.05; // Meter
		float err_tolr_y = 0.05; // Meter
		float trans_thresh_ratio = 0.5; // Ratio for translation threshold

		int obs_thresh = 2;

		int window_size = 10;

		float pixel_val_lower_lim = 0.0f;
		float pixel_val_upper_lim = 255.0f;

		OnlineCaliMode mode = OnlineCaliMode::kOptimize;

		CovisCheckConf covis_conf;
		OnlineOptimizerConf opt_conf;

	};

	enum class OnlineCaliState {
		kInitializing = 0,
		kRunning = 1
	};
}