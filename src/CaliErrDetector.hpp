#pragma once

#include <memory>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "Types.hpp"

namespace phc {
	class CaliErrDetector {
	public:
		CaliErrDetector(const ErrDetectorConf &config);
		~CaliErrDetector();

		void SetExtri(const Eigen::Isometry3f &extri);
		void SetIntri(float fx, float fy, float cx, float cy);

		void AddDataFrame(const PtCloudXYZI_t::Ptr& ptcloud_ptr, const cv::Mat& img, const Eigen::Isometry3f& T_wl);

		void Detect(float& center_cost, std::vector<float> &grid_costs);
		void Detect(float& center_cost);

		void ClearDataPool();

		static float WorsePercentage(const float& center_cost, const std::vector<float>& grid_costs);
		
	private:
		class Impl;
		std::unique_ptr<Impl> impl_;
	};
}