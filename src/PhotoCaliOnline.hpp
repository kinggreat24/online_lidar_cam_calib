#pragma once

#include <memory>
#include <deque>

#include "Types.hpp"

// To be removed
#include "DataRecorder.hpp"

namespace phc {
	class PhotoCaliOnline {
	public:
		PhotoCaliOnline();
		PhotoCaliOnline(const OnlineCaliConf &conf);
		~PhotoCaliOnline();

		PtCloudXYZI_t::Ptr GetVisPtCloud();

		void SetCamIntrinsics(float fx, float fy, float cx, float cy);
		void SetInitExtri(const Eigen::Isometry3f& T_cl);
		void AddDataFrame(const PtCloudXYZI_t::Ptr& ptcloud_ptr, const cv::Mat& img, const Eigen::Isometry3f& T_wl);
		void Calibrate();

		void GetExtri(Eigen::Isometry3f &T_cl) const;
		void GetCamIntri(Eigen::Matrix3f &intri_mat) const;
		
		// To be removed
		void SetDataRecorder(utils::DataRecorder::Ptr recorder_ptr);

		void GenCovisDataBag(const std::string &name) const;
		void GenPyrGradDataBag() const;
		void GenErrCompDataBag(const std::string& name, const Eigen::Isometry3f& err_T_cl) const;

	private:
		class Impl;
		std::unique_ptr<Impl> impl_;
	};
}