#pragma once

#include <memory>

#include "CovisGraph.hpp"

namespace phc {
	class CovisDataPool {
	public:
		CovisDataPool();
		~CovisDataPool();

		bool Empty() const;

		void Add(const CovisGraph &cg, const PtCloudXYZI_t &cloud_local, const std::vector<DataFrame> &data_local, const Eigen::Isometry3f& T_cl, CovisGraph& cg_out);

		const PtXYZI_t& GetPt(const PtId_t &pid) const;
		const std::vector<ImgWithPose_t>& AllImgsWithPose() const;

		// Pose in T_cw
		const std::pair<cv::Mat, Eigen::Isometry3f>& GetImgWithPose(const FrameId_t &f_id) const;
		void GetImgsWithPose(const std::vector<FrameId_t>& fids, std::vector<std::pair<cv::Mat, Eigen::Isometry3f>> &result) const;

		const cv::Mat& GetImg(const FrameId_t& f_id) const;
		void GetImgs(const std::unordered_set<FrameId_t> &fids, std::vector<cv::Mat> &result) const;

		void Clear();

		PtCloudXYZI_t::Ptr GetCloud() const;

		const Eigen::Isometry3f& GetExtri() const;

		size_t CloudSize() const;
		size_t ImgNum() const;
	private:
		class Impl;
		std::unique_ptr<Impl> impl_;

	};
}