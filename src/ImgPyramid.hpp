#pragma once

#include <mutex>
#include <memory>

#include <boost/functional/hash.hpp>

#include <opencv2/core/mat.hpp>

#include <Eigen/Core>

namespace phc {
	class ImgPyramid {
	public:
		ImgPyramid(const cv::Mat& img, const Eigen::Matrix3f& intri)
			: img_(img), intri_mat_(intri) {}

		void GetLevel(cv::Mat &img, Eigen::Matrix3f &intri, unsigned int level) const;

		void GetLevelImg(cv::Mat &img, unsigned int level) const;
		void GetLevelIntri(Eigen::Matrix3f &intri, unsigned int level) const;

		static void ImgPyrDown(const cv::Mat &origin_img, cv::Mat &level_img, unsigned int level);
		static void IntriPyrDown(const Eigen::Matrix3f& origin_intri, Eigen::Matrix3f& level_intri, unsigned int level);
	private:
		cv::Mat img_;
		Eigen::Matrix3f intri_mat_;
	};

	class PyramidCache {
	public:
		PyramidCache(PyramidCache &other) = delete;
		void operator=(const PyramidCache&) = delete;

		static PyramidCache& Instance();

		const cv::Mat& Find(const uchar* addr, unsigned int level) const;
		void Add(const uchar* addr, unsigned int level, const cv::Mat &img);
	private:
		PyramidCache() = default;

		std::unordered_map<std::pair<const uchar*, unsigned int>, cv::Mat, boost::hash<std::pair<const uchar*, unsigned int>>> cache_;

		static std::unique_ptr<PyramidCache> instance_;
		static std::mutex instance_mutex_;
	};
}