#include <opencv2/imgproc.hpp>

#include "ImgPyramid.hpp"

using phc::ImgPyramid;
using phc::PyramidCache;

void ImgPyramid::GetLevel(cv::Mat& img, Eigen::Matrix3f& intri, unsigned int level) const {

	GetLevelImg(img, level);
	GetLevelIntri(intri, level);
}

void ImgPyramid::GetLevelImg(cv::Mat& img, unsigned int level) const {
	
	ImgPyrDown(img_, img, level);
}

void ImgPyramid::GetLevelIntri(Eigen::Matrix3f& intri, unsigned int level) const {
	
	IntriPyrDown(intri_mat_, intri, level);
}

void ImgPyramid::ImgPyrDown(const cv::Mat& origin_img, cv::Mat& level_img, unsigned int level) {
	if (level == 0) {
		level_img = origin_img;
		return;
	}

	PyramidCache& cache = PyramidCache::Instance();
	cv::Mat cache_img = cache.Find(origin_img.ptr(), level);
	if (cache_img.empty()) { // No cache found
		level_img = origin_img;
		for (unsigned int i = 0; i < level; ++i) {
			cv::pyrDown(level_img, level_img);
		}

		// Add to cache
		cache.Add(origin_img.ptr(), level, level_img);
	}
	else { // Cache found
		level_img = cache_img;
	}
}

void ImgPyramid::IntriPyrDown(const Eigen::Matrix3f& origin_intri, Eigen::Matrix3f& level_intri, unsigned int level) {
	if (level == 0) {
		level_intri = origin_intri;
		return;
	}

	level_intri = origin_intri;
	for (unsigned int i = 0; i < level; ++i) {
		level_intri = level_intri / 2.0f;
	}
	level_intri(2, 2) = 1.0f;
}

std::unique_ptr<PyramidCache> PyramidCache::instance_(nullptr);
std::mutex PyramidCache::instance_mutex_;

PyramidCache& PyramidCache::Instance() {
	std::lock_guard<std::mutex> lock(instance_mutex_);
	if (!instance_) {
		instance_.reset(new PyramidCache);
	}

	return *instance_;
}

const cv::Mat& PyramidCache::Find(const uchar* addr, unsigned int level) const {
	static std::mutex mutex;

	std::lock_guard<std::mutex> lock(mutex);

	std::pair<const uchar*, unsigned int> p(addr, level);
	auto search = cache_.find(p);
	if (search != cache_.end()) {
		return search->second;
	}
	else {
		return cv::Mat(); // Empty mat
	}
	
}

void PyramidCache::Add(const uchar* addr, unsigned int level, const cv::Mat& img) {
	static std::mutex mutex;

	std::lock_guard<std::mutex> lock(mutex);

	std::pair<const uchar*, unsigned int> p(addr, level);
	cache_[p] = img;
}