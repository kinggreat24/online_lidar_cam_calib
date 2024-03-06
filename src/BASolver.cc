#include "BASolver.hpp"

using phc::PhotometricError;
using phc::PhotometricErrorRotOnly;
using phc::PhotometricErrorTransOnly;
using phc::PhotoErrRotWeighted;
using phc::PhotoErrTransWeighted;
using phc::BAImgInfo;

float PhotometricError::fx_ = 0.0f;
float PhotometricError::fy_ = 0.0f;
float PhotometricError::cx_ = 0.0f;
float PhotometricError::cy_ = 0.0f;

std::vector<cv::Mat> PhotometricError::imgs_;
std::vector<Eigen::Isometry3f> PhotometricError::T_lw_;
std::vector<std::shared_ptr<ceres::Grid2D<double>>> PhotometricError::img_grid_;
std::vector<std::shared_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double>>>> PhotometricError::interpolators_;

BAImgInfo PhotometricErrorRotOnly::img_info_;
BAImgInfo PhotometricErrorTransOnly::img_info_;

BAImgInfo PhotoErrRotWeighted::img_info_;
BAImgInfo PhotoErrTransWeighted::img_info_;

void BAImgInfo::AddImgWithPose(const cv::Mat& img, const Eigen::Isometry3f& T_lw) {
	using ceres::Grid2D;
	using ceres::BiCubicInterpolator;
	using std::shared_ptr;

	// Push pose
	T_lw_.push_back(T_lw);

	// Push image
	cv::Mat img_f = img.clone();
	img_f.convertTo(img_f, CV_64FC1);
	imgs_.push_back(img_f);

	// Prepare interpolator and 2d grid
	double* ptr = imgs_.back().ptr<double>(0);
	shared_ptr<Grid2D<double>> grid_ptr(new Grid2D<double>(ptr, 0, img_f.rows, 0, img_f.cols));
	img_grid_.push_back(grid_ptr);

	shared_ptr<BiCubicInterpolator<Grid2D<double>>> intp_ptr(new BiCubicInterpolator<Grid2D<double>>(*grid_ptr));
	interpolators_.push_back(intp_ptr);
}

const ceres::BiCubicInterpolator<ceres::Grid2D<double>>& BAImgInfo::GetInterpolator(size_t idx) const {
	return *interpolators_[idx];
}

Eigen::Isometry3f BAImgInfo::GetTlw(size_t idx) const {
	return T_lw_[idx];
}

void BAImgInfo::Clear(){
	intri_.Clear();
	T_lw_.clear();
	imgs_.clear();
	img_grid_.clear();
	interpolators_.clear();

	rot_weight_thresh_ = 0.0f;
	trans_weight_thresh_ = 0.0f;
}