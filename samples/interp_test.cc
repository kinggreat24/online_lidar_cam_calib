#include <iostream>

#include <opencv2/core.hpp>

#include "Utils.hpp"

using phc::utils::GetSubPixelValBilinear;

int main() {
	cv::Mat img = cv::Mat::zeros(2, 2, CV_8UC1);
	img.at<uchar>(0, 0) = 1;
	img.at<uchar>(1, 0) = 2;
	img.at<uchar>(0, 1) = 3;
	img.at<uchar>(1, 1) = 4;

	std::cout << img << std::endl;

	float interp_val = GetSubPixelValBilinear(img, Eigen::Vector2f(0.5, 0.5));

	std::cout << interp_val << std::endl;

	Eigen::VectorXf vx(3);
	vx << 1.0f, 1.0f, 1.0f;
	

	return EXIT_SUCCESS;
}