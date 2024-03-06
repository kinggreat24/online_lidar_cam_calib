#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "ImgPyramid.hpp"
#include "Types.hpp"

int main(int argc, char const* argv[]) {
	cv::Mat img = cv::imread("F:\\Datasets\\KITTI\\sequence\\05\\image_0\\000000.png", cv::IMREAD_GRAYSCALE);
	Eigen::Matrix3f dummy_intri;
	dummy_intri << 
		2, 0, 4,
		0, 8, 6,
		0, 0, 1;

	phc::CamIntri intri = phc::CamIntri::FromMat(dummy_intri);

	std::cout << intri.AsMat() << std::endl;

	phc::ImgPyramid pyramid(img, dummy_intri);

	for (int level = 0; level < 5; ++level) {
		cv::Mat level_img;
		Eigen::Matrix3f level_intri;

		pyramid.GetLevel(level_img, level_intri, level);

		std::cout << level_intri << std::endl;

		cv::imshow("level", level_img);
		cv::waitKey(0);
	}

	return EXIT_SUCCESS;

}
