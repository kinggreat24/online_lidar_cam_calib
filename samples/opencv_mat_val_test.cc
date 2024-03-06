#include <opencv2/opencv.hpp>

int main() {
	cv::Mat img = cv::imread("F:\\Datasets\\KITTI\\sequence\\05\\image_0\\000702.png", cv::IMREAD_GRAYSCALE);

	const uchar& pixel_val = img.at<uchar>(148, 897);
	std::cout << static_cast<int>(pixel_val) << std::endl;
	cv::imshow("img", img);
	cv::waitKey(0);
	return EXIT_SUCCESS;
}