#include <pcl/common/transforms.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <matplot/matplot.h>

#include <glog/logging.h>

#include "Utils.hpp"
#include "Evaluator.hpp"

using phc::utils::Evaluator;

void Evaluator::ShowProjectionOnImg(const cv::Mat& img, const pcl::PointCloud<pcl::PointXYZI>& ptcloud) const {
	cv::Mat r_map_origin = PtcloudToRangeMap(ptcloud, origin_extri_, img.cols, img.rows);
	cv::Mat r_map_init = PtcloudToRangeMap(ptcloud, init_extri_, img.cols, img.rows);
	cv::Mat r_map_result = PtcloudToRangeMap(ptcloud, result_extri_, img.cols, img.rows);

	cv::Mat color_map_origin = RangeMapToBGR(r_map_origin);
	cv::Mat color_map_init = RangeMapToBGR(r_map_init);
	cv::Mat color_map_result = RangeMapToBGR(r_map_result);

	cv::Mat img_bgr;
	cv::cvtColor(img, img_bgr, cv::COLOR_GRAY2BGR);

	cv::Mat vis_img_origin, vis_img_init, vis_img_result;
	cv::addWeighted(img_bgr, 0.3, color_map_origin, 0.7, 0.0, vis_img_origin);
	cv::addWeighted(img_bgr, 0.3, color_map_init, 0.7, 0.0, vis_img_init);
	cv::addWeighted(img_bgr, 0.3, color_map_result, 0.7, 0.0, vis_img_result);

	cv::imshow("vis_img_origin", vis_img_origin);
	cv::imshow("vis_img_init", vis_img_init);
	cv::imshow("vis_img_result", vis_img_result);

	cv::waitKey(0);
}

void Evaluator::PlotHist(const std::vector<double>& residuals) {
	CHECK(!residuals.empty()) << "[Evaluator::PlotHist] Empty input value";

	std::vector<double> residual_sort = residuals;
	std::sort(residual_sort.begin(), residual_sort.end());

	double median = residual_sort.at(residual_sort.size() / 2);
	double res_min = residual_sort.front();
	double res_max = residual_sort.back();

	double sum = std::accumulate(residuals.begin(), residuals.end(), 0.0);
	double mean = sum / residuals.size();

	int num_mean_less = std::count_if(residual_sort.begin(), residual_sort.end(), 
		[mean](double res) {
			return res < mean;
		});

	LOG(INFO) << "[Residual Hist] mean: " << mean;
	LOG(INFO) << "[Residual Hist] median: " << median;
	LOG(INFO) << "[Residual Hist] % less than mean: " << static_cast<double>(num_mean_less) / static_cast<double>(residual_sort.size()) * 100.0;
	LOG(INFO) << "[Residual Hist] min: " << res_min;
	LOG(INFO) << "[Residual Hist] max: " << res_max;

	auto hist = matplot::hist(residuals);
	matplot::show();
}

void Evaluator::PlotHist(const std::vector<float>& residuals) {
	std::vector<double> d_vec(residuals.begin(), residuals.end());
	PlotHist(d_vec);
}

cv::Mat Evaluator::PtcloudToRangeMap(const pcl::PointCloud<pcl::PointXYZI>& ptcloud, const Eigen::Isometry3f& T_cl, int width, int height) const {
	using pcl::PointCloud;
	using pcl::PointXYZI;

	PointCloud<PointXYZI> cloud_cam; // cloud in camera frame
	pcl::transformPointCloud(ptcloud, cloud_cam, T_cl.matrix());

	std::vector<cv::Point3f> pts_cam;
	for (const PointXYZI& p : cloud_cam.points) {
		if (p.z < 0.0f) { continue; }
		pts_cam.emplace_back(p.x, p.y, p.z);
	}

	cv::Mat intri_mat_cv;
	cv::eigen2cv(intri_mat_, intri_mat_cv);

	std::vector<cv::Point2f> pixels;
	cv::projectPoints(pts_cam, cv::Vec3f::zeros(), cv::Vec3f::zeros(),
		intri_mat_cv, cv::Vec4f(), pixels);

	// Pixels with value < 0 is considered as invalid
	cv::Mat range_map = cv::Mat(height, width, CV_32F, cv::Scalar(-1.0));
	for (size_t i = 0; i < pts_cam.size(); ++i) {
		int x = pixels[i].x;  // col
		int y = pixels[i].y;  // row
		float range = sqrtf(pts_cam[i].x * pts_cam[i].x + pts_cam[i].y * pts_cam[i].y + pts_cam[i].z * pts_cam[i].z);

		float& val = range_map.at<float>(y, x);

		if (x < 0 || x >= width || y < 0 || y >= height) { continue; }

		// When a pxiel has multiple range, we choose the minimal
		if (val > 0.0f && val < range) { continue; }

		val = range;
	}

	return range_map;
}

cv::Mat Evaluator::RangeMapToBGR(const cv::Mat& map) const {
	// Should be BGR
	cv::Mat color_map = cv::Mat(map.rows, map.cols, CV_8UC3, cv::Scalar(0, 0, 0));

	float min, max;
	utils::FindCVMat32FMinMax(map, min, max);

	for (int u = 0; u < color_map.cols; ++u) {
		for (int v = 0; v < color_map.rows; ++v) {
			const float& val = map.at<float>(v, u);
			if (val < 0.0f) { continue; }

			cv::Vec3b& pixel = color_map.at<cv::Vec3b>(v, u);
			pixel = utils::MapValToBGR(min, max, val);
		}
	}

	return color_map;
}

void Evaluator::CompareNorm() const {

	Eigen::Vector3f origin_trans = origin_extri_.translation();
	Eigen::Vector3f init_trans = init_extri_.translation();
	Eigen::Vector3f result_trans = result_extri_.translation();

	Eigen::Quaternionf origin_rot(origin_extri_.rotation());
	Eigen::Quaternionf init_rot(init_extri_.rotation());
	Eigen::Quaternionf result_rot(result_extri_.rotation());

	float trans_norm_origin_init = (origin_trans - init_trans).norm();
	float trans_norm_origin_result = (origin_trans - result_trans).norm();
	float trans_norm_result_init = (result_trans - init_trans).norm();

	LOG(INFO) << "Distance between original and init extrinsics (translation): " << trans_norm_origin_init;
	LOG(INFO) << "Distance between original and optimized extrinsics (translation): " << trans_norm_origin_result;
	LOG(INFO) << "Distance between init and optimized extrinsics (translation): " << trans_norm_result_init;

	if (trans_norm_origin_init > trans_norm_origin_result) {
		LOG(INFO) << "May have a better optimized result (translation)";
	}
	else {
		LOG(INFO) << "May have a worse optimized result (translation)";
	}

	float rot_diff_origin_init = origin_rot.angularDistance(init_rot) / M_PI * 180.0f;
	float rot_diff_origin_result = origin_rot.angularDistance(result_rot) / M_PI * 180.0f;
	float rot_diff_result_init = result_rot.angularDistance(init_rot) / M_PI * 180.0f;

	LOG(INFO) << "Distance between original and init extrinsics (rotation in degree): " << rot_diff_origin_init;
	LOG(INFO) << "Distance between original and optimized extrinsics (rotation in degree): " << rot_diff_origin_result;
	LOG(INFO) << "Distance between init and optimized extrinsics (rotation in degree): " << rot_diff_result_init;

	if (rot_diff_origin_init > rot_diff_origin_result) {
		LOG(INFO) << "May have a better optimized result (rotation)";
	}
	else {
		LOG(INFO) << "May have a worse optimized result (rotation)";
	}

	using Vector6f = Eigen::Matrix<float, 6, 1>;
	// row: around x-axis
	// pitch: around y-axis
	// yaw: around z-axis
	Vector6f origin_xyzrpy = utils::IsometryToXYZRPY(origin_extri_);
	Vector6f result_xyzrpy = utils::IsometryToXYZRPY(result_extri_);
	Vector6f error = result_xyzrpy - origin_xyzrpy;

	LOG(INFO) << "Extrinsics error (x_t y_t z_t x_r y_r z_r): " 
		<< error(0) << " " << error(1) << " " << error(2) << " "
		<< error(3) << " " << error(4) << " " << error(5) << " ";

}

void Evaluator::CostCompare(const std::vector<float>& err_costs) {
	using matplot::bar;
	using matplot::plot;
	using matplot::show;

	std::vector<float> origin_cost(err_costs.size());
	std::fill(origin_cost.begin(), origin_cost.end(), err_costs.back());

	bar(err_costs);
	matplot::hold(matplot::on);
	plot(origin_cost);
	show();
}

void Evaluator::GetOriginResultErr(Eigen::Matrix<float, 6, 1>& err) {
	using Vector6f = Eigen::Matrix<float, 6, 1>;
	// row: around x-axis
	// pitch: around y-axis
	// yaw: around z-axis
	Vector6f origin_xyzrpy = utils::IsometryToXYZRPY(origin_extri_);
	Vector6f result_xyzrpy = utils::IsometryToXYZRPY(result_extri_);
	err = result_xyzrpy - origin_xyzrpy;
}

void Evaluator::DistRotTrans(const Eigen::Isometry3f& extri_1, const Eigen::Isometry3f& extri_2, float& dist_trans, float& dist_rot) {
	using Eigen::Vector3f;
	using Eigen::Quaternionf;

	Vector3f trans_1 = extri_1.translation();
	Vector3f trans_2 = extri_2.translation();

	Quaternionf rot_1(extri_1.rotation());
	Quaternionf rot_2(extri_2.rotation());

	dist_trans = (trans_1 - trans_2).norm();
	dist_rot = rot_1.angularDistance(rot_2) / M_PI * 180.0f;  // To degree
}