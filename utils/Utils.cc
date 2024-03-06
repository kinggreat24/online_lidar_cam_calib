#include <fstream>

#include <glog/logging.h>

#include <pcl/common/transforms.h>
#include <pcl/common/io.h>
#include <pcl/filters/extract_indices.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <Eigen/Core>

#include "Utils.hpp"
#include "Rand.hpp"

namespace {
	const float kDeg2Rad = M_PI / 180.0f;
}

void phc::utils::RemoveCloudPtsOutOfRange(pcl::PointCloud<pcl::PointXYZI>& ptcloud, float range_min, float range_max) {
	using pcl::PointXYZI;

	CHECK(!ptcloud.empty());
	CHECK(range_min > 0.0f && range_max > 0.0f);

	auto it = std::remove_if(ptcloud.points.begin(), ptcloud.points.end(),
		[range_min, range_max](PointXYZI p) {
			float square_dis = p.x * p.x + p.y * p.y + p.z * p.z;
			return (square_dis < range_min * range_min || square_dis > range_max * range_max);
		});

	ptcloud.points.erase(it, ptcloud.points.end());
}

void phc::utils::ProjectPoint(const Eigen::Vector3f& pt, Eigen::Vector2f& pixel, const Eigen::Isometry3f& trans_mat, const Eigen::Matrix3f& intri_mat) {
	Eigen::Vector3f homo_pixel = intri_mat * trans_mat * pt;  // Homo x)
	homo_pixel /= homo_pixel.z();

	pixel.x() = homo_pixel.x();
	pixel.y() = homo_pixel.y();
}

// https://stackoverflow.com/questions/13299409/how-to-get-the-image-pixel-at-real-locations-in-opencv
float phc::utils::GetSubPixelValBilinear(const cv::Mat& img, const Eigen::Vector2f& pixel) {
	// CHECK(!img.empty());

	int x = static_cast<int>(pixel.x());
	int y = static_cast<int>(pixel.y());

	int x0 = cv::borderInterpolate(x, img.cols, cv::BORDER_REFLECT_101);
	int x1 = cv::borderInterpolate(x + 1, img.cols, cv::BORDER_REFLECT_101);
	int y0 = cv::borderInterpolate(y, img.rows, cv::BORDER_REFLECT_101);
	int y1 = cv::borderInterpolate(y + 1, img.rows, cv::BORDER_REFLECT_101);

	float a = pixel.x() - static_cast<float>(x);
	float c = pixel.y() - static_cast<float>(y);

	float val_00 = static_cast<float>(img.at<uchar>(y0, x0));
	float val_01 = static_cast<float>(img.at<uchar>(y1, x0));
	float val_10 = static_cast<float>(img.at<uchar>(y0, x1));
	float val_11 = static_cast<float>(img.at<uchar>(y1, x1));

	return (val_00 * (1.f - a) + val_10 * a) * (1.f - c) + (val_01 * (1.f - a) + val_11 * a) * c;
}

float phc::utils::GetSubPixelRelValBilinear(const cv::Mat& img, const Eigen::Vector2f& pixel) {
	// 3x3 window
	float val_00 = GetSubPixelValBilinear(img, Eigen::Vector2f(pixel.x() - 1, pixel.y() - 1));
	float val_01 = GetSubPixelValBilinear(img, Eigen::Vector2f(pixel.x(), pixel.y() - 1));
	float val_02 = GetSubPixelValBilinear(img, Eigen::Vector2f(pixel.x() + 1, pixel.y() - 1));
	float val_10 = GetSubPixelValBilinear(img, Eigen::Vector2f(pixel.x() - 1, pixel.y()));
	float val_11 = GetSubPixelValBilinear(img, Eigen::Vector2f(pixel.x(), pixel.y()));
	float val_12 = GetSubPixelValBilinear(img, Eigen::Vector2f(pixel.x() + 1, pixel.y()));
	float val_20 = GetSubPixelValBilinear(img, Eigen::Vector2f(pixel.x() - 1, pixel.y() + 1));
	float val_21 = GetSubPixelValBilinear(img, Eigen::Vector2f(pixel.x(), pixel.y() + 1));
	float val_22 = GetSubPixelValBilinear(img, Eigen::Vector2f(pixel.x() + 1, pixel.y() + 1));

	float mean = (val_00 + val_01 + val_02 + val_10 + val_11 + val_12 + val_20 + val_21 + val_22) / 9.0f;

	return val_11 - mean;
}

float phc::utils::VarianceCompute(const Eigen::VectorXf& vec) {
	Eigen::VectorXf tmp = vec - Eigen::VectorXf::Ones(vec.size()) * vec.mean();

	return tmp.squaredNorm() / tmp.size();
}

Eigen::Vector2f phc::utils::VarianceComputeWeighted(const Eigen::VectorXf& vec, const Eigen::VectorXf& weight_rot, const Eigen::VectorXf& weight_trans) {
	// No assert on hot path
	// CHECK_EQ(vec.size(), weight_rot.size());
	// CHECK_EQ(vec.size(), weight_trans.size());

	Eigen::VectorXf squared_diff = (vec - Eigen::VectorXf::Ones(vec.size()) * vec.mean()).array().pow(2.0f);
	Eigen::VectorXf weighted_squared_diff_rot = squared_diff.cwiseProduct(weight_rot);
	Eigen::VectorXf weighted_squared_diff_trans = squared_diff.cwiseProduct(weight_trans);

	Eigen::Vector2f cost_vec(weighted_squared_diff_rot.sum(), weighted_squared_diff_trans.sum());
	cost_vec /= vec.size();

	return cost_vec;
}

float phc::utils::VarianceComputeWeighted(const Eigen::VectorXf& vec, const Eigen::VectorXf& weight) {
	Eigen::VectorXf squared_diff = (vec - Eigen::VectorXf::Ones(vec.size()) * vec.mean()).array().pow(2.0f);
	Eigen::VectorXf weighted_squared_diff = squared_diff.cwiseProduct(weight);

	return weighted_squared_diff.sum() / vec.size();
}

void phc::utils::ImgMarkPixel(const cv::Mat& in, cv::Mat& out, const Eigen::Vector2f& subpixel, const std::string& text) {
	static const cv::Point2f kTextOffset(10.0f, 10.0f);

	cv::cvtColor(in, out, cv::COLOR_GRAY2BGR);

	cv::Point2f location(subpixel.x(), subpixel.y());
	cv::circle(out, location, 1, cv::Scalar(0, 255, 0), cv::FILLED);

	if (!text.empty()) {
		cv::putText(out, text, location + kTextOffset, cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255));
	}
}

void phc::utils::ImgMarkPixel(cv::Mat& img, const Eigen::Vector2f& subpixel, const std::string& text) {
	cv::Point2f location(subpixel.x(), subpixel.y());
	cv::circle(img, location, 1, cv::Scalar(0, 0, 255), cv::FILLED);
	if (!text.empty()) {
		cv::putText(img, text, location, cv::FONT_HERSHEY_SIMPLEX, 0.3, cv::Scalar(0, 255, 0));
	}
}

void phc::utils::ImgMarkPixel(cv::Mat& img, const Eigen::Vector2f& subpixel, uint8_t r, uint8_t g, uint8_t b) {
	cv::Point2f location(subpixel.x(), subpixel.y());
	// img.at<cv::Vec3b>(location) = cv::Vec3b(b, g, r);
	cv::circle(img, location, 1, cv::Scalar(b, g, r), cv::FILLED);
}

void phc::utils::PtcloudSetColor(pcl::PointCloud<pcl::PointXYZRGB>& cloud, uint32_t r, uint32_t g, uint32_t b) {
	uint32_t rgb = (static_cast<uint32_t>(r) << 16 |
		static_cast<uint32_t>(g) << 8 |
		static_cast<uint32_t>(b));

	for (auto& p : cloud.points) {
		p.rgb = *reinterpret_cast<float*>(&rgb);
	}
}

void phc::utils::PtcloudSetColor(pcl::PointCloud<pcl::PointXYZRGB>& cloud, const pcl::Indices& pt_indies, uint32_t r, uint32_t g, uint32_t b) {
	uint32_t rgb = (static_cast<uint32_t>(r) << 16 |
		static_cast<uint32_t>(g) << 8 |
		static_cast<uint32_t>(b));

	for (const auto& idx : pt_indies) {
		cloud.points[idx].rgb = *reinterpret_cast<float*>(&rgb);
	}
}

void phc::utils::PtcloudSetColorJetMap(pcl::PointCloud<pcl::PointXYZRGB>& cloud, const pcl::Indices& pt_indices, const std::vector<float>& jet_val) {
	for (int i = 0; i < jet_val.size(); ++i) {
		float jval = jet_val[i];
		pcl::index_t idx = pt_indices[i];

		uint8_t r, g, b;
		ColorMapJet(jval, 0.0f, 1.0f, r, g, b);

		uint32_t rgb = (static_cast<uint32_t>(r) << 16 |
			static_cast<uint32_t>(g) << 8 |
			static_cast<uint32_t>(b));

		cloud.points[idx].rgb = *reinterpret_cast<float*>(&rgb);
	}
}

void phc::utils::FindCVMat32FMinMax(const cv::Mat& mat, float& min, float& max) {
	CHECK_EQ(mat.type(), CV_32F);

	float depth_min = std::numeric_limits<float>::max();
	float depth_max = std::numeric_limits<float>::min();

	// Values < 0.0f is considered as invalid
	for (int u = 0; u < mat.cols; ++u) {
		for (int v = 0; v < mat.rows; ++v) {
			const float& depth = mat.at<float>(v, u);
			if (depth < 0.0f) { continue; }

			if (depth < depth_min) { depth_min = depth; }
			if (depth > depth_max) { depth_max = depth; }
		}
	}

	min = depth_min;
	max = depth_max;
}

cv::Vec3b phc::utils::MapValToBGR(float min, float max, float val) {
	// RGB
	static std::array<cv::Vec3b, 3> colors{
		cv::Vec3b(0, 0, 255),  // Red
		cv::Vec3b(0, 255, 0),  // Green
		cv::Vec3b(255, 0, 0)   // Blue
	};

	float i_f = (val - min) / (max - min) * (colors.size() - 1);

	int i = static_cast<int>(i_f);
	float f = fmod(i_f, 1);

	if (f < std::numeric_limits<float>::epsilon()) {
		return colors[i];
	}
	else {
		cv::Vec3b lower = colors[i];
		cv::Vec3b upper = colors[i + 1];

		return cv::Vec3b{
			static_cast<uchar>(lower[0] + f * (upper[0] - lower[0])),
			static_cast<uchar>(lower[1] + f * (upper[1] - lower[1])),
			static_cast<uchar>(lower[2] + f * (upper[2] - lower[2]))
		};

	}
}

Eigen::Matrix<float, 6, 1> phc::utils::IsometryToXYZRPY(const Eigen::Isometry3f& iso) {
	using Vector6f = Eigen::Matrix<float, 6, 1>;

	Vector6f xyzrpy;

	Eigen::Vector3f xyz = iso.translation();
	Eigen::Vector3f rpy = iso.rotation().eulerAngles(0, 1, 2);

	xyzrpy(0) = xyz(0);
	xyzrpy(1) = xyz(1);
	xyzrpy(2) = xyz(2);
	xyzrpy(3) = rpy(0);
	xyzrpy(4) = rpy(1);
	xyzrpy(5) = rpy(2);

	return xyzrpy;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr phc::utils::ConcatClouds(const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& clouds, const std::vector<Eigen::Isometry3f>& T_wl) {
	using pcl::PointCloud;
	using pcl::PointXYZI;

	// Must equal in size
	CHECK_EQ(clouds.size(), T_wl.size());
	
	PointCloud<PointXYZI>::Ptr result(new PointCloud<PointXYZI>);

	for (size_t i = 0; i < clouds.size(); ++i) {
		const PointCloud<PointXYZI>& cloud = *clouds[i];
		const Eigen::Isometry3f& _T_wl = T_wl[i];

		PointCloud<PointXYZI> cloud_cp;
		pcl::copyPointCloud(cloud, cloud_cp);

		// Transform to world frame and concat
		pcl::transformPointCloud(cloud_cp, cloud_cp, _T_wl.matrix());
		*result += cloud_cp;
	}

	return result;
}

inline float phc::utils::EuclideanDistToOrigin(float x, float y, float z) {
	return Eigen::Vector3f(x, y, z).norm();
}

Eigen::Quaternionf phc::utils::EulerToQuat(float x, float y, float z) {
	using Eigen::Quaternionf;
	using Eigen::AngleAxisf;
	using Eigen::Vector3f;

	Quaternionf q;
	q = AngleAxisf(x, Vector3f::UnitX())
		* AngleAxisf(y, Vector3f::UnitY())
		* AngleAxisf(z, Vector3f::UnitZ());

	return q;
}

void phc::utils::RandSampleInterestPts(std::vector<pcl::index_t>& poi, const pcl::PointCloud<pcl::PointXYZI>& cloud, const float& sample_ratio) {
	int min = 0;
	int max = cloud.size() - 1;
	int sample_num = cloud.size() * sample_ratio;

	poi = Rand::UniformDistInt(min, max, sample_num);
}

void phc::utils::GenGridIsometry3f(std::vector<Eigen::Isometry3f>& grid, const Eigen::Isometry3f& center, int step_num, float rot_step, float trans_step) {
	static constexpr float kDim = 6;
	
	// step = 1 -> [-1, 0, 1]
	const int num_per_dimension = 1 + 2 * step_num;
	const int start_num = 0 - step_num;
	const int grid_size = std::pow(num_per_dimension, kDim);

	grid.clear();
	grid.reserve(grid_size);

	std::vector<int> loc(num_per_dimension);
	std::iota(loc.begin(), loc.end(), start_num);

	Eigen::Vector3f init_trans(center.translation());
	Eigen::Vector3f init_rot(center.rotation().eulerAngles(0, 1, 2));

	float rot_step_size_rad = rot_step * kDeg2Rad;

	for (int tran_x = 0; tran_x < loc.size(); ++tran_x) {
		for (int tran_y = 0; tran_y < loc.size(); ++tran_y) {
			for (int tran_z = 0; tran_z < loc.size(); ++tran_z) {
				for (int rot_x = 0; rot_x < loc.size(); ++rot_x) {
					for (int rot_y = 0; rot_y < loc.size(); ++rot_y) {
						for (int rot_z = 0; rot_z < loc.size(); ++rot_z) {
							Eigen::Vector3f trans(static_cast<float>(init_trans.x() + loc[tran_x] * trans_step),
								static_cast<float>(init_trans.y() + loc[tran_y] * trans_step),
								static_cast<float>(init_trans.z() + loc[tran_z] * trans_step));
							Eigen::Vector3f rot(static_cast<float>(init_rot.x() + loc[rot_x] * rot_step_size_rad),
								static_cast<float>(init_rot.y() + loc[rot_y] * rot_step_size_rad),
								static_cast<float>(init_rot.z() + loc[rot_z] * rot_step_size_rad));

							Eigen::Isometry3f iso(utils::EulerToQuat(rot.x(), rot.y(), rot.z()));
							iso.pretranslate(trans);

							grid.push_back(iso);
						}
					}
				}
			}
		}
	}
}

Eigen::Vector2f phc::utils::SobelGradientAtSubPixel(const cv::Mat& img, const Eigen::Vector2f& pixel) {
	Eigen::Matrix3f kernel_x;
	kernel_x << 
		-1.0f, 0.0f, 1.0f, 
		-2.0f, 0.0f, 2.0f, 
		-1.0f, 0.0f, 1.0f;

	Eigen::Matrix3f kernel_y;
	kernel_y <<
		-1.0f, -2.0f, -1.0f,
		0.0f, 0.0f, 0.0f,
		1.0f, 2.0f, 1.0f;

	Eigen::Matrix3f pixel_window;
	pixel_window <<
		GetSubPixelValBilinear(img, Eigen::Vector2f(pixel.x() - 1, pixel.y() - 1)),
		GetSubPixelValBilinear(img, Eigen::Vector2f(pixel.x(), pixel.y() - 1)), 
		GetSubPixelValBilinear(img, Eigen::Vector2f(pixel.x() + 1, pixel.y() - 1)),
		GetSubPixelValBilinear(img, Eigen::Vector2f(pixel.x() - 1, pixel.y())),
		GetSubPixelValBilinear(img, Eigen::Vector2f(pixel.x(), pixel.y())),
		GetSubPixelValBilinear(img, Eigen::Vector2f(pixel.x() + 1, pixel.y())), 
		GetSubPixelValBilinear(img, Eigen::Vector2f(pixel.x() - 1, pixel.y() + 1)),
		GetSubPixelValBilinear(img, Eigen::Vector2f(pixel.x(), pixel.y() + 1)),
		GetSubPixelValBilinear(img, Eigen::Vector2f(pixel.x() + 1, pixel.y() + 1));

	return Eigen::Vector2f(
		(kernel_x.array() * pixel_window.array()).sum(),
		(kernel_y.array() * pixel_window.array()).sum());
}

Eigen::Matrix<float, 6, 1> phc::utils::Vec6Average(const std::vector<Eigen::Matrix<float, 6, 1>>& in) {
	using Vector6f = Eigen::Matrix<float, 6, 1>;

	Vector6f sum;
	for (const auto& vec : in) {
		sum += vec;
	}

	return sum / static_cast<float>(in.size());
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr phc::utils::ColorPointCloud(const cv::Mat& img, const pcl::PointCloud<pcl::PointXYZI>& cloud, const Eigen::Matrix3f& intri, const Eigen::Isometry3f& T_cl) {
	using pcl::PointCloud;
	using pcl::PointXYZRGB;
	using pcl::PointXYZI;

	PointCloud<PointXYZRGB>::Ptr result(new PointCloud<PointXYZRGB>);
	pcl::copyPointCloud(cloud, *result);

	for (size_t i = 0; i < cloud.size(); ++i) {
		const PointXYZI& pt_xyzi = cloud.at(i);
		PointXYZRGB& pt_xyzrgb = result->at(i);

		Eigen::Vector2f pixel;
		ProjectPoint(Eigen::Vector3f(pt_xyzi.x, pt_xyzi.y, pt_xyzi.z), pixel, T_cl, intri);

		float val = GetSubPixelValBilinear(img, pixel);

		uint32_t rgb = (static_cast<uint32_t>(val) << 16 |
			static_cast<uint32_t>(val) << 8 |
			static_cast<uint32_t>(val));

		pt_xyzrgb.rgb = *reinterpret_cast<float*>(&rgb);
	}

	return result;
}

std::vector<Eigen::Isometry3f> phc::utils::SingleAxisIsoGrid(float rot_step, int rstep_num, float trans_step, int tstep_num, const std::string& axis) {
	Eigen::Vector3f unit_rot;
	Eigen::Vector3f unit_trans;

	std::vector<Eigen::Isometry3f> grid;

	if (axis == "x") {
		unit_rot = Eigen::Vector3f::UnitX();
		unit_trans = Eigen::Vector3f::UnitX();
	}
	else if (axis == "y") {
		unit_rot = Eigen::Vector3f::UnitY();
		unit_trans = Eigen::Vector3f::UnitY();
	}
	else if (axis == "z") {
		unit_rot = Eigen::Vector3f::UnitZ();
		unit_trans = Eigen::Vector3f::UnitZ();
	}
	else {
		LOG(FATAL) << "Given axis not defined";
	}

	float rot_step_rad = kDeg2Rad * rot_step;

	for (int i = 0; i < rstep_num; ++i) {
		for (int j = 0; j < tstep_num; ++j) {
			Eigen::Vector3f rot = i * rot_step_rad * unit_rot;
			Eigen::Vector3f trans = j * trans_step * unit_trans;

			Eigen::Isometry3f iso(Eigen::Matrix3f::Identity());
			iso.translate(trans);
			iso.rotate(EulerToQuat(rot.x(), rot.y(), rot.z()));

			grid.push_back(iso);
		}
	}

	return grid;
}

std::vector<Eigen::Isometry3f> phc::utils::RandIsoGrid(float rot_step, int rstep_num, float trans_step, int tstep_num) {
	std::vector<Eigen::Isometry3f> grid;
	for (int i = 0; i < rstep_num; ++i) {
		for (int j = 0; j < tstep_num; ++j) {
			Eigen::Isometry3f iso = utils::Rand::Isometry3fFixErr(rot_step * (float)i, trans_step * (float)j);

			grid.push_back(iso);
		}
	}

	return grid;
}

void phc::utils::ProjectPtcloudToImg(const pcl::PointCloud<pcl::PointXYZI>& ptcloud, const Eigen::Isometry3f& extri, const Eigen::Matrix3f& intri_mat,
	int img_width, int img_height, std::vector<Eigen::Vector2f>& res) {

	res.clear();
	res.reserve(ptcloud.size());

	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_cam(new pcl::PointCloud<pcl::PointXYZI>);

	pcl::transformPointCloud(ptcloud, *cloud_cam, extri.matrix());

	for (const auto& pt : *cloud_cam) {
		if (pt.z < 0.0f) {
			// LOG(WARNING) << "Point behind camera";
			// LOG(WARNING) << pt;
			continue;
		}

		Eigen::Vector2f pixel;
		ProjectPoint(Eigen::Vector3f(pt.x, pt.y, pt.z), pixel, Eigen::Isometry3f::Identity(), intri_mat);
		
		if (pixel.x() < 0.0f || pixel.x() > img_width || pixel.y() < 0.0f || pixel.y() > img_height) {
			// LOG(WARNING) << "Point outside image";
			// LOG(WARNING) << pixel;
			continue;
		}
		
		res.push_back(pixel);
	}
}

void phc::utils::ProjectPtcloudToImg(const pcl::PointCloud<pcl::PointXYZI>& ptcloud, const Eigen::Isometry3f& extri, const Eigen::Matrix3f& intri_mat, cv::Mat& res) {
	CHECK(!res.empty());
	
	std::vector<Eigen::Vector2f> pixels;
	ProjectPtcloudToImg(ptcloud, extri, intri_mat, res.cols, res.rows, pixels);
	for (const Eigen::Vector2f& p : pixels) {
		ImgMarkPixel(res, p, "");
	}
}

void phc::utils::ColorMapJet(float v, float vmin, float vmax, uint8_t& r, uint8_t& g, uint8_t& b) {
	r = 255;
	g = 255;
	b = 255;

	if (v < vmin) {
		v = vmin;
	}

	if (v > vmax) {
		v = vmax;
	}

	double dr, dg, db;

	if (v < 0.05) { //   0.1242
		db = 0.504 + ((1. - 0.504) / 0.1242) * v;
		dg = dr = 0.;
	}
	else if (v < 0.2) { // 0.3747
		db = 1.;
		dr = 0.;
		dg = (v - 0.1242) * (1. / (0.3747 - 0.1242));
	}
	else if (v < 0.3) { // 0.6253
		db = (0.6253 - v) * (1. / (0.6253 - 0.3747));
		dg = 1.;
		dr = (v - 0.3747) * (1. / (0.6253 - 0.3747));
	}
	else if (v < 0.8) { // 0.8758
		db = 0.;
		dr = 1.;
		dg = (0.8758 - v) * (1. / (0.8758 - 0.6253));
	}
	else {
		db = 0.;
		dg = 0.;
		dr = 1. - (v - 0.8758) * ((1. - 0.504) / (1. - 0.8758));
	}

	r = (uint8_t)(255 * dr);
	g = (uint8_t)(255 * dg);
	b = (uint8_t)(255 * db);
}

Eigen::Matrix<float, 6, 1> phc::utils::Jacobian(float fx, float fy, float x, float y, float z, float gx, float gy) {
	Eigen::Matrix<float, 6, 1> jac;
	jac << fx * gx / z, fy* gy / z, (-x * fx * gx - y * fy * gy) / (z*z),
		(-x * y * fx * gx - fy * gy * (y*y + z*z)) / (z*z), (x * y * fy * gy + fx * gx * (x*x + z*z)) / (z*z), (-y * fx * gx + x * fy * gy) / z;

	return jac;
}

void phc::utils::JacobianNormCompute(const pcl::PointCloud<pcl::PointXYZI>& ptcloud, const pcl::PointIndices& indices, const Eigen::Isometry3f& T_cam_cloud, const cv::Mat& image, const Eigen::Matrix3f& intri, std::vector<float>& res) {
	using pcl::PointXYZI;
	using pcl::PointCloud;
	
	CHECK(!ptcloud.empty());
	CHECK(!image.empty());

	res.clear();
	res.reserve(indices.indices.size());

	PointCloud<PointXYZI>::Ptr extracted(new PointCloud<PointXYZI>);

	pcl::ExtractIndices<PointXYZI> extract;
	extract.setInputCloud(ptcloud.makeShared());
	extract.setIndices(std::make_shared<pcl::PointIndices>(indices));
	extract.setNegative(false);
	extract.filter(*extracted);

	std::vector<Eigen::Vector2f> pixels;
	ProjectPtcloudToImg(*extracted, T_cam_cloud, intri, image.cols, image.rows, pixels);

	LOG(INFO) << "Projected pixels: " << pixels.size();
	LOG(INFO) << "Extracted points: " << extracted->size();

	CHECK(pixels.size() == extracted->size());

	float fx = intri(0, 0);
	float fy = intri(1, 1);
	// float cx = intri(0, 2);
	// float cy = intri(1, 2);
	for (size_t i = 0; i < extracted->size(); ++i) {
		const Eigen::Vector2f &pixel = pixels[i];
		const PointXYZI &pt = extracted->at(i);
		Eigen::Vector2f gradient = SobelGradientAtSubPixel(image, pixel);
		float norm = Jacobian(fx, fy, pt.x, pt.y, pt.z, gradient.x(), gradient.y()).norm();
		res.push_back(norm);
	}
}

void phc::utils::RGBCloudImgProjection(const pcl::PointCloud<pcl::PointXYZRGB>& ptcloud, const pcl::PointIndices& indices, const Eigen::Isometry3f& T_cam_cloud, const cv::Mat& image, const Eigen::Matrix3f& intri, cv::Mat &out) {
	using pcl::PointXYZRGB;
	using pcl::PointXYZI;
	using pcl::PointCloud;
	
	CHECK(!ptcloud.empty());
	CHECK(!image.empty());

	cv::cvtColor(image, out, cv::COLOR_GRAY2BGR);

	PointCloud<PointXYZRGB>::Ptr extracted(new PointCloud<PointXYZRGB>);

	pcl::ExtractIndices<PointXYZRGB> extract;
	extract.setInputCloud(ptcloud.makeShared());
	extract.setIndices(std::make_shared<pcl::PointIndices>(indices));
	extract.setNegative(false);
	extract.filter(*extracted);

	PointCloud<PointXYZI>::Ptr extract_xyzi(new PointCloud<PointXYZI>);
	pcl::copyPointCloud(*extracted, *extract_xyzi);

	std::vector<Eigen::Vector2f> pixels;
	ProjectPtcloudToImg(*extract_xyzi, T_cam_cloud, intri, image.cols, image.rows, pixels);

	// LOG(INFO) << pixels.size() << " " << extracted->size();
	CHECK(pixels.size() == extract_xyzi->size());

	cv::Mat proj_img = cv::Mat::zeros(image.size(), CV_8UC3);
	for (size_t i = 0; i < extracted->size(); ++i) {
		const Eigen::Vector2f &pixel = pixels[i];
		const PointXYZRGB &pt = extracted->at(i);
		ImgMarkPixel(proj_img, pixel, pt.r, pt.g, pt.b);
	}

	cv::addWeighted(out, 0.5, proj_img, 0.6, 0.0, out);
}

void phc::utils::NormalizeVector(std::vector<float>& vec) {
	Eigen::Map<Eigen::ArrayXf> arr(vec.data(), vec.size());
	float max = arr.maxCoeff();
	float min = arr.minCoeff();
	arr = (arr - min) / (max - min);
}

void phc::utils::DumpFloatListToFile(const std::vector<float>& data, const std::string& path) {
	std::ofstream fout(path);
	for (const float& d : data) {
		fout << d << std::endl;
	}
	fout.close();
}

std::vector<Eigen::Isometry3f> phc::utils::LoadPertubationFile(const std::string& path) {
	std::vector<Eigen::Isometry3f> results;
	
	LOG(INFO) << "Loading pertubation file: " << path << " ...";

	std::ifstream fin(path);
	while (!fin.eof()) {
		std::string line; // tx ty tz qx qy qz qw
		std::getline(fin, line);
		if (line.empty()) {
			break;
		}

		LOG(INFO) << line;

		std::stringstream ss(line);
		Eigen::Vector3f t;
		Eigen::Quaternionf q;
		ss >> t.x() >> t.y() >> t.z() >> q.x() >> q.y() >> q.z() >> q.w();

		Eigen::Isometry3f T(q);
		T.pretranslate(t);
		results.push_back(T);
	}

	LOG(INFO) << "Loading pertubation file: " << path << " ... Done";
	
	return results;
}