#pragma once

#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/mat.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>

namespace phc {
	namespace utils {
		void RemoveCloudPtsOutOfRange(pcl::PointCloud<pcl::PointXYZI> &ptcloud, float range_min, float range_max);
		pcl::PointCloud<pcl::PointXYZI>::Ptr ConcatClouds(const std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> &clouds, const std::vector<Eigen::Isometry3f> & T_wl);
		void RandSampleInterestPts(std::vector<pcl::index_t>& poi, const pcl::PointCloud<pcl::PointXYZI>& cloud, const float& sample_ratio);

		void ProjectPoint(const Eigen::Vector3f &pt, Eigen::Vector2f &pixel, const Eigen::Isometry3f &trans_mat, const Eigen::Matrix3f &intri_mat);
		void ProjectPtcloudToImg(const pcl::PointCloud<pcl::PointXYZI> &ptcloud, const Eigen::Isometry3f &extri, const Eigen::Matrix3f &intri_mat, int img_width, int img_height, std::vector<Eigen::Vector2f>& res);
		void ProjectPtcloudToImg(const pcl::PointCloud<pcl::PointXYZI> &ptcloud, const Eigen::Isometry3f &extri, const Eigen::Matrix3f &intri_mat, cv::Mat& res);

		void ImgMarkPixel(const cv::Mat &in, cv::Mat &out, const Eigen::Vector2f &subpixel, const std::string &text);
		void ImgMarkPixel(cv::Mat &img, const Eigen::Vector2f& subpixel, const std::string& text);
		void ImgMarkPixel(cv::Mat &img, const Eigen::Vector2f& subpixel, uint8_t r, uint8_t g, uint8_t b);

		float GetSubPixelValBilinear(const cv::Mat &img, const Eigen::Vector2f &pixel);
		float GetSubPixelRelValBilinear(const cv::Mat& img, const Eigen::Vector2f& pixel);
		float VarianceCompute(const Eigen::VectorXf &vec);
		float VarianceComputeWeighted(const Eigen::VectorXf& vec, const Eigen::VectorXf& weight);
		Eigen::Vector2f VarianceComputeWeighted(const Eigen::VectorXf& vec, const Eigen::VectorXf& weight_rot, const Eigen::VectorXf &weight_trans);

		void PtcloudSetColor(pcl::PointCloud<pcl::PointXYZRGB>& cloud, uint32_t r, uint32_t g, uint32_t b);
		void PtcloudSetColor(pcl::PointCloud<pcl::PointXYZRGB>& cloud, const pcl::Indices& pt_indies, uint32_t r, uint32_t g, uint32_t b);
		void PtcloudSetColorJetMap(pcl::PointCloud<pcl::PointXYZRGB>& cloud, const pcl::Indices& pt_indies, const std::vector<float>& jet_val);

		float EuclideanDistToOrigin(float x, float y, float z);

		void FindCVMat32FMinMax(const cv::Mat &mat, float &min, float &max);

		// https://stackoverflow.com/questions/20792445/calculate-rgb-value-for-a-range-of-values-to-create-heat-map
		cv::Vec3b MapValToBGR(float min, float max, float val);
		void ColorMapJet(float v, float vmin, float vmax, uint8_t& r, uint8_t& g, uint8_t& b);

		Eigen::Matrix<float, 6, 1> IsometryToXYZRPY(const Eigen::Isometry3f& iso);
		Eigen::Quaternionf EulerToQuat(float x, float y, float z);

		Eigen::Vector2f SobelGradientAtSubPixel(const cv::Mat& img, const Eigen::Vector2f& pixel);

		// Generate grid for isometry3f, center included
		// rot_step in degrees
		// trans_step in meters
		void GenGridIsometry3f(std::vector<Eigen::Isometry3f> &grid, const Eigen::Isometry3f &center, int step_num, float rot_step, float trans_step);

		Eigen::Matrix<float, 6, 1> Vec6Average(const std::vector<Eigen::Matrix<float, 6, 1>> &in);

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr ColorPointCloud(const cv::Mat& img, const pcl::PointCloud<pcl::PointXYZI>& cloud, const Eigen::Matrix3f& intri, const Eigen::Isometry3f& extri);

		std::vector<Eigen::Isometry3f> SingleAxisIsoGrid(float rot_step, int rstep_num, float trans_step, int tstep_num, const std::string& axis);
		std::vector<Eigen::Isometry3f> RandIsoGrid(float rot_step, int rstep_num, float trans_step, int tstep_num);

		Eigen::Matrix<float, 6, 1> Jacobian(float fx, float fy, float x, float y, float z, float gx, float gy);
		void JacobianNormCompute(const pcl::PointCloud<pcl::PointXYZI>& ptcloud, const pcl::PointIndices& indices, const Eigen::Isometry3f &T_cam_cloud,  const cv::Mat &image, const Eigen::Matrix3f& intri, std::vector<float> &res);

		void NormalizeVector(std::vector<float> &vec);

		void RGBCloudImgProjection(const pcl::PointCloud<pcl::PointXYZRGB>& ptcloud, const pcl::PointIndices& indices, const Eigen::Isometry3f& T_cam_cloud, const cv::Mat& image, const Eigen::Matrix3f& intri, cv::Mat &out);
	
		void DumpFloatListToFile(const std::vector<float> & data, const std::string &path);

		std::vector<Eigen::Isometry3f> LoadPertubationFile(const std::string &path);
	}
}