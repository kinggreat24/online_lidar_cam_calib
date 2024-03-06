#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/mat.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace phc {
	namespace utils {
		class Evaluator {
		public:
			Evaluator(const Eigen::Isometry3f& origin_extri, const Eigen::Isometry3f& init_extri, const Eigen::Isometry3f& result_extri, const Eigen::Matrix3f& intri_mat)
				: origin_extri_(origin_extri), init_extri_(init_extri), result_extri_(result_extri), intri_mat_(intri_mat) {}

			void ShowProjectionOnImg(const cv::Mat& img, const pcl::PointCloud<pcl::PointXYZI>& ptcloud) const;
			
			void CompareNorm() const;

			void GetOriginResultErr(Eigen::Matrix<float, 6, 1> &err);

			static void CostCompare(const std::vector<float>& err_costs);
			static void DistRotTrans(const Eigen::Isometry3f &extri_1, const Eigen::Isometry3f& extri_2, float &dist_trans, float &dist_rot);
			static void PlotHist(const std::vector<double>& residuals);
			static void PlotHist(const std::vector<float>& residuals);

		private:

			cv::Mat PtcloudToRangeMap(const pcl::PointCloud<pcl::PointXYZI>& ptcloud, const Eigen::Isometry3f& T_cl, int width, int height) const;
			cv::Mat RangeMapToBGR(const cv::Mat& map) const;

			Eigen::Isometry3f origin_extri_;
			Eigen::Isometry3f init_extri_;
			Eigen::Isometry3f result_extri_;

			Eigen::Matrix3f intri_mat_;
		};
	}
}