#pragma once

#include <vector>
#include <string>
#include <memory>

#include <Eigen/Core>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/core.hpp>

#include "mem2disk/Mem2Disk.hpp"

namespace phc {
	namespace utils {
		class PyrGradientDataBag : public m2d::IDataObj {
		public:
			using PyrGradDataType = std::vector<std::vector<float>>;
			using PyrImgDataType = std::vector<cv::Mat>;
			using Ptr = std::shared_ptr<PyrGradientDataBag>;

			PyrGradientDataBag(const std::string &name);

			void AddData(const PyrImgDataType &img, const PyrGradDataType &data);

			void DumpToDisk() override;

		private:

			std::string name_;

			std::vector<PyrGradDataType> data_;
			std::vector<PyrImgDataType> imgs_;
		};
		class GridSearchDataBag : public m2d::IDataObj {
		public:
			using Ptr = std::shared_ptr<GridSearchDataBag>;
			using Vector6f = Eigen::Matrix<float, 6, 1>;

			GridSearchDataBag(const std::string& name);
			void AddData(int iter, const Vector6f& err);

			void DumpToDisk() override;

		private:
			std::vector<std::pair<int, Vector6f>> data_;
			std::string name_;
		};
		class CovisDataBag : public m2d::IDataObj {
		public:
			using Ptr = std::shared_ptr<CovisDataBag>;
			CovisDataBag(const std::string& name);

			void SetPtcloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud);
			void AddImg(const cv::Mat &img);
			void AddPtDetail(const std::vector<cv::Vec2i> &loc, const std::vector<float> &val, const std::vector<cv::Mat> &imgs);

			void DumpToDisk() override;

		private:
			struct PtDetail {
				std::vector<cv::Vec2i> pixel_loc;
				std::vector<float> pixel_val;
				std::vector<cv::Mat> imgs;
			};

			void DumpSinglePtDetail(const std::string &path, const PtDetail &detail);

			pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_;
			std::vector<cv::Mat> imgs_;

			std::vector<PtDetail> pt_details_;

			std::string name_;
		};
		class ErrCompDataBag : public m2d::IDataObj {
		public:
			using Ptr = std::shared_ptr<ErrCompDataBag>;
			
			ErrCompDataBag(const std::string &name);

			void AddOriginDetail(const std::vector<cv::Vec2i>& loc, const std::vector<float>& val, const std::vector<cv::Mat>& imgs);
			void AddErrDetail(const std::vector<cv::Vec2i>& loc, const std::vector<float>& val, const std::vector<cv::Mat>& imgs);

			void DumpToDisk() override;
		private:
			struct PtDetail {
				std::vector<cv::Vec2i> pixel_loc;
				std::vector<float> pixel_val;
				std::vector<cv::Mat> imgs;
			};

			void DumpSinglePtDetail(const std::string& path, const PtDetail& detail);

			std::string name_;

			std::vector<PtDetail> origin_detail_;
			std::vector<PtDetail> err_detail_;
		};
		class WorsePercentDataBag : public m2d::IDataObj {
		public:
			using Ptr = std::shared_ptr<WorsePercentDataBag>;

			WorsePercentDataBag(const std::string &name);

			void AddData(int index, float worse_percent);

			void DumpToDisk() override;
		private:
			std::vector<std::pair<int, float>> data_;
			std::string name_;
		};

		class FrameNumWithFinalErrDataBag : public m2d::IDataObj {
		public:
			using Ptr = std::shared_ptr<FrameNumWithFinalErrDataBag>;
			using Vector6f = Eigen::Matrix<float, 6, 1>;

			FrameNumWithFinalErrDataBag(const std::string &name);

			// err: x_t, y_t, z_t, x_r, y_r, z_r
			void AddData(int frame_num, const Vector6f &err);

			void DumpToDisk() override;
		private:
			std::string name_;
			std::vector<std::pair<int, Vector6f>> data_;
		};

		class Vec3DataBag : public m2d::IDataObj {
		public:
			using Ptr = std::shared_ptr<Vec3DataBag>;

			Vec3DataBag(const std::string& name);

			void AddData(float x, float y, float z);

			void DumpToDisk() override;
		private:
			std::string name_;

			std::vector<Eigen::Vector3f> data_;
		};

		class TimeSpanMeasureDataBag : public m2d::IDataObj {
		public:
			using Ptr = std::shared_ptr<TimeSpanMeasureDataBag>;

			TimeSpanMeasureDataBag(const std::string &name);

			void TimerStart();
			void TimerStopAndRecord();

			void DumpToDisk() override;
		private:
			std::string name_;
			std::vector<double> data_;

			std::chrono::steady_clock::time_point start_;
		};
	}
}