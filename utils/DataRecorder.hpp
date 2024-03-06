#pragma once

#include <memory>
#include <vector>
#include <string>
#include <filesystem>

#include <Eigen/Core>

#include <pcl/types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/core/mat.hpp>

namespace phc {
	namespace utils {

		using ResInfoId_t = size_t;

		struct ResidualInfo {
			std::vector<double> residuals;
			std::string name;
		};

		struct ProjInfo {
			Eigen::Vector2f subpixel;
			cv::Mat img;
			size_t frame_id;
			float sp_val_interp; // Interpolated subpixel val
		};

		struct SinglePtInfo{
			pcl::index_t glob_cloud_idx;
			std::vector<ProjInfo> projs;
			float cost;
		};

		class DataRecorder {
		public:
			using Ptr = std::shared_ptr<DataRecorder>;

			DataRecorder() = default;

			inline std::string GetDumpPath() const { return dp_root_.string(); }

			inline void SetResidualBeforeOpt(const std::vector<double>& residual) {resdiual_bef_opt_ = residual;}
			inline void SetResidualAfterOpt(const std::vector<double>& residual) { residual_aft_opt_ = residual; }

			inline std::vector<double> GetResidualBeforeOpt() const {return resdiual_bef_opt_;}
			inline std::vector<double> GetResidualAfterOpt() const { return residual_aft_opt_; }

			inline const std::vector<ResidualInfo>& GetAllResudialInfo() { return res_info_pool_; }

			ResInfoId_t SetNewResidualInfo(const std::vector<double>& residual, const std::string &name);

			void SetDumpPath(const std::string &path);
			void DumpInfo() const;
			void DumpInfoSinglePt(const SinglePtInfo& info) const;
			void DumpVisCloud() const;

			void GetRecordedPtIdx(std::vector<pcl::index_t> &indices);
			void GetInfoWithGlobCloudIdx(SinglePtInfo&info, pcl::index_t idx);

			void StartNewPtSession();
			void PtSessionSetGlobPtIdx(const pcl::index_t &idx);
			void PtSessionSetCost(const float &c);
			void PtSessionAddProjInfo(const ProjInfo &info);

			void AddVisibleCloudForFrames(const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> &clouds);

			std::vector<float> costs_rot;
			std::vector<float> costs_trans;
		private:
			void WriteTextInfo(const std::string &file_path, const SinglePtInfo& info) const;

			const std::string kTextFileName_ = std::string("info.txt");

			size_t current_idx_ = 0;
			std::vector<SinglePtInfo> pts_info_;

			std::vector<ResidualInfo> res_info_pool_;

			std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> vis_clouds_;

			std::vector<double> resdiual_bef_opt_;
			std::vector<double> residual_aft_opt_;

			std::filesystem::path dp_root_;
		};
	}
}