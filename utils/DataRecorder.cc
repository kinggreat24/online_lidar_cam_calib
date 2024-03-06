#include <iostream>
#include <fstream>

#include <pcl/io/pcd_io.h>

#include <glog/logging.h>

#include <opencv2/imgcodecs.hpp>

#include "DataRecorder.hpp"
#include "Utils.hpp"

using phc::utils::DataRecorder;
using phc::utils::ResInfoId_t;

void DataRecorder::StartNewPtSession() {
	pts_info_.push_back(SinglePtInfo{});
	current_idx_ = pts_info_.size() - 1;
}

void DataRecorder::PtSessionSetGlobPtIdx(const pcl::index_t& idx) {
	pts_info_[current_idx_].glob_cloud_idx = idx;
}

void DataRecorder::PtSessionSetCost(const float& c) {
	pts_info_[current_idx_].cost = c;
}

void DataRecorder::PtSessionAddProjInfo(const ProjInfo& info) {
	pts_info_[current_idx_].projs.push_back(info);
}

void DataRecorder::SetDumpPath(const std::string& path) {
	using fsp = std::filesystem::path;
	using std::filesystem::exists;
	using std::filesystem::create_directory;

	dp_root_ = fsp(path);

	if (!exists(dp_root_)) {
		create_directory(dp_root_);
	}
}

void DataRecorder::GetRecordedPtIdx(std::vector<pcl::index_t>& indices) {
	indices.clear();
	for (const auto& pt : pts_info_) {
		indices.push_back(pt.glob_cloud_idx);
	}
}

void DataRecorder::GetInfoWithGlobCloudIdx(SinglePtInfo& info, pcl::index_t idx) {
	for (const auto& pt : pts_info_) {
		if (pt.glob_cloud_idx == idx) {
			info = pt;
		}
	}
}

void DataRecorder::DumpInfo() const {
	CHECK(!dp_root_.string().empty());
	for (const auto &info : pts_info_) {
		DumpInfoSinglePt(info);
	}
	DumpVisCloud();
}

void DataRecorder::WriteTextInfo(const std::string& file_path, const SinglePtInfo &info) const {
	std::ofstream out_stream(file_path);

	out_stream << "glob_cloud_idx: " << info.glob_cloud_idx << std::endl;
	out_stream << "cost: " << info.cost << std::endl;
	out_stream << "-----" << std::endl;
	for (const auto &proj : info.projs) {
		const uchar& pixel_val = proj.img.at<uchar>(static_cast<int>(proj.subpixel.y()), static_cast<int>(proj.subpixel.x()));
		out_stream << "frame_id: " << proj.frame_id << std::endl;
		out_stream << "subpixel: " << proj.subpixel << std::endl;
		out_stream << "Interpolated subpixel val: " << proj.sp_val_interp << std::endl;
		out_stream << "Pixel val: " << static_cast<int>(pixel_val) << std::endl;
		out_stream << "--" << std::endl;
	}

	out_stream.close();
}

void DataRecorder::DumpInfoSinglePt(const SinglePtInfo& info) const {
	using fsp = std::filesystem::path;
	using std::filesystem::exists;

	const std::string ptinfo_dir("points_info");

	fsp info_path = dp_root_ / ptinfo_dir / std::to_string(info.glob_cloud_idx);
	if (!exists(info_path)) {
		std::filesystem::create_directories(info_path);
	}

	fsp txt_file_path = info_path / fsp(kTextFileName_);
	WriteTextInfo(txt_file_path.string(), info);

	for (const auto& proj : info.projs) {
		fsp img_file_path = info_path / fsp(std::to_string(proj.frame_id) + ".png");

		cv::Mat out_img;
		utils::ImgMarkPixel(proj.img, out_img, proj.subpixel, std::to_string(proj.sp_val_interp));
		cv::imwrite(img_file_path.string(), out_img);
	}
}

void DataRecorder::AddVisibleCloudForFrames(const std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr>& clouds) {
	vis_clouds_ = clouds;
}

void DataRecorder::DumpVisCloud() const {
	namespace fs = std::filesystem;
	using fs::exists;

	const std::string viscloud_dir("vis_cloud");

	fs::path clouds_path = dp_root_ / viscloud_dir;
	if (!exists(clouds_path)) {
		fs::create_directories(clouds_path);
	}

	for (size_t fid = 0; fid < vis_clouds_.size(); ++fid) {
		fs::path file_path = clouds_path / fs::path(std::to_string(fid) + ".pcd");
		pcl::io::savePCDFileBinary(file_path.string(), *vis_clouds_[fid]);
	}
}

ResInfoId_t DataRecorder::SetNewResidualInfo(const std::vector<double>& residual, const std::string& name) {
	ResidualInfo r_info;
	r_info.residuals = residual;
	r_info.name = name;

	res_info_pool_.push_back(r_info);

	return res_info_pool_.size() - 1;
}