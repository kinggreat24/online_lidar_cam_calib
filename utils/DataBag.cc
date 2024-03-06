#include <fstream>
#include <filesystem>

#include <glog/logging.h>

#include <pcl/io/pcd_io.h>

#include <opencv2/imgcodecs.hpp>

#include "DataBag.hpp"

using phc::utils::WorsePercentDataBag;
using phc::utils::FrameNumWithFinalErrDataBag;
using phc::utils::Vec3DataBag;
using phc::utils::CovisDataBag;
using phc::utils::GridSearchDataBag;
using phc::utils::PyrGradientDataBag;
using phc::utils::ErrCompDataBag;
using phc::utils::TimeSpanMeasureDataBag;

WorsePercentDataBag::WorsePercentDataBag(const std::string& name) 
	:name_(name) {

}

void WorsePercentDataBag::AddData(int index, float worse_percent) {
	data_.push_back(std::make_pair(index, worse_percent));
}

void WorsePercentDataBag::DumpToDisk() {
	namespace sf = std::filesystem;

	std::string root = m2d::DataManager::GetRoot();
	CHECK(!root.empty());

	sf::path file_path(root);
	file_path = file_path / sf::path(name_ + ".txt");

	std::ofstream fs(file_path.string());
	for (const auto &idx_data : data_) {
		fs << idx_data.first << " " << idx_data.second << std::endl;
	}

	fs.close();
}

FrameNumWithFinalErrDataBag::FrameNumWithFinalErrDataBag(const std::string& name) : name_(name){

}

void FrameNumWithFinalErrDataBag::AddData(int frame_num, const Vector6f& err) {
	data_.push_back(std::make_pair(frame_num, err));
}

void FrameNumWithFinalErrDataBag::DumpToDisk() {
	namespace sf = std::filesystem;

	std::string root = m2d::DataManager::GetRoot();
	CHECK(!root.empty());

	sf::path file_path(root);
	file_path = file_path / sf::path(name_ + ".txt");

	std::ofstream fs(file_path.string());
	for (const auto& fn_err : data_) {
		const Vector6f& err = fn_err.second;
		fs << fn_err.first
			<< " " << err(0) << " " << err(1) << " " << err(2)
			<< " " << err(3) << " " << err(4) << " " << err(5) << std::endl;
	}

	fs.close();
}

Vec3DataBag::Vec3DataBag(const std::string& name) : name_(name) {}

void Vec3DataBag::AddData(float x, float y, float z) {
	data_.push_back(Eigen::Vector3f(x, y, z));
}

void Vec3DataBag::DumpToDisk() {
	namespace sf = std::filesystem;
	using Eigen::Vector3f;

	std::string root = m2d::DataManager::GetRoot();
	CHECK(!root.empty());

	sf::path file_path(root);
	file_path = file_path / sf::path(name_ + ".txt");

	std::ofstream fs(file_path.string());
	for (const Vector3f& vec3 : data_) {
		fs << " " << vec3.x() << " " << vec3.y() << " " << vec3.z() << std::endl;
	}

	fs.close();
}

CovisDataBag::CovisDataBag(const std::string& name) : name_(name) {

}

void CovisDataBag::SetPtcloud(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud) {
	cloud_ = cloud;
}

void CovisDataBag::AddImg(const cv::Mat& img) {
	imgs_.push_back(img);
}

void CovisDataBag::AddPtDetail(const std::vector<cv::Vec2i>& loc, const std::vector<float>& val, const std::vector<cv::Mat>& imgs) {
	CHECK_EQ(loc.size(), val.size());
	CHECK_EQ(loc.size(), imgs.size());

	pt_details_.push_back(PtDetail{loc, val, imgs});
}

void CovisDataBag::DumpToDisk() {
	namespace sf = std::filesystem;

	CHECK(!cloud_->empty());
	CHECK(!imgs_.empty());

	std::string root = m2d::DataManager::GetRoot();
	CHECK(!root.empty());

	sf::path dump_path(root);
	dump_path = dump_path / sf::path(name_);

	if (!sf::exists(dump_path)) {
		sf::create_directories(dump_path);
	}
	
	// Dump point cloud
	sf::path cloud_file = dump_path / sf::path("cloud.pcd");
	pcl::io::savePCDFileBinary(cloud_file.string(), *cloud_);

	for (int i = 0; i < imgs_.size(); ++i) {
		sf::path img_file = dump_path / sf::path(std::to_string(i) + ".png");
		cv::imwrite(img_file.string(), imgs_[i]);
	}

	// Dump point detail
	sf::path detail_path = dump_path / sf::path("details");
	for (int i = 0; i < pt_details_.size(); ++i) {
		sf::path pt_path = detail_path / sf::path(std::to_string(i));
		DumpSinglePtDetail(pt_path.string(), pt_details_[i]);
	}
}

void CovisDataBag::DumpSinglePtDetail(const std::string& path, const PtDetail& detail) {
	namespace sf = std::filesystem;

	sf::path pt_path(path);
	if (!sf::exists(pt_path)) {
		sf::create_directories(pt_path);
	}

	sf::path txt_path = pt_path / sf::path("data.txt");
	std::ofstream fs(txt_path.string());
	for (int i = 0; i < detail.imgs.size(); ++i) {
		const cv::Vec2i& loc = detail.pixel_loc[i];
		const float& val = detail.pixel_val[i];
		fs << i << " "
			<< loc[0] << " " << loc[1] << " "
			<< val << std::endl;

		sf::path img_path = pt_path / sf::path(std::to_string(i) + ".png");
		cv::imwrite(img_path.string(), detail.imgs[i]);
	}
	fs.close();
}

GridSearchDataBag::GridSearchDataBag(const std::string& name) : name_(name) {}

void GridSearchDataBag::AddData(int iter, const Vector6f& err) {
	data_.push_back(std::make_pair(iter, err));
}

void GridSearchDataBag::DumpToDisk() {
	namespace sf = std::filesystem;

	std::string root = m2d::DataManager::GetRoot();
	CHECK(!root.empty());

	sf::path file_path(root);
	file_path = file_path / sf::path(name_ + ".txt");

	std::ofstream fs(file_path.string());
	for (const auto& fn_err : data_) {
		const Vector6f& err = fn_err.second;
		fs << fn_err.first
			<< " " << err(0) << " " << err(1) << " " << err(2)
			<< " " << err(3) << " " << err(4) << " " << err(5) << std::endl;
	}

	fs.close();
}

PyrGradientDataBag::PyrGradientDataBag(const std::string &name) : name_(name) {

}

void PyrGradientDataBag::AddData(const PyrImgDataType& img, const PyrGradDataType& data) {
	imgs_.push_back(img);
	data_.push_back(data);
}

void PyrGradientDataBag::DumpToDisk() {
	namespace sf = std::filesystem;

	std::string root = m2d::DataManager::GetRoot();
	CHECK(!root.empty());

	sf::path dump_path(root);
	dump_path = dump_path / name_;

	for (int i = 0; i < imgs_.size(); ++i) {
		sf::path data_path = dump_path / std::to_string(i);

		if (!sf::exists(data_path)) {
			sf::create_directories(data_path);
		}

		for (int lvl = data_[i].size() - 1; lvl >= 0 ; --lvl) {
			sf::path grad_path = data_path / (std::to_string(lvl) + ".txt");
			sf::path img_path = data_path / (std::to_string(lvl) + ".png");

			cv::imwrite(img_path.string(), imgs_[i][lvl]);
			
			std::ofstream fs(grad_path.string());
			for (const float &val : data_[i][lvl]) {
				fs << val << std::endl;
			}
			fs.close();
		}
	}
}

ErrCompDataBag::ErrCompDataBag(const std::string& name) : name_(name) {

}

void ErrCompDataBag::AddOriginDetail(const std::vector<cv::Vec2i>& loc, const std::vector<float>& val, const std::vector<cv::Mat>& imgs) {
	CHECK_EQ(loc.size(), val.size());
	CHECK_EQ(loc.size(), imgs.size());

	origin_detail_.push_back(PtDetail{ loc, val, imgs });
}

void ErrCompDataBag::AddErrDetail(const std::vector<cv::Vec2i>& loc, const std::vector<float>& val, const std::vector<cv::Mat>& imgs) {
	CHECK_EQ(loc.size(), val.size());
	CHECK_EQ(loc.size(), imgs.size());

	err_detail_.push_back(PtDetail{ loc, val, imgs });
}

void ErrCompDataBag::DumpToDisk() {
	namespace sf = std::filesystem;

	std::string root = m2d::DataManager::GetRoot();
	CHECK(!root.empty());

	sf::path dump_path(root);
	dump_path = dump_path / sf::path(name_);

	if (!sf::exists(dump_path)) {
		sf::create_directories(dump_path);
	}

	sf::path detail_path;
	
	// Dump origin detail
	detail_path = dump_path / sf::path("origin");
	for (int i = 0; i < origin_detail_.size(); ++i) {
		sf::path pt_path = detail_path / sf::path(std::to_string(i));
		DumpSinglePtDetail(pt_path.string(), origin_detail_[i]);
	}

	// Dump error detail
	detail_path = dump_path / sf::path("err");
	for (int i = 0; i < err_detail_.size(); ++i) {
		sf::path pt_path = detail_path / sf::path(std::to_string(i));
		DumpSinglePtDetail(pt_path.string(), err_detail_[i]);
	}
}

void ErrCompDataBag::DumpSinglePtDetail(const std::string& path, const PtDetail& detail) {
	namespace sf = std::filesystem;

	sf::path pt_path(path);
	if (!sf::exists(pt_path)) {
		sf::create_directories(pt_path);
	}

	sf::path txt_path = pt_path / sf::path("data.txt");
	std::ofstream fs(txt_path.string());
	for (int i = 0; i < detail.imgs.size(); ++i) {
		const cv::Vec2i& loc = detail.pixel_loc[i];
		const float& val = detail.pixel_val[i];
		fs << i << " "
			<< loc[0] << " " << loc[1] << " "
			<< val << std::endl;

		sf::path img_path = pt_path / sf::path(std::to_string(i) + ".png");
		cv::imwrite(img_path.string(), detail.imgs[i]);
	}
	fs.close();
}

TimeSpanMeasureDataBag::TimeSpanMeasureDataBag(const std::string &name) : name_(name) {
	
}

void TimeSpanMeasureDataBag::TimerStart() {
	start_ = std::chrono::steady_clock::now();
}

void TimeSpanMeasureDataBag::TimerStopAndRecord() {
	using std::chrono::steady_clock;
	using std::chrono::duration;
	using std::chrono::duration_cast;
	steady_clock::time_point end = steady_clock::now();
	data_.push_back(duration_cast<duration<double>>(end - start_).count());
}

void TimeSpanMeasureDataBag::DumpToDisk() {
	namespace sf = std::filesystem;

	std::string root = m2d::DataManager::GetRoot();
	CHECK(!root.empty());

	sf::path file_path(root);
	file_path = file_path / sf::path(name_ + ".txt");

	std::ofstream fs(file_path.string());
	for (const double& d : data_) {
		fs << d << std::endl;
	}

	fs.close();
}