#include <opencv2/core/mat.hpp>

#include <pcl/common/transforms.h>
#include <pcl/common/io.h>

#include "CovisDataPool.hpp"

using phc::CovisDataPool;

class CovisDataPool::Impl {
public:
	Impl() : cloud_(new PtCloudXYZI_t) {}

	PtId_t AddPt(const PtXYZI_t& pt) {
		cloud_->push_back(pt);
		return (cloud_->size() - 1);
	}

	FrameId_t AddImgWithPose(const cv::Mat &img, const Eigen::Isometry3f &T_cw) {
		// CHECK(!img.empty());
		imgs_.push_back(std::make_pair(img, T_cw));
		return (imgs_.size() - 1);
	}

	PtCloudXYZI_t::Ptr cloud_;

	std::vector<std::pair<cv::Mat, Eigen::Isometry3f>> imgs_; // In T_cw

	Eigen::Isometry3f T_cl_;
	bool extri_set_flag = false;
};

CovisDataPool::CovisDataPool() : impl_(new Impl) {
	
}

CovisDataPool::~CovisDataPool() {

}

bool CovisDataPool::Empty() const {
	return impl_->imgs_.empty();
}

void CovisDataPool::Add(const CovisGraph& cg, const PtCloudXYZI_t& cloud_local, const std::vector<DataFrame>& data_local, const Eigen::Isometry3f &T_cl, CovisGraph& cg_out) {
	std::unordered_map<PtId_t, PtId_t> pid_map;
	std::unordered_map<FrameId_t, FrameId_t> fid_map;

	if (!impl_->extri_set_flag) {
		impl_->T_cl_ = T_cl;
		impl_->extri_set_flag = true;
	}

	std::vector<PtId_t> pt_indices;
	cg.GetAllPts(pt_indices);
	for (const auto& pt_idx : pt_indices) {
		PtId_t new_pt_id = impl_->AddPt(cloud_local.at(pt_idx));
		pid_map.insert(std::make_pair(pt_idx, new_pt_id));
	}

	std::vector<FrameId_t> f_indices;
	cg.GetAllFrames(f_indices);
	for (const auto& f_idx : f_indices) {
		Eigen::Isometry3f T_cw = T_cl * data_local[f_idx].T_wl.inverse();

		// LOG(INFO) << "f_idx: " << f_idx;
		// LOG(INFO) << "data_local size: " << data_local.size();
		// CHECK(!data_local.at(f_idx).img.empty());
		FrameId_t new_fid = impl_->AddImgWithPose(data_local[f_idx].img, T_cw);
		fid_map.insert(std::make_pair(f_idx, new_fid));
	}

	// New covis graph with id mapped to global space
	for (const auto& pid : pt_indices) {
		std::unordered_set<FrameId_t> vis_f;
		cg.FindAllVisibleFrames(pid, vis_f);
		for (const FrameId_t &fid : vis_f) {
			cg_out.Insert(fid_map[fid], pid_map[pid]);
		}
	}
}

const phc::PtXYZI_t& CovisDataPool::GetPt(const PtId_t& pid) const{
	return impl_->cloud_->at(pid);
}

size_t CovisDataPool::CloudSize() const {
	return impl_->cloud_->size();
}

size_t CovisDataPool::ImgNum() const {
	return impl_->imgs_.size();
}

phc::PtCloudXYZI_t::Ptr CovisDataPool::GetCloud() const{
	PtCloudXYZI_t::Ptr res(new PtCloudXYZI_t);
	pcl::copyPointCloud(*impl_->cloud_, *res);
	return res;
}

const std::pair<cv::Mat, Eigen::Isometry3f>& CovisDataPool::GetImgWithPose(const FrameId_t& f_id) const{
	return impl_->imgs_.at(f_id);
}

const cv::Mat& CovisDataPool::GetImg(const FrameId_t& f_id) const {
	return impl_->imgs_.at(f_id).first;
}

void CovisDataPool::GetImgs(const std::unordered_set<FrameId_t>& fids, std::vector<cv::Mat>& result) const {
	for (const auto &id : fids) {
		result.push_back(impl_->imgs_.at(id).first);
	}
}

void CovisDataPool::GetImgsWithPose(const std::vector<FrameId_t>& fids, std::vector<std::pair<cv::Mat, Eigen::Isometry3f>>& result) const {
	for (const auto &fid : fids) {
		result.push_back(impl_->imgs_.at(fid));
	}
}

const std::vector<phc::ImgWithPose_t>& CovisDataPool::AllImgsWithPose() const {
	return impl_->imgs_;
}

void CovisDataPool::Clear() {
	impl_->cloud_->clear();
	impl_->imgs_.clear();
}

const Eigen::Isometry3f& CovisDataPool::GetExtri() const {
	return impl_->T_cl_;
}