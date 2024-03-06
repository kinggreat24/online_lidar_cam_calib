#include <vector>

#include <glog/logging.h>

#include <CovisGraph.hpp>

using phc::CovisGraph;

void CovisGraph::Insert(const FrameId_t& f_id, const PtId_t& pt_id) {
	using std::make_pair;

	// Handle frame_id -> point_id map
	if (f_pt_map_.find(f_id) == f_pt_map_.end()) { // New frame id encountered
		std::unordered_set<PtId_t> pt_set;
		pt_set.insert(pt_id);
		f_pt_map_.insert(make_pair(f_id, pt_set));
	}
	else { // Known frame id
		std::unordered_set<PtId_t>& pt_set = f_pt_map_.at(f_id);
		pt_set.insert(pt_id);
	}

	// Handle point_id -> frame_id map
	if (pt_f_map_.find(pt_id) == pt_f_map_.end()) { // New point id encountered
		std::unordered_set<FrameId_t> f_set;
		f_set.insert(f_id);
		pt_f_map_.insert(make_pair(pt_id, f_set));
	}
	else { // Known point id 
		std::unordered_set<FrameId_t>& f_set = pt_f_map_.at(pt_id);
		f_set.insert(f_id);
	}
}

void CovisGraph::ErasePt(const PtId_t& pt_id) {
	// Handle point_id -> frame_id map
	pt_f_map_.erase(pt_id);

	// Handle frame_id -> point_id map
	for (auto& f_pt : f_pt_map_) {
		// std::unordered_set<PtId_t>& pt_set = f_pt_map_.at(f_pt.first);
		// pt_set.erase(pt_id);
		f_pt.second.erase(pt_id);
	}
}

void CovisGraph::Clear() {
	pt_f_map_.clear();
	f_pt_map_.clear();
}

void CovisGraph::EraseLessObservedPt(const int& thresh) {
	std::vector<PtId_t> pts_to_delete;
	for (const auto &pt_f : pt_f_map_) {
		if (pt_f.second.size() < thresh) {
			pts_to_delete.push_back(pt_f.first);
		}
	}

	for (const PtId_t& pt : pts_to_delete) {
		ErasePt(pt);
	}
}

void CovisGraph::FindAllVisiblePts(const FrameId_t& f_id, std::unordered_set<PtId_t>& result) const {
	// CHECK(f_pt_map_.find(f_id) != f_pt_map_.end());
	
	result.clear();
	if (f_pt_map_.find(f_id) != f_pt_map_.end()) {
		result = f_pt_map_.at(f_id);
	}
}

void CovisGraph::FindAllVisibleFrames(const PtId_t& pt_id, std::unordered_set<FrameId_t>& result) const {
	// CHECK(pt_f_map_.find(pt_id) != pt_f_map_.end());

	result.clear();
	if (pt_f_map_.find(pt_id) != pt_f_map_.end()) {
		result = pt_f_map_.at(pt_id);
	}
}

void CovisGraph::FindAllVisibleFrames(const PtId_t& pt_id, std::vector<FrameId_t>& result) const{
	result.clear();
	if (pt_f_map_.find(pt_id) != pt_f_map_.end()) {
		std::unordered_set<FrameId_t> fids = pt_f_map_.at(pt_id);
		
		result.reserve(fids.size());
		for (const auto &fid : fids) {
			result.push_back(fid);
		}
	}
}

void CovisGraph::GetAllPts(std::vector<PtId_t>& result) const {
	result.clear();
	result.reserve(pt_f_map_.size());
	for (const auto &pt_f : pt_f_map_) {
		result.push_back(pt_f.first);
	}
}

void CovisGraph::GetAllFrames(std::vector<FrameId_t>& result) const {
	result.clear();
	result.reserve(f_pt_map_.size());
	for (const auto& f_pt : f_pt_map_) {
		result.push_back(f_pt.first);
	}
}

void CovisGraph::Merge(const CovisGraph& cg) {
	std::vector<FrameId_t> frame_ids;
	cg.GetAllFrames(frame_ids);
	for (const FrameId_t& f_id : frame_ids) {
		std::unordered_set<PtId_t> pt_ids;
		cg.FindAllVisiblePts(f_id, pt_ids);
		for (const PtId_t& pt_id : pt_ids) {
			Insert(f_id, pt_id);
		}
	}
}

void CovisGraph::SetRotWeight(const FrameId_t& f_id, const PtId_t& pt_id, float weight){
	std::pair<FrameId_t, PtId_t> p(f_id, pt_id);
	weights_rot_[p] = weight;
}

void CovisGraph::SetTransWeight(const FrameId_t& f_id, const PtId_t& pt_id, float weight) {
	std::pair<FrameId_t, PtId_t> p(f_id, pt_id);
	weights_trans_[p] = weight;
}

void CovisGraph::GenTransWeightMask(std::unordered_set<PtId_t>& mask) {
	CHECK(!weights_trans_.empty());

	for (const auto &ptid_fs : pt_f_map_) {
		const PtId_t& pt_id = ptid_fs.first;
		const std::unordered_set<FrameId_t>& f_ids = ptid_fs.second;

		std::vector<float> weights;
		weights.reserve(f_ids.size());
		for (const auto &f_id : f_ids) {
			const float& w = weights_trans_.at(std::make_pair(f_id, pt_id));
			weights.push_back(w);
		}

		// We use 0.5 as threshold since the translation weight is either 1.0 or 0.0
		int non_zero_cnt =
			std::count_if(weights.begin(), weights.end(), 
				[](const float& w) {
					return w > 0.5f;
				});

		if (non_zero_cnt < 2) {
			mask.insert(pt_id);
		}
	}
}

float CovisGraph::QueryWeightRot(const FrameId_t& fid, const PtId_t& ptid) const {
	return weights_rot_.at(std::make_pair(fid, ptid));
}

float CovisGraph::QueryWeightTrans(const FrameId_t& fid, const PtId_t& ptid) const {
	return weights_trans_.at(std::make_pair(fid, ptid));
}

void CovisGraph::QueryWeightRotPoint(const PtId_t& pt_id, std::vector<FrameId_t>& vis_frames, std::vector<float>& weights) const {
	// Query visible frames
	FindAllVisibleFrames(pt_id, vis_frames);

	// Query weights
	weights.reserve(vis_frames.size());
	for (const FrameId_t& f_id : vis_frames) {
		std::pair<FrameId_t, PtId_t> fid_ptid = std::make_pair(f_id, pt_id);
		float w = weights_rot_.at(fid_ptid);
		weights.push_back(w);
	}
}

void CovisGraph::QueryWeightTransPoint(const PtId_t& pt_id, std::vector<FrameId_t>& vis_frames, std::vector<float>& weights) const {
	// Query visible frames
	FindAllVisibleFrames(pt_id, vis_frames);

	// Query weights
	weights.reserve(vis_frames.size());
	for (const FrameId_t& f_id : vis_frames) {
		std::pair<FrameId_t, PtId_t> fid_ptid = std::make_pair(f_id, pt_id);
		float w = weights_trans_.at(fid_ptid);
		weights.push_back(w);
	}
}