#pragma once

#include <vector>
#include <iterator>

#include <unordered_map>
#include <unordered_set>

#include <boost/functional/hash.hpp>

#include <pcl/types.h>

#include "Types.hpp"

namespace phc {
	// Consider provide an iterator
	// https://www.internalpointers.com/post/writing-custom-iterators-modern-cpp
	// https://www.fluentcpp.com/2018/04/27/tag-dispatching/
	class CovisGraph {
	public:
		CovisGraph() = default;

		inline bool Empty() const { return f_pt_map_.empty(); }
		inline bool Weighted() const { return !weights_rot_.empty(); }

		void Insert(const FrameId_t &f_id, const PtId_t &pt_id);
		void ErasePt(const PtId_t& pt_id);
		void Clear();

		void Merge(const CovisGraph &cg);

		// Erase points with less visible frames than given threshold
		void EraseLessObservedPt(const int &thresh);

		// After weighting, some points will be assigned with a 0 weight on translation.
		// If a point only has 1 non-zero weight across all observations,
		// then there is no need to add the point to optimization.
		// This function generate a mask containing all points with 1 or 0 non-zero translation weight
		// Will not clear the input mask, only add
		void GenTransWeightMask(std::unordered_set<PtId_t> &mask);

		void FindAllVisiblePts(const FrameId_t &f_id, std::unordered_set<PtId_t> &result) const;
		void FindAllVisibleFrames(const PtId_t& pt_id, std::unordered_set<FrameId_t>& result) const;
		void FindAllVisibleFrames(const PtId_t& pt_id, std::vector<FrameId_t>& result) const;

		void GetAllPts(std::vector<PtId_t> &result) const;
		void GetAllFrames(std::vector<FrameId_t>& result) const;

		void SetRotWeight(const FrameId_t& f_id, const PtId_t& pt_id, float weight);
		void SetTransWeight(const FrameId_t& f_id, const PtId_t& pt_id, float weight);

		float QueryWeightRot(const FrameId_t &fid, const PtId_t &ptid) const;
		float QueryWeightTrans(const FrameId_t& fid, const PtId_t& ptid) const;

		// Query weight given a point only
		// Return visible frames and corresponding weights
		// vis_frames and weights are of same size, and the values are stored in corresponding position
		void QueryWeightRotPoint(const PtId_t &pt_id, std::vector<FrameId_t> &vis_frames, std::vector<float> &weights) const;
		void QueryWeightTransPoint(const PtId_t& pt_id, std::vector<FrameId_t>& vis_frames, std::vector<float>& weights) const;


	private:
		std::unordered_map<FrameId_t, std::unordered_set<PtId_t>> f_pt_map_;
		std::unordered_map<PtId_t, std::unordered_set<FrameId_t>> pt_f_map_;

		std::unordered_map<std::pair<FrameId_t, PtId_t>, float, boost::hash<std::pair<FrameId_t, PtId_t>>> weights_rot_;
		std::unordered_map<std::pair<FrameId_t, PtId_t>, float, boost::hash<std::pair<FrameId_t, PtId_t>>> weights_trans_;
	};

}