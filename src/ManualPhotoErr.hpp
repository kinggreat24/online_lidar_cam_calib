#pragma once

#include "CovisDataPool.hpp"
#include "CovisGraph.hpp"

namespace phc {
	class ManualPhotoErr {
	public:
		ManualPhotoErr(const CovisDataPool& dp, const CovisGraph& cg, const CamIntri& intri);

		inline void SetPyramidLevel(int lvl) { pyr_lvl_ = lvl; }
		inline void SetExtri(const Eigen::Isometry3f& T_cl) { T_cl_ = T_cl; }
		inline void SetTransMask(const std::unordered_set<PtId_t>& mask) { trans_mask_ = mask; }

		void Compute(std::vector<float> &costs_rot, std::vector<float> &costs_trans);
		void Compute(float &cost);

	private:
		enum CostComputeOption{
			kComputeRotationCost = 0,
			kComputeTranslationCost = 1
		};

		float SinglePtCost(const pcl::index_t& pt_id, CostComputeOption opt);
		bool PtInTransMask(const PtId_t &ptid);

		int pyr_lvl_ = 0;

		CamIntri intri_;

		Eigen::Isometry3f T_cl_;

		std::unordered_set<PtId_t> trans_mask_;

		const CovisDataPool& dp_;
		const CovisGraph& cg_;
	};
}