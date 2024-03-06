#pragma once

#include <memory>

namespace phc {
	class PhotoCaliExRange {
	public:
		PhotoCaliExRange();
		~PhotoCaliExRange();

		void SetCamIntri(float fx, float fy, float cx, float cy);
		void SetExtri(const Eigen::Isometry3f& T_cl);
		void SetDataLoader(kitti::KittiLoader::Ptr loader_ptr);

		void OptRandSampleSingleLvlPyr(size_t start_idx, size_t num_total, size_t chunk_size, int chunk_num, unsigned int pyr_lvl);
	private:
		class Impl;
		std::unique_ptr<Impl> impl_;
	};
}