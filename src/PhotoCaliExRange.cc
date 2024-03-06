#include <glog/logging.h>

#include "kitti_loader/KittiLoader.hpp"
#include "Types.hpp"
#include "PhotoCaliExRange.hpp"

using kitti::KittiLoader;
using phc::PhotoCaliExRange;

class PhotoCaliExRange::Impl {
public:
	Impl() = default;


	KittiLoader::Ptr dataset_loader = nullptr;
	CamIntri intri;
	Eigen::Isometry3f extri;
};

PhotoCaliExRange::PhotoCaliExRange() : impl_(new Impl) {

}

PhotoCaliExRange::~PhotoCaliExRange() {
	
}

void PhotoCaliExRange::SetCamIntri(float fx, float fy, float cx, float cy){
	impl_->intri.fx = fx;
	impl_->intri.fy = fy;
	impl_->intri.cx = cx;
	impl_->intri.cy = cy;
}

void PhotoCaliExRange::SetExtri(const Eigen::Isometry3f& T_cl){
	impl_->extri = T_cl;
}

void PhotoCaliExRange::SetDataLoader(kitti::KittiLoader::Ptr loader_ptr) {
	CHECK(loader_ptr);
	impl_->dataset_loader = loader_ptr;
}

void PhotoCaliExRange::OptRandSampleSingleLvlPyr(size_t start_idx, size_t num_total, size_t chunk_size, int chunk_num, unsigned int pyr_lvl) {

}