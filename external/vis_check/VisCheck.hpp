#pragma once

#include <unordered_set>

#include <Eigen/Geometry>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>

namespace visc{

using Pt3d = pcl::PointXYZ;
using Pt2d = pcl::PointXY;
using Pt3dCloud = pcl::PointCloud<Pt3d>;
using Pt2dCloud = pcl::PointCloud<Pt2d>;

using PtIndices = pcl::PointIndices;

using RigidTransform6d = Eigen::Isometry3f;

struct CamIntrinsics{
    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
};

class VisCheck{
public:
    VisCheck() = default;

    inline void SetK(unsigned int k) { k_ = k; } // Set number of nearest neighbor for knn search
    inline void SetVisScoreThreshMeanShift(float ms) { mean_shift_ = ms; }
    inline void SetVisScoreThresh(float thresh) { vis_score_thresh_ = thresh; }
    inline void SetDiscardEdgeSize(unsigned int size) {discard_edge_size_ = size;}
    inline void SetMaxViewRange(float range) { max_view_range_ = range; }

    inline float GetMeanVisScore() const {return mean_vis_score_;}

    void SetInputCloud(const Pt3dCloud::ConstPtr cloud_ptr);
    void SetCamera(const CamIntrinsics& intri, const RigidTransform6d& pose, unsigned int img_width, unsigned int img_height);
    void ComputeVisibility(PtIndices& visible_pts);
    void ComputeVisibilityInterestIndices(PtIndices& visible_pts, const PtIndices& interest_indices);
private:
    float ComputeVisibilityScore(float d, float d_min, float d_max);
    float ComputeEuclideanDistToOrigin(const Pt3d& pt);
    float ComputeSquaredEuclideanDistToOrigin(const Pt3d& pt);

    std::unordered_set<pcl::index_t> PtIndice2Uset(const PtIndices& indices);

    void ProjectToImageSpace(Pt3dCloud::ConstPtr cloud_3d, Pt2dCloud::Ptr cloud_proj, std::vector<unsigned int>& indice);

    Pt3dCloud::ConstPtr cloud_3d_ptr_ = nullptr;

    RigidTransform6d cam_pose_; // Transform from pointcloud frame to camera frame
    CamIntrinsics cam_intri_;

    unsigned int img_width_ = 0;
    unsigned int img_height_ = 0;

    unsigned int discard_edge_size_ = 0; 

    unsigned int k_ = 7; // Number of nearest neighbor for knn search
    float vis_score_thresh_ = 0.95f;
    float mean_shift_ = 0.0f;
    float mean_vis_score_ = 0.0f;
    float max_view_range_ = -1.0f;
};

}