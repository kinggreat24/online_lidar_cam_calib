#include <glog/logging.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/io.h>
#include <pcl/io/pcd_io.h>

#include "kitti_loader/KittiLoader.hpp"

#include "vis_check/VisCheck.hpp"

#include "Utils.hpp"
#include "Config.hpp"
#include "Rand.hpp"

namespace {
    const int kStartIdx = 30;
    const int kFrameNum = 10;
}

int main(int argc, char const* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: ./don_sample <path-to-config-file>" << std::endl;
        return EXIT_FAILURE;
    }

    FLAGS_logtostderr = 1;
    google::InitGoogleLogging(argv[0]);
    LOG(INFO) << "GLOG inited";

    std::string config_path(argv[1]);
    if (!phc::utils::Config::SetFile(config_path)) {
        LOG(FATAL) << "Fail to load config file.";
    }

    // Init dataset loader
    bool if_success = false;
    kitti::KittiLoader kitti_dataset(phc::utils::Config::Get<std::string>("dataset_path"), if_success);
    CHECK(if_success);

    Eigen::Isometry3f extri = kitti_dataset.GetExtrinsics();

    kitti::Intrinsics cam_intri = kitti_dataset.GetLeftCamIntrinsics();
    Eigen::Matrix3f cam_intri_mat;
    cam_intri_mat << cam_intri.fx, 0.0f, cam_intri.cx,
					 0.0f, cam_intri.fy, cam_intri.cy,
					 0.0f, 0.0f, 1.0f;

    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> clouds;
    std::vector<Eigen::Isometry3f> poses;

    int img_width = 0;
    int img_height = 0;

    for (size_t i = kStartIdx; i < kStartIdx + kFrameNum; ++i) {
        kitti::Frame f = kitti_dataset[i];
        Eigen::Isometry3f T_wl = f.gt_pose * extri;

        if (img_width == 0 || img_height == 0) {
            img_width = f.left_img.cols;
            img_height = f.right_img.rows;
        }

        phc::utils::RemoveCloudPtsOutOfRange(*f.ptcloud, 5.0f, 100.0f);

        clouds.push_back(f.ptcloud);
        poses.push_back(T_wl);
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr glob_cloud = phc::utils::ConcatClouds(clouds, poses);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*glob_cloud, *cloud_xyz);

    visc::VisCheck vis_checker;
    vis_checker.SetInputCloud(cloud_xyz);
    vis_checker.SetMaxViewRange(70.0f);
    vis_checker.SetVisScoreThresh(0.97f);
    vis_checker.SetDiscardEdgeSize(5);

    Eigen::Isometry3f T_cw = extri * poses.front().inverse();
    vis_checker.SetCamera(visc::CamIntrinsics{cam_intri.fx, cam_intri.fy, cam_intri.cx, cam_intri.cy}, T_cw, img_width, img_height);

    visc::PtIndices res;
    vis_checker.ComputeVisibility(res);

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_xyzi(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::copyPointCloud(*cloud_xyz, *cloud_xyzi);

    std::vector<float> jacobian_norm;
    phc::utils::JacobianNormCompute(*cloud_xyzi, res, T_cw, kitti_dataset[kStartIdx].left_img, cam_intri_mat, jacobian_norm);
    phc::utils::NormalizeVector(jacobian_norm);

    // phc::utils::DumpFloatListToFile(jacobian_norm, "D://jacobian_norm.txt");

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr vis_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*cloud_xyz, *vis_cloud);

    phc::utils::PtcloudSetColor(*vis_cloud, 141, 143, 140);
    //phc::utils::PtcloudSetColor(*vis_cloud, res.indices, 0, 255, 0);
    phc::utils::PtcloudSetColorJetMap(*vis_cloud, res.indices, jacobian_norm);

    pcl::io::savePCDFileBinary("D://vis_cloud.pcd", *vis_cloud);

    cv::Mat jacob_img;
    phc::utils::RGBCloudImgProjection(*vis_cloud, res, T_cw, kitti_dataset[kStartIdx].left_img, cam_intri_mat, jacob_img);

    cv::imwrite("D://jacob_img.png", jacob_img);

    return EXIT_SUCCESS;
}