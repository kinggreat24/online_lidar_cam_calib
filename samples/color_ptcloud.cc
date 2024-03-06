#include <iostream>

#include <glog/logging.h>

#include <pcl/io/pcd_io.h>

#include "kitti_loader/KittiLoader.hpp"

#include "Utils.hpp"
#include "Config.hpp"
#include "Types.hpp"

int main(int argc, char const* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: ./kitti_loader <path-to-config-file>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string config_path(argv[1]);
    if (!phc::utils::Config::SetFile(config_path)) {
        std::cerr << "Fail to load config file." << std::endl;
        return EXIT_FAILURE;
    }

    google::InitGoogleLogging(argv[0]);
    LOG(INFO) << "GLOG inited";

    // Init dataset loader
    bool if_success = false;
    kitti::KittiLoader kitti_dataset(phc::utils::Config::Get<std::string>("dataset_path"), if_success);
    CHECK(if_success);

    Eigen::Isometry3f origin_extri = kitti_dataset.GetExtrinsics();
    kitti::Intrinsics intri_kitti = kitti_dataset.GetLeftCamIntrinsics();

    phc::CamIntri intri{ intri_kitti.fx, intri_kitti.fy, intri_kitti.cx, intri_kitti.cy };
    
    kitti::Frame frame = kitti_dataset[888];
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr vis_cloud =
        phc::utils::ColorPointCloud(frame.left_img, *frame.ptcloud, intri.AsMat(), origin_extri);

    pcl::io::savePCDFileBinary("E://colored_cloud.pcd", *vis_cloud);

    return EXIT_SUCCESS;
}