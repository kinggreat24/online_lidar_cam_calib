#include <glog/logging.h>

#include <pcl/io/pcd_io.h>

#include "kitti_loader/KittiLoader.hpp"

#include "PhotoCali.hpp"
#include "Config.hpp"

int main(int argc, char const* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: ./kitti_loader <path-to-config-file>" << std::endl;
        return EXIT_FAILURE;
    }

    FLAGS_logtostderr = 1;
    google::InitGoogleLogging(argv[0]);
    LOG(INFO) << "GLOG inited";

    std::string config_path(argv[1]);
    if (!phc::utils::Config::SetFile(config_path)) {
        LOG(FATAL) << "Fail to load config file.";
    }

    bool if_success = false;
    kitti::KittiLoader kitti_dataset(phc::utils::Config::Get<std::string>("dataset_path"), if_success);
    CHECK(if_success);

    Eigen::Isometry3f origin_extri = kitti_dataset.GetExtrinsics();


    phc::PhotoCali calibrator;
    for (size_t i = 0; i < 200; ++i) {
        kitti::Frame f = kitti_dataset[i];
        // Eigen::Isometry3f T_wl = f.gt_pose * origin_extri;
        Eigen::Isometry3f T_wl = f.gt_pose;

        calibrator.AddDataFrame(f.ptcloud, f.left_img, T_wl);
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr concat_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    calibrator.GetConcatPtcloud(*concat_cloud);

    std::string cloud_path("D:/concat_cloud.pcd");
    pcl::io::savePCDFileBinary(cloud_path, *concat_cloud);

    return EXIT_SUCCESS;
}