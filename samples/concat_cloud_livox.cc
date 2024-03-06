#include <glog/logging.h>

#include <pcl/io/pcd_io.h>

#include "livox_loader/LivoxLoader.hpp"

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

    lvx::LivoxLoader livox_loader(phc::utils::Config::Get<std::string>("dataset_path"));

    Eigen::Isometry3f origin_extri = livox_loader.GetExtrinsic();


    phc::PhotoCali calibrator;
    for (size_t i = 0; i < 10; ++i) {
        lvx::Frame f = livox_loader[i];
        Eigen::Isometry3f T_wl = f.T_wl;

        calibrator.AddDataFrame(f.ptcloud, f.img, T_wl);
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr concat_cloud(new pcl::PointCloud<pcl::PointXYZI>);
    calibrator.GetConcatPtcloud(*concat_cloud);

    std::string cloud_path("D:/Datasets/concat_cloud.pcd");
    pcl::io::savePCDFileBinary(cloud_path, *concat_cloud);

    return EXIT_SUCCESS;
}