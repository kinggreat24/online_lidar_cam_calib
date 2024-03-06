#include <glog/logging.h>

#include <pcl/common/io.h>
#include <pcl/io/pcd_io.h>

#include "kitti_loader/KittiLoader.hpp"

#include "PhotoCali.hpp"
#include "Config.hpp"
#include "Utils.hpp"

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

    LOG(INFO) << config_path;

    bool if_success = false;
    kitti::KittiLoader kitti_dataset(phc::utils::Config::Get<std::string>("dataset_path"), if_success);
    CHECK(if_success);

    kitti::Frame frame = kitti_dataset[0];
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud = frame.ptcloud;

    float max_dist = -std::numeric_limits<float>::max();
    float min_dist = std::numeric_limits<float>::max();
    for (pcl::PointXYZI &pt : cloud->points) {
        pt.intensity = pt.getVector3fMap().norm();

        // LOG(INFO) << pt;

        if (pt.intensity < min_dist) { min_dist = pt.intensity; }
        if (pt.intensity > max_dist) { max_dist = pt.intensity; }
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr vis_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::copyPointCloud(*cloud, *vis_cloud);

    for (size_t i = 0; i < cloud->size(); ++i) {
        const pcl::PointXYZI& pt_xyzi = cloud->at(i);
        cv::Vec3b bgr = phc::utils::MapValToBGR(min_dist, max_dist, pt_xyzi.intensity);

        uint32_t rgb = (static_cast<uint32_t>(bgr[2]) << 16 |
            static_cast<uint32_t>(bgr[1]) << 8 |
            static_cast<uint32_t>(bgr[0]));

        pcl::PointXYZRGB& pt_xyzrgb = vis_cloud->at(i);
        pt_xyzrgb.rgb = *reinterpret_cast<float*>(&rgb);
    }

    pcl::io::savePCDFile("D:\\cloud.pcd", *vis_cloud);

    std::ofstream txt_file("D:\\dist.txt");
    for (const pcl::PointXYZI& pt : cloud->points) {
        txt_file << pt.intensity << std::endl;
    }
    txt_file.close();

    return EXIT_SUCCESS;
}