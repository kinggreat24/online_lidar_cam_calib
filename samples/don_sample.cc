#include <glog/logging.h>

#include <pcl/io/pcd_io.h>
#include <pcl/search/organized.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/don.h>

#include "kitti_loader/KittiLoader.hpp"

#include "Utils.hpp"
#include "Config.hpp"

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

    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> clouds;
    std::vector<Eigen::Isometry3f> poses;

    for (size_t i = 50; i < 70; ++i) {
        kitti::Frame f = kitti_dataset[i];
        Eigen::Isometry3f T_wl = f.gt_pose * extri;

        clouds.push_back(f.ptcloud);
        poses.push_back(T_wl);
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr glob_cloud = phc::utils::ConcatClouds(clouds, poses);

    pcl::search::Search<pcl::PointXYZI>::Ptr search_tree(new pcl::search::KdTree<pcl::PointXYZI>(false));
    search_tree->setInputCloud(glob_cloud);

    pcl::NormalEstimationOMP<pcl::PointXYZI, pcl::PointNormal> ne;
    ne.setInputCloud(glob_cloud);
    ne.setSearchMethod(search_tree);
    ne.setViewPoint(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max());

    float scale_small = phc::utils::Config::Get<float>("don.scale_small");
    float scale_large = phc::utils::Config::Get<float>("don.scale_large");
    float threshold = phc::utils::Config::Get<float>("don.threshold");

    // Compute normal on small scale
    LOG(INFO) << "Computing normals for scale: " << scale_small;
    pcl::PointCloud<pcl::PointNormal>::Ptr normal_small_scale(new pcl::PointCloud<pcl::PointNormal>);
    ne.setRadiusSearch(scale_small);
    ne.compute(*normal_small_scale);

    // Compute normal on large scale
    LOG(INFO) << "Computing normals for scale: " << scale_large;
    pcl::PointCloud<pcl::PointNormal>::Ptr normal_large_scale(new pcl::PointCloud<pcl::PointNormal>);
    ne.setRadiusSearch(scale_large);
    ne.compute(*normal_large_scale);

    // DoN compute
    pcl::PointCloud<pcl::PointNormal>::Ptr don_cloud(new pcl::PointCloud<pcl::PointNormal>);
    pcl::copyPointCloud(*glob_cloud, *don_cloud);

    pcl::DifferenceOfNormalsEstimation<pcl::PointXYZI, pcl::PointNormal, pcl::PointNormal> don;
    don.setInputCloud(glob_cloud);
    don.setNormalScaleLarge(normal_large_scale);
    don.setNormalScaleSmall(normal_small_scale);

    if (!don.initCompute()) {
        LOG(FATAL) << "Error: Could not initialize DoN feature operator";
    }

    don.computeFeature(*don_cloud);

    pcl::io::savePCDFileBinary("E://don.pcd", *don_cloud);

    pcl::ConditionOr<pcl::PointNormal>::Ptr range_cond(new pcl::ConditionOr<pcl::PointNormal>());
    range_cond->addComparison(pcl::FieldComparison<pcl::PointNormal>::ConstPtr(new pcl::FieldComparison<pcl::PointNormal>("curvature", pcl::ComparisonOps::LT, threshold)));

    pcl::ConditionalRemoval<pcl::PointNormal> cond_rem;
    cond_rem.setCondition(range_cond);
    cond_rem.setInputCloud(don_cloud);

    pcl::PointCloud<pcl::PointNormal>::Ptr doncloud_filtered(new pcl::PointCloud<pcl::PointNormal>);

    LOG(INFO) << "Original Pointcloud: " << don_cloud->size() << " data points.";

    cond_rem.filter(*doncloud_filtered);

    pcl::io::savePCDFileBinary("E://filtered.pcd", *doncloud_filtered);

    LOG(INFO) << "Filtered Pointcloud: " << doncloud_filtered->size() << " data points.";

    return EXIT_SUCCESS;
}