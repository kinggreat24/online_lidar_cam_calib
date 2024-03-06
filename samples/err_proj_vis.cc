#include <iostream>

#include <glog/logging.h>

#include "kitti_loader/KittiLoader.hpp"

#include "Config.hpp"
#include "Utils.hpp"
#include "Rand.hpp"

namespace {
    const bool kApplyErr = false;
}

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

    FLAGS_logtostderr = phc::utils::Config::Get<int>("glog_to_stderr");
    FLAGS_log_dir = phc::utils::Config::Get<std::string>("glog_directory");
    google::InitGoogleLogging(argv[0]);
    LOG(INFO) << "GLOG inited";

    // Init dataset loader
    bool if_success = false;
    kitti::KittiLoader kitti_dataset(phc::utils::Config::Get<std::string>("dataset_path"), if_success);
    CHECK(if_success);

    kitti::Intrinsics intri = kitti_dataset.GetLeftCamIntrinsics();
    Eigen::Matrix3f intri_mat;
    intri_mat <<
        intri.fx, 0.0f, intri.cx,
        0.0f, intri.fy, intri.cy,
        0.0f, 0.0f, 1.0f;

    Eigen::Isometry3f extri;
    if (kApplyErr) {
        std::vector<Eigen::Isometry3f> errs = phc::utils::Rand::UniformIsometry3f(2.0f, 0.1f, 1);
        extri = errs.front() * kitti_dataset.GetExtrinsics();
    }
    else {
        extri = kitti_dataset.GetExtrinsics();
    }

    int start_f_idx = phc::utils::Config::Get<int>("start_frame_idx");
    int f_num = phc::utils::Config::Get<int>("frame_num");

    for (int i = start_f_idx; i < start_f_idx + f_num; ++i) {
        kitti::Frame f = kitti_dataset[i];
        cv::Mat viz_img = f.left_img.clone();
        phc::utils::ProjectPtcloudToImg(*f.ptcloud, extri, intri_mat, viz_img);

        cv::imshow("img", viz_img);
        cv::waitKey(0);
    }
    

	return EXIT_SUCCESS;
}