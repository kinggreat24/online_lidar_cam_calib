#include <glog/logging.h>

#include <pcl/io/pcd_io.h>

#include "kitti_loader/KittiLoader.hpp"

#include "PhotoCali.hpp"
#include "Config.hpp"
#include "StopWatch.hpp"
#include "DataRecorder.hpp"

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

    // Init dataset loader
    bool if_success = false;
    kitti::KittiLoader kitti_dataset(phc::utils::Config::Get<std::string>("dataset_path"), if_success);
    CHECK(if_success);

    // Calibrator instance
    phc::PhotoCali calibrator;

    // Set recorder
    phc::utils::DataRecorder::Ptr recorder(new phc::utils::DataRecorder);
    calibrator.SetRecorder(recorder);

    // Set dataset info
    Eigen::Isometry3f origin_extri = kitti_dataset.GetExtrinsics();
    kitti::Intrinsics kitti_intri = kitti_dataset.GetLeftCamIntrinsics();
    calibrator.SetExtri(origin_extri);
    calibrator.SetCamIntri(kitti_intri.fx, kitti_intri.fy, kitti_intri.cx, kitti_intri.cy);

    int start_f_idx = phc::utils::Config::Get<int>("start_frame_idx");
    int f_num = phc::utils::Config::Get<int>("frame_num");
    for (int i = start_f_idx; i < start_f_idx + f_num; ++i) {
        kitti::Frame f = kitti_dataset[i];
        Eigen::Isometry3f T_wl = f.gt_pose * origin_extri;

        calibrator.AddDataFrame(f.ptcloud, f.left_img, T_wl);
    }

    // Set sample ratio
    calibrator.SetRandSampleRatio(0.001f);

    // Compute cost
    phc::utils::StopWatch sw;
    float cost = calibrator.ComputeCost();

    LOG(INFO) << "Time used: " << sw.GetTimeElapse();
    LOG(INFO) << "Total cost: " << cost;

    recorder->SetDumpPath(phc::utils::Config::Get<std::string>("data_dump_path"));
    recorder->DumpInfo();

    return EXIT_SUCCESS;
}