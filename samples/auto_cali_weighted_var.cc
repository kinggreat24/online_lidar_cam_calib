#include <glog/logging.h>

#include <pcl/io/pcd_io.h>

#include "kitti_loader/KittiLoader.hpp"

#include "PhotoCali.hpp"
#include "Config.hpp"
#include "StopWatch.hpp"
#include "DataRecorder.hpp"
#include "Evaluator.hpp"

phc::Vector6f IsometryToXYZRPY(const Eigen::Isometry3f& iso);
Eigen::Isometry3f GetPertubationForExtri();

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

    // Set dataset info
    Eigen::Isometry3f origin_extri = kitti_dataset.GetExtrinsics();
    kitti::Intrinsics kitti_intri = kitti_dataset.GetLeftCamIntrinsics();
    calibrator.SetCamIntri(kitti_intri.fx, kitti_intri.fy, kitti_intri.cx, kitti_intri.cy);

    // Set initial extrinsics
    Eigen::Isometry3f init_extri;
    if (phc::utils::Config::Get<int>("extri_error.enable")) {
        // Pertubation for extrinsics
        Eigen::Isometry3f extri_ptb = GetPertubationForExtri();
        init_extri = extri_ptb * origin_extri;
        LOG(INFO) << "Apply error to loaded extrinsics: ON";
    }
    else {
        init_extri = origin_extri;
        LOG(INFO) << "Apply error to loaded extrinsics: OFF";
    }

    calibrator.SetExtri(init_extri);

    int start_f_idx = phc::utils::Config::Get<int>("start_frame_idx");
    int f_num = phc::utils::Config::Get<int>("frame_num");
    for (int i = start_f_idx; i < start_f_idx + f_num; ++i) {
        kitti::Frame f = kitti_dataset[i];
        Eigen::Isometry3f T_wl = f.gt_pose * origin_extri;

        calibrator.AddDataFrame(f.ptcloud, f.left_img, T_wl);
    }

    // phc::utils::DataRecorder::Ptr redisual_recorder(new phc::utils::DataRecorder);
    // calibrator.SetRecorder(redisual_recorder);

    // Set residual block remove percent for optimizer
    // calibrator.SetResBlkRemovePercent(phc::utils::Config::Get<double>("optimizer.res_remove_percent"));

    // Compute cost
    phc::utils::StopWatch sw;
    Eigen::Isometry3f optimized_extri;
    // calibrator.OptimizeSingleLvlPyrWeightedVar(optimized_extri, phc::utils::Config::Get<int>("optimizer.pyramid_level"));
    calibrator.OptimizeSingleLvlPyrRepeated(optimized_extri, phc::utils::Config::Get<int>("optimizer.pyramid_level"));
    LOG(INFO) << "Time used: " << sw.GetTimeElapse();

    phc::Vector6f origin_extri_vec6 = IsometryToXYZRPY(origin_extri);
    phc::Vector6f opt_extri_vec6 = IsometryToXYZRPY(optimized_extri);

    LOG(INFO) << "Original extrinsics: \n" << origin_extri_vec6;
    LOG(INFO) << "Optimized extrinsics: \n" << opt_extri_vec6;
    LOG(INFO) << "Distance norm: " << (origin_extri_vec6 - opt_extri_vec6).norm();

    phc::utils::Evaluator evaluator(origin_extri, init_extri, optimized_extri, calibrator.GetIntriMat());
    kitti::Frame eva_f = kitti_dataset[888];
    evaluator.CompareNorm();
    evaluator.ShowProjectionOnImg(eva_f.left_img, *eva_f.ptcloud);

    // const std::vector<phc::utils::ResidualInfo> &res_info = redisual_recorder->GetAllResudialInfo();
    // for (const phc::utils::ResidualInfo &info : res_info) {
    //     LOG(INFO) << "Plotting: " << info.name;
    //     evaluator.PlotHist(info.residuals);
    // }

    return EXIT_SUCCESS;
}

phc::Vector6f IsometryToXYZRPY(const Eigen::Isometry3f& iso) {
    phc::Vector6f xyzrpy;

    Eigen::Vector3f xyz = iso.translation();
    Eigen::Vector3f rpy = iso.rotation().eulerAngles(0, 1, 2);

    xyzrpy(0) = xyz(0);
    xyzrpy(1) = xyz(1);
    xyzrpy(2) = xyz(2);
    xyzrpy(3) = rpy(0);
    xyzrpy(4) = rpy(1);
    xyzrpy(5) = rpy(2);

    return xyzrpy;
}

Eigen::Isometry3f GetPertubationForExtri() {
    using Eigen::Isometry3f;
    using Eigen::Vector3f;
    using Eigen::AngleAxisf;
    using Eigen::Quaternionf;
    using phc::utils::Config;

    float err_rot_x = Config::Get<float>("extri_error.rot.x");
    float err_rot_y = Config::Get<float>("extri_error.rot.y");
    float err_rot_z = Config::Get<float>("extri_error.rot.z");

    float err_trans_x = Config::Get<float>("extri_error.trans.x");
    float err_trans_y = Config::Get<float>("extri_error.trans.y");
    float err_trans_z = Config::Get<float>("extri_error.trans.z");

    LOG(INFO) << "Extri pertub:";
    LOG(INFO) << "rotation xyz (in degree): " << err_rot_x << " " << err_rot_y << " " << err_rot_z;
    LOG(INFO) << "translation xyz (in meter): " << err_trans_x << " " << err_trans_y << " " << err_trans_z;

    // Perturbation for extriinsics
    Quaternionf q_ptb = AngleAxisf(M_PI / 180.0 * err_rot_x, Vector3f::UnitX())
        * AngleAxisf(M_PI / 180.0 * err_rot_y, Vector3f::UnitY())
        * AngleAxisf(M_PI / 180.0 * err_rot_z, Vector3f::UnitZ());

    Isometry3f extri_ptb(q_ptb);
    extri_ptb.pretranslate(
        Vector3f(
            err_trans_x,
            err_trans_y,
            err_trans_z));

    return extri_ptb;
}