#include <glog/logging.h>

#include <pcl/io/pcd_io.h>

#include "livox_loader/LivoxLoader.hpp"
#include "mem2disk/Mem2Disk.hpp"

#include "PhotoCaliOnline.hpp"
#include "Config.hpp"
#include "StopWatch.hpp"
#include "DataRecorder.hpp"
#include "Evaluator.hpp"

phc::Vector6f IsometryToXYZRPY(const Eigen::Isometry3f& iso);
Eigen::Isometry3f GetPertubationForExtri();
void LoadConfig(phc::OnlineCaliConf& conf);

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
    lvx::LivoxLoader livox_loader(phc::utils::Config::Get<std::string>("dataset_path"));

    // Load configuration 
    phc::OnlineCaliConf conf;
    LoadConfig(conf);

    // Calibrator instance
    phc::PhotoCaliOnline calibrator(conf);

    // Set dataset info
    Eigen::Isometry3f origin_extri = livox_loader.GetExtrinsic();
    lvx::Intrinsic livox_intri = livox_loader.GetIntrinsic();
    calibrator.SetCamIntrinsics(livox_intri.fx, livox_intri.fy, livox_intri.cx, livox_intri.cy);

    m2d::DataManager::SetRoot(phc::utils::Config::Get<std::string>("data_dump_path"));

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

    calibrator.SetInitExtri(origin_extri);

    int start_f_idx = phc::utils::Config::Get<int>("start_frame_idx");
    int f_num = phc::utils::Config::Get<int>("frame_num");
    for (int i = start_f_idx; i < start_f_idx + f_num; ++i) {
        lvx::Frame f = livox_loader[i];
        // Eigen::Isometry3f T_wl = f.gt_pose * origin_extri;
        Eigen::Isometry3f T_wl = f.T_wl;

        calibrator.AddDataFrame(f.ptcloud, f.img, T_wl);
    }

    calibrator.GenCovisDataBag("livox_covis");
    calibrator.GenPyrGradDataBag();
    // calibrator.GenErrCompDataBag("ErrComp", init_extri);

    m2d::DataManager::DumpAll();

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

void LoadConfig(phc::OnlineCaliConf& conf) {
    using phc::utils::Config;

    conf.ptcloud_clip_min = Config::Get<float>("calibrator.ptcloud_clip.min");
    conf.ptcloud_clip_max = Config::Get<float>("calibrator.ptcloud_clip.max");
    conf.sample_ratio = Config::Get<float>("calibrator.sample_ratio");

    // Weight computation parameter
    conf.err_tolr_x = Config::Get<float>("calibrator.err_tolr_x");
    conf.err_tolr_y = Config::Get<float>("calibrator.err_tolr_y");
    conf.trans_thresh_ratio = Config::Get<float>("calibrator.trans_thresh_ratio");

    // Minimum observation number threshold
    conf.obs_thresh = Config::Get<int>("calibrator.obs_thresh");

    conf.window_size = Config::Get<int>("calibrator.window_size");

    // Mode config
    int mode_number = Config::Get<int>("calibrator.computation_mode");
    if (mode_number == 0) {
        conf.mode = phc::OnlineCaliMode::kManualCost;
    }
    else if (mode_number == 1) {
        conf.mode = phc::OnlineCaliMode::kOptimize;
    }

    // Covisibility check parameters
    conf.covis_conf.max_view_range = Config::Get<float>("calibrator.covis_check.max_view_range");
    conf.covis_conf.score_thresh = Config::Get<float>("calibrator.covis_check.score_thresh");
    conf.covis_conf.edge_discard = Config::Get<int>("calibrator.covis_check.edge_discard");

    // Optimization parameter
    conf.opt_conf.start_frame_num = Config::Get<int>("calibrator.optimize.start_frame_num");
    conf.opt_conf.pyramid_lvl = Config::Get<int>("calibrator.optimize.pyramid_lvl");
    conf.opt_conf.max_iter = Config::Get<int>("calibrator.optimize.max_iter");
    conf.opt_conf.residual_reserve_percent = Config::Get<float>("calibrator.optimize.residual_reserve_percent");

    conf.pixel_val_lower_lim = Config::Get<float>("calibrator.pixel_val_lower_lim");
    conf.pixel_val_upper_lim = Config::Get<float>("calibrator.pixel_val_upper_lim");
}