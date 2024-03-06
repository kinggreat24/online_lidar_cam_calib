#include <glog/logging.h>

#include <pcl/io/pcd_io.h>

#include "kitti_loader/KittiLoader.hpp"

#include "PhotoCaliOnline.hpp"
#include "Config.hpp"
#include "StopWatch.hpp"
#include "DataRecorder.hpp"
#include "Evaluator.hpp"
#include "DataBag.hpp"
#include "Rand.hpp"

phc::Vector6f IsometryToXYZRPY(const Eigen::Isometry3f& iso);
Eigen::Isometry3f GetPertubationForExtri();
void LoadConfig(phc::OnlineCaliConf& conf);
void SingleExtrinsicCompute(Eigen::Isometry3f init_extri, const phc::OnlineCaliConf& conf, const kitti::KittiLoader& kitti_dataset, const std::string& name);

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

    // Set dataset info
    Eigen::Isometry3f origin_extri = kitti_dataset.GetExtrinsics();
    kitti::Intrinsics kitti_intri = kitti_dataset.GetLeftCamIntrinsics();

    // Set initial extrinsics
    std::vector<Eigen::Isometry3f> extri_errors = phc::utils::Rand::UniformIsometry3f(2.0f, 0.1f, 1);
    Eigen::Isometry3f init_extri = extri_errors.front() * origin_extri;

    // Set data dump root
    m2d::DataManager::SetRoot(phc::utils::Config::Get<std::string>("data_dump_path"));
    phc::utils::GridSearchDataBag::Ptr gs_bag(new phc::utils::GridSearchDataBag("grid_search"));

    // Load configuration 
    phc::OnlineCaliConf conf;
    LoadConfig(conf);

    // Calibrator instance
    phc::PhotoCaliOnline calibrator(conf);
    calibrator.SetCamIntrinsics(kitti_intri.fx, kitti_intri.fy, kitti_intri.cx, kitti_intri.cy);
    calibrator.SetInitExtri(init_extri);

    int start_f_idx = phc::utils::Config::Get<int>("start_frame_idx");
    int f_num = phc::utils::Config::Get<int>("frame_num");
    for (int i = start_f_idx; i < start_f_idx + f_num; ++i) {
        kitti::Frame f = kitti_dataset[i];
        Eigen::Isometry3f T_wl = f.gt_pose * origin_extri;

        calibrator.AddDataFrame(f.ptcloud, f.left_img, T_wl);
    }

    Eigen::Matrix3f intri_mat;
    calibrator.GetCamIntri(intri_mat);

    for (int iter = 0; iter < 5; ++iter) {
        Eigen::Isometry3f optimized_extri;
        calibrator.GetExtri(optimized_extri);

        phc::Vector6f err;
        phc::utils::Evaluator eval(origin_extri, init_extri, optimized_extri, intri_mat);
        eval.GetOriginResultErr(err);

        gs_bag->AddData(iter, err);

        calibrator.Calibrate();
    }

    m2d::DataManager::Add(gs_bag);
    m2d::DataManager::DumpAll();

    return EXIT_SUCCESS;
}

void SingleExtrinsicCompute(Eigen::Isometry3f init_extri, const phc::OnlineCaliConf& conf, const kitti::KittiLoader& kitti_dataset, const std::string& name) {
    phc::utils::FrameNumWithFinalErrDataBag::Ptr fnum_err_bag(new phc::utils::FrameNumWithFinalErrDataBag(name));
    m2d::DataManager::Add(fnum_err_bag);

    kitti::Intrinsics kitti_intri = kitti_dataset.GetLeftCamIntrinsics();

    // Calibrator instance
    phc::PhotoCaliOnline calibrator(conf);
    calibrator.SetCamIntrinsics(kitti_intri.fx, kitti_intri.fy, kitti_intri.cx, kitti_intri.cy);
    calibrator.SetInitExtri(init_extri);

    Eigen::Isometry3f origin_extri = kitti_dataset.GetExtrinsics();

    // std::queue<int> frame_nums({ 0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000 });
    std::queue<int> frame_nums({ 0, 200, 400, 600, 800, 1000 });

    int start_f_idx = phc::utils::Config::Get<int>("start_frame_idx");
    CHECK_LT(start_f_idx + frame_nums.back(), kitti_dataset.Size());
    for (int i = start_f_idx; i < kitti_dataset.Size(); ++i) {
        if (frame_nums.empty()) {
            break;
        }
        kitti::Frame f = kitti_dataset[i];
        Eigen::Isometry3f T_wl = f.gt_pose * origin_extri;

        int f_num = i - start_f_idx;
        if (f_num == frame_nums.front()) {
            LOG(INFO) << "Frame number: " << f_num;

            if (f_num != 0) { // Avoid calling calibrate with no data
                calibrator.Calibrate();
            }

            Eigen::Isometry3f optimized_extri;
            calibrator.GetExtri(optimized_extri);

            Eigen::Matrix3f intri_mat;
            calibrator.GetCamIntri(intri_mat);
            phc::utils::Evaluator eval(origin_extri, init_extri, optimized_extri, intri_mat);

            phc::Vector6f err;
            eval.GetOriginResultErr(err);

            fnum_err_bag->AddData(frame_nums.front(), err);

            frame_nums.pop();

            calibrator.SetInitExtri(init_extri);
        }

        calibrator.AddDataFrame(f.ptcloud, f.left_img, T_wl);
    }
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