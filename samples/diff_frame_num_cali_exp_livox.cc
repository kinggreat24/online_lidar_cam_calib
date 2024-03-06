#include <queue>

#include <glog/logging.h>

#include <pcl/io/pcd_io.h>

#include "livox_loader/LivoxLoader.hpp"

#include "PhotoCaliOnline.hpp"
#include "Config.hpp"
#include "StopWatch.hpp"
#include "DataRecorder.hpp"
#include "Evaluator.hpp"
#include "DataBag.hpp"
#include "Rand.hpp"

void LoadConfig(phc::OnlineCaliConf& conf);
void SingleExtrinsicCompute(Eigen::Isometry3f init_extri, const phc::OnlineCaliConf& conf, const lvx::LivoxLoader& livox_loader, const std::string& name);


int main(int argc, char const* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: ./livox_loader <path-to-config-file>" << std::endl;
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

    // Set dataset info
    Eigen::Isometry3f origin_extri = livox_loader.GetExtrinsic();

    // Set initial extrinsic
    Eigen::Isometry3f init_extri;

    m2d::DataManager::SetRoot(phc::utils::Config::Get<std::string>("data_dump_path"));

    std::vector<Eigen::Isometry3f> extri_errors = phc::utils::Rand::UniformIsometry3f(2.0f, 0.1f, 20);
    // std::vector<Eigen::Isometry3f> single_err(5);
    // std::fill(single_err.begin(), single_err.end(), extri_errors.front());
    for (int i = 0; i < extri_errors.size(); ++i) {
        init_extri = extri_errors[i] * origin_extri;

        std::string name("fnum_err_" + std::to_string(i));
        SingleExtrinsicCompute(init_extri, conf, livox_loader, name);
    }

    m2d::DataManager::DumpAll();

    return EXIT_SUCCESS;
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

void SingleExtrinsicCompute(Eigen::Isometry3f init_extri, const phc::OnlineCaliConf& conf, const lvx::LivoxLoader& livox_loader, const std::string& name) {
    phc::utils::FrameNumWithFinalErrDataBag::Ptr fnum_err_bag(new phc::utils::FrameNumWithFinalErrDataBag(name));
    m2d::DataManager::Add(fnum_err_bag);

    lvx::Intrinsic lvx_intri = livox_loader.GetIntrinsic();

    // Calibrator instance
    phc::PhotoCaliOnline calibrator(conf);
    calibrator.SetCamIntrinsics(lvx_intri.fx, lvx_intri.fy, lvx_intri.cx, lvx_intri.cy);
    calibrator.SetInitExtri(init_extri);

    Eigen::Isometry3f origin_extri = livox_loader.GetExtrinsic();

    std::queue<int> frame_nums({ 0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000 });
    // std::queue<int> frame_nums({ 0, 400, 800});

    int start_f_idx = phc::utils::Config::Get<int>("start_frame_idx");
    CHECK_LT(start_f_idx + frame_nums.back(), livox_loader.Size());
    for (int i = start_f_idx; i < livox_loader.Size(); ++i) {
        if (frame_nums.empty()) {
            break;
        }
        lvx::Frame f = livox_loader[i];
        // Eigen::Isometry3f T_wl = f.gt_pose * origin_extri;
        Eigen::Isometry3f T_wl = f.T_wl;

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

        calibrator.AddDataFrame(f.ptcloud, f.img, T_wl);
    }
}