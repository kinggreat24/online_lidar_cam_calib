#include <glog/logging.h>

#include <pcl/io/pcd_io.h>

#include "kitti_loader/KittiLoader.hpp"

#include "Config.hpp"
#include "StopWatch.hpp"
#include "DataRecorder.hpp"
#include "Evaluator.hpp"
#include "CaliErrDetector.hpp"
#include "DataBag.hpp"

phc::Vector6f IsometryToXYZRPY(const Eigen::Isometry3f& iso);
Eigen::Isometry3f GetPertubationForExtri();
void LoadConfig(phc::ErrDetectorConf& conf);

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

    // Load configuration 
    phc::ErrDetectorConf conf;
    LoadConfig(conf);

    // Calibrator instance
    phc::CaliErrDetector detector(conf);

    // Set dataset info
    Eigen::Isometry3f origin_extri = kitti_dataset.GetExtrinsics();
    kitti::Intrinsics kitti_intri = kitti_dataset.GetLeftCamIntrinsics();
    detector.SetIntri(kitti_intri.fx, kitti_intri.fy, kitti_intri.cx, kitti_intri.cy);

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

    detector.SetExtri(init_extri);

    m2d::DataManager::SetRoot(phc::utils::Config::Get<std::string>("data_dump_path"));
    phc::utils::WorsePercentDataBag::Ptr worse_percent_bag(new phc::utils::WorsePercentDataBag("worse_percent"));

    int start_f_idx = phc::utils::Config::Get<int>("start_frame_idx");
    int f_num = phc::utils::Config::Get<int>("frame_num");
    for (int i = start_f_idx; i < start_f_idx + f_num; ++i) {
        kitti::Frame f = kitti_dataset[i];
        Eigen::Isometry3f T_wl = f.gt_pose * origin_extri;

        detector.AddDataFrame(f.ptcloud, f.left_img, T_wl);

        if (i == 501) {
            // detector.SetExtri(init_extri);
        }

        if ((i + 1) % 100 == 0) {
            float origin_cost;
            std::vector<float> grid_costs;
            detector.Detect(origin_cost, grid_costs);

            LOG(INFO) << grid_costs.front() << " " << origin_cost;

            float worse_percent = phc::CaliErrDetector::WorsePercentage(origin_cost, grid_costs);
            LOG(INFO) << "Cost computation at " << i << ", worse_percent: " << worse_percent;
            worse_percent_bag->AddData(i, worse_percent);

            detector.ClearDataPool();
        }
    }

    m2d::DataManager::Add(worse_percent_bag);
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

void LoadConfig(phc::ErrDetectorConf& conf) {
    using phc::utils::Config;

    conf.ptcloud_clip_min = Config::Get<float>("detector.ptcloud_clip.min");
    conf.ptcloud_clip_max = Config::Get<float>("detector.ptcloud_clip.max");
    conf.sample_ratio = Config::Get<float>("detector.sample_ratio");

    // Weight computation parameter
    conf.err_tolr_x = Config::Get<float>("detector.err_tolr_x");
    conf.err_tolr_y = Config::Get<float>("detector.err_tolr_y");
    conf.trans_thresh_ratio = Config::Get<float>("detector.trans_thresh_ratio");

    // Minimum observation number threshold
    conf.obs_thresh = Config::Get<int>("detector.obs_thresh");

    conf.window_size = Config::Get<int>("detector.window_size");

    conf.pyramid_lvl = Config::Get<int>("detector.pyramid_lvl");

    conf.extri_sample_num = Config::Get<int>("detector.extri_sample_num");

    // Covisibility check parameters
    conf.covis_conf.max_view_range = Config::Get<float>("detector.covis_check.max_view_range");
    conf.covis_conf.score_thresh = Config::Get<float>("detector.covis_check.score_thresh");
    conf.covis_conf.edge_discard = Config::Get<int>("detector.covis_check.edge_discard");

    conf.pixel_val_lower_lim = Config::Get<float>("detector.pixel_val_lower_lim");
    conf.pixel_val_upper_lim = Config::Get<float>("detector.pixel_val_upper_lim");
}