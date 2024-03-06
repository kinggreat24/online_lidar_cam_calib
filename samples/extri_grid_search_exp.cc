#include <random>

#include <glog/logging.h>

#include <pcl/io/pcd_io.h>

#include "kitti_loader/KittiLoader.hpp"

#include "PhotoCali.hpp"
#include "Config.hpp"
#include "StopWatch.hpp"
#include "DataRecorder.hpp"
#include "Evaluator.hpp"

void GenExtriGrid(std::vector<Eigen::Isometry3f>& extris, const Eigen::Isometry3f& init_extri, float trans_step, float rot_step);
void GenExtriGridTransOnly(std::vector<Eigen::Isometry3f>& extris, const Eigen::Isometry3f& init_extri, float trans_step);
void GenExtriGridRotOnly(std::vector<Eigen::Isometry3f>& extris, const Eigen::Isometry3f& init_extri, float rot_step);
void SampleExtri(std::vector<Eigen::Isometry3f>& extris, int num);
Eigen::Quaternionf EulerToQuat(float x, float y, float z);

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

    // Set dataset info
    Eigen::Isometry3f origin_extri = kitti_dataset.GetExtrinsics();
    kitti::Intrinsics kitti_intri = kitti_dataset.GetLeftCamIntrinsics();
    
    // Load error tolerance
    float err_tlrc_x = phc::utils::Config::Get<float>("dist_thresh.tolerance_x");
    float err_tlrc_y = phc::utils::Config::Get<float>("dist_thresh.tolerance_y");

    // Generate error extrinsics around original one
    std::vector<Eigen::Isometry3f> err_extris;
    float trans_step = phc::utils::Config::Get<float>("grid_gen.trans_step");
    float rot_step = phc::utils::Config::Get<float>("grid_gen.rot_step");
    // GenExtriGrid(err_extris, origin_extri, trans_step, rot_step);
    GenExtriGridTransOnly(err_extris, origin_extri, trans_step);
    // GenExtriGridRotOnly(err_extris, origin_extri, rot_step / 180.0f * M_PI);

    // Sample err extris
    // SampleExtri(err_extris, 5);

    // Origin extrinsics cost compute
    phc::PhotoCali calibrator;
    calibrator.SetCamIntri(kitti_intri.fx, kitti_intri.fy, kitti_intri.cx, kitti_intri.cy);
    calibrator.SetExtri(origin_extri);
    calibrator.SetErrTolerance(err_tlrc_x, err_tlrc_y);

    int start_f_idx = phc::utils::Config::Get<int>("start_frame_idx");
    int f_num = phc::utils::Config::Get<int>("frame_num");
    for (int i = start_f_idx; i < start_f_idx + f_num; ++i) {
        kitti::Frame f = kitti_dataset[i];
        Eigen::Isometry3f T_wl = f.gt_pose * origin_extri;
        calibrator.AddDataFrame(f.ptcloud, f.left_img, T_wl);
    }

    // float origin_cost = calibrator.ComputeCost();
    float origin_cost = calibrator.ComputeCostWeighted().y();
    LOG(INFO) << "Cost for original extrinsics: " << origin_cost;

    // phc::utils::DataRecorder recorder;
    // std::vector<std::shared_ptr<phc::PtCloudXYZRGB_t>> vis_clouds;
    // .GetVisibleClouds(vis_clouds);
    // recorder.AddVisibleCloudForFrames(vis_clouds);

    // recorder.SetDumpPath(phc::utils::Config::Get<std::string>("recorder.dump_path"));
    // recorder.DumpVisCloud();

    // Compute cost for error extrinsics
    std::vector<float> err_costs;
    err_costs.reserve(err_extris.size());
    for (const Eigen::Isometry3f &extri : err_extris) {
        phc::PhotoCali calibrator_tmp;
        calibrator_tmp.SetErrTolerance(err_tlrc_x, err_tlrc_y);
        calibrator_tmp.SetGlobCloud(calibrator.GetGlobCloud());
        calibrator_tmp.SetCamIntri(kitti_intri.fx, kitti_intri.fy, kitti_intri.cx, kitti_intri.cy);
        calibrator_tmp.SetExtri(extri);

        for (int i = start_f_idx; i < start_f_idx + f_num; ++i) {
            kitti::Frame f = kitti_dataset[i];
            Eigen::Isometry3f T_wl = f.gt_pose * origin_extri;
            calibrator_tmp.AddDataFrame(f.ptcloud, f.left_img, T_wl);
        }

        // float cost = calibrator_tmp.ComputeCost();
        float cost = calibrator_tmp.ComputeCostWeighted().y();
        err_costs.push_back(cost);
        LOG(INFO) << "Cost for err extrinsics: " << cost;
    }

    // Compute worse percent (higher is better)
    int worse_cnt = 0;
    for (const float &err_c : err_costs) {
        if (err_c > origin_cost) {
            worse_cnt += 1;
        }
    }
    float worse_percent = static_cast<float>(worse_cnt) / static_cast<float>(err_costs.size());
    LOG(INFO) << "Worse percent: " << 100.0f * worse_percent << "%";

    err_costs.push_back(origin_cost);

    phc::utils::Evaluator::CostCompare(err_costs);

    return EXIT_SUCCESS;
}

// Generate extrinsics around init_extri (not included)
void GenExtriGrid(std::vector<Eigen::Isometry3f>& extris, const Eigen::Isometry3f& init_extri, float trans_step, float rot_step) {
    static constexpr int kDims = 6;
    static constexpr std::array<int, 3> loc{ -1, 0, 1 };

    Eigen::Vector3f init_trans(init_extri.translation());
    Eigen::Vector3f init_rot(init_extri.rotation().eulerAngles(0, 1, 2));

    for (int tran_x = 0; tran_x < loc.size(); ++tran_x) {
        for (int tran_y = 0; tran_y < loc.size(); ++tran_y) {
            for (int tran_z = 0; tran_z < loc.size(); ++tran_z) {
                for (int rot_x = 0; rot_x < loc.size(); ++rot_x) {
                    for (int rot_y = 0; rot_y < loc.size(); ++rot_y) {
                        for (int rot_z = 0; rot_z < loc.size(); ++rot_z) {
                            Eigen::Vector3f trans(static_cast<float>(init_trans.x() + loc[tran_x] * trans_step),
                                static_cast<float>(init_trans.y() + loc[tran_y] * trans_step),
                                static_cast<float>(init_trans.z() + loc[tran_z] * trans_step));
                            Eigen::Vector3f rot(static_cast<float>(init_rot.x() + loc[rot_x] * rot_step),
                                static_cast<float>(init_rot.y() + loc[rot_y] * rot_step),
                                static_cast<float>(init_rot.z() + loc[rot_z] * rot_step));

                            Eigen::Isometry3f iso(EulerToQuat(rot.x(), rot.y(), rot.z()));
                            iso.pretranslate(trans);

                            extris.push_back(iso);
                        }
                    }
                }
            }
        }
    }
}

void GenExtriGridTransOnly(std::vector<Eigen::Isometry3f>& extris, const Eigen::Isometry3f& init_extri, float trans_step) {
    static constexpr std::array<int, 3> loc{ -1, 0, 1 };

    Eigen::Vector3f init_trans(init_extri.translation());
    Eigen::Matrix3f init_rot(init_extri.rotation());

    for (int tran_x = 0; tran_x < loc.size(); ++tran_x) {
        for (int tran_y = 0; tran_y < loc.size(); ++tran_y) {
            for (int tran_z = 0; tran_z < loc.size(); ++tran_z) {
                Eigen::Vector3f trans(
                    static_cast<float>(init_trans.x() + loc[tran_x] * trans_step),
                    static_cast<float>(init_trans.y() + loc[tran_y] * trans_step),
                    static_cast<float>(init_trans.z() + loc[tran_z] * trans_step));

                Eigen::Isometry3f iso(init_rot);
                iso.pretranslate(trans);

                extris.push_back(iso);
            }
        }
    }
}

void GenExtriGridRotOnly(std::vector<Eigen::Isometry3f>& extris, const Eigen::Isometry3f& init_extri, float rot_step) {
    static constexpr std::array<int, 3> loc{ -1, 0, 1 };

    Eigen::Vector3f init_trans(init_extri.translation());
    Eigen::Vector3f init_rot(init_extri.rotation().eulerAngles(0, 1, 2));

    for (int rot_x = 0; rot_x < loc.size(); ++rot_x) {
        for (int rot_y = 0; rot_y < loc.size(); ++rot_y) {
            for (int rot_z = 0; rot_z < loc.size(); ++rot_z) {
                Eigen::Vector3f rot(
                    static_cast<float>(init_rot.x() + loc[rot_x] * rot_step),
                    static_cast<float>(init_rot.y() + loc[rot_y] * rot_step),
                    static_cast<float>(init_rot.z() + loc[rot_z] * rot_step));

                Eigen::Isometry3f iso(EulerToQuat(rot.x(), rot.y(), rot.z()));
                iso.pretranslate(init_trans);

                extris.push_back(iso);
            }
        }
    }
}

void SampleExtri(std::vector<Eigen::Isometry3f>& extris, int num) {
    if (num > extris.size()) {
        num = extris.size();
        LOG(WARNING) << "num > extris.size(), num set to extris.size()";
    }
    std::vector<Eigen::Isometry3f> samples;
    std::sample(extris.begin(), extris.end(), std::back_inserter(samples), num, std::mt19937{ std::random_device{}() });

    extris = samples;
}

Eigen::Quaternionf EulerToQuat(float x, float y, float z) {
    using Eigen::Quaternionf;
    using Eigen::AngleAxisf;
    using Eigen::Vector3f;

    Quaternionf q;
    q = AngleAxisf(x, Vector3f::UnitX())
        * AngleAxisf(y, Vector3f::UnitY())
        * AngleAxisf(z, Vector3f::UnitZ());

    return q;
}