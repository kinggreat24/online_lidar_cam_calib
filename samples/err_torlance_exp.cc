#include <glog/logging.h>

#include <Eigen/Core>

#include <matplot/matplot.h>

#include "kitti_loader/KittiLoader.hpp"

#include "Config.hpp"

float ThreshComputeX(float fx, int img_width, float torl);
float ThreshComputeY(float fy, int img_height, float torl);
void PlotThreshFunc(float fx, float fy, int img_width, int img_height);

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

    kitti::Intrinsics kitti_intri = kitti_dataset.GetLeftCamIntrinsics();
    kitti::Frame data_frame = kitti_dataset[0];

    int img_width = data_frame.left_img.cols;
    int img_height = data_frame.left_img.rows;

    // torlerance set to 0.005m
    // If the error caused by extrinsics is less than the torlance, 
    // at the thresh distance, the error on image plane will be less than 1 pixel
    // float torlance_x = 0.025;
    // float torlance_y = 0.025;

    // float thresh_x = ThreshComputeX(kitti_intri.fx, img_width, torlance_x);
    // float thresh_y = ThreshComputeY(kitti_intri.fy, img_height, torlance_y);

    // LOG(INFO) << "thresh_x: " << thresh_x;
    // LOG(INFO) << "thresh_y: " << thresh_y;

    PlotThreshFunc(kitti_intri.fx, kitti_intri.fy, img_width, img_height);

    return EXIT_SUCCESS;
}

float ThreshComputeX(float fx, int img_width, float torl) {
    float fov_x = 2 * atan2f(img_width, (2 * fx)); // In radians
    float res_x = fov_x / img_width;
    float thresh = torl / res_x;

    return thresh;
}

float ThreshComputeY(float fy, int img_height, float torl) {
    float fov_y = 2 * atan2f(img_height, (2 * fy)); // In radians
    float res_y = fov_y / img_height;
    float thresh = torl / res_y;

    return thresh;
}

void PlotThreshFunc(float fx, float fy, int img_width, int img_height) {
    matplot::fplot(
        [fx, img_width](double torl) {return ThreshComputeX(fx, img_width, torl); },
        std::array<double, 2>{0.0, 0.1}, "r");

    matplot::hold(matplot::on);

    matplot::fplot(
        [fy, img_height](double torl) {return ThreshComputeY(fy, img_height, torl); },
        std::array<double, 2>{0.0, 0.1}, "b");

    matplot::hold(matplot::off);
    matplot::grid(matplot::on);

    matplot::show();
}