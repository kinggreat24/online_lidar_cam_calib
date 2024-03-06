#include <glog/logging.h>

#include "Config.hpp"

using phc::utils::Config;

cv::FileStorage Config::file_;

bool Config::SetFile(const std::string &file_path){
    file_ = cv::FileStorage(file_path.c_str(), cv::FileStorage::READ);

    if(file_.isOpened() == false){
        LOG(FATAL) << "Fail to load file " << file_path;
        file_.release();
        return false;
    }

    return true;
}