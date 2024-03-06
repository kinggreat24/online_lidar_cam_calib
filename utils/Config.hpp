#pragma once

#include <string>
#include <memory>

#include <opencv2/core/persistence.hpp>

namespace phc{
    namespace utils {
        class Config {
        public:

            static bool SetFile(const std::string& file_path);

            template<typename T>
            static T Get(const std::string& key) {
                return static_cast<T>(file_[key]);
            }

        private:
            Config() = default;

            // static std::unique_ptr<Config> config_;
            static cv::FileStorage file_;

        };
    }

}