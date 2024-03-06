#pragma once

#include <memory>
#include <vector>
#include <random>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace phc {
	namespace utils {
		class Rand {
		public:
			static void Init();
			static int UniformDistInt(const int& min, const int& max);
			static std::vector<int> UniformDistInt(const int& min, const int& max, const int& size);
			static std::vector<int> UniformDistIntNoRepeat(const int& min, const int& max, const int& size);

			static std::vector<float> UniformFloat(const float& min, const float& max, const int& size);
			static std::vector<float> NormalFloat(float mean, float sigma, int size);

			// Rotation range for each axis: [-rot_lim, rot_lim] in degrees
			// Translation range for each axis: [-trans_lim, trans_lim] in meters
			static std::vector<Eigen::Isometry3f> UniformIsometry3f(const float &rot_lim, const float &trans_lim, const int &size);
			static std::vector<Eigen::Isometry3f> NormalIsometry3f(float rot_mean, float rot_sigma, float trans_mean, float trans_sigma, int size);

			static Eigen::Isometry3f Isometry3fFixErr(float err_rot_deg, float err_trans);

			static std::mt19937& GetMt19937();
		private:
			class Impl;
			static std::unique_ptr<Impl> impl_;

			Rand() = default;
		};
	}
}