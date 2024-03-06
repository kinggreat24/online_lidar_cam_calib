#include <random>
#include <numeric>

#include <glog/logging.h>

#include "Rand.hpp"

using phc::utils::Rand;

std::unique_ptr<Rand::Impl> Rand::impl_ = nullptr;

namespace {
	const float kPi = 3.14159265359f;
	const float kDeg2Rad = kPi / 180.0f;
}

class Rand::Impl {
public:
	// Impl() : mt_(std::random_device()()) {}
	Impl() : mt_(0) {}

	int UniformDistInt(int min, int max);
	std::vector<int> UniformDistInt(int min, int max, int size);
	

	std::mt19937 mt_;
};

void Rand::Init() {
	// CHECK(!impl_) << "Rand::Init() is called multiple times !";
	impl_.reset(new Rand::Impl);
}

int Rand::UniformDistInt(const int& min, const int& max) {
	if (!impl_) {
		Init();
	}
	return impl_->UniformDistInt(min, max);
}

std::vector<int> Rand::UniformDistInt(const int& min, const int& max, const int& size) {
	if (!impl_) {
		Init();
	}
	return impl_->UniformDistInt(min, max, size);
}

std::vector<int> Rand::UniformDistIntNoRepeat(const int& min, const int& max, const int& size) {
	CHECK_LT(min, max);
	
	if (!impl_) {
		Init();
	}

	std::vector<int> sequence(max - min + 1);
	std::iota(sequence.begin(), sequence.end(), min);

	std::shuffle(sequence.begin(), sequence.end(), impl_->mt_);

	return sequence;
}

std::mt19937& Rand::GetMt19937() {
	if (!impl_) {
		Init();
	}

	return impl_->mt_;
}

int Rand::Impl::UniformDistInt(int min, int max) {
	std::uniform_int_distribution<int> udist(min, max);
	return udist(mt_);
}

std::vector<int> Rand::Impl::UniformDistInt(int min, int max, int size) {
	std::uniform_int_distribution<int> udist(min, max);
	std::vector<int> res(size);
	for (int i = 0; i < size; ++i) {
		res[i] = udist(mt_);
	}

	return res;
}

std::vector<float> Rand::UniformFloat(const float& min, const float& max, const int& size) {
	if (!impl_) {
		Init();
	}

	std::uniform_real_distribution<float> udist(min, max);
	std::mt19937& mt = impl_->mt_;

	std::vector<float> result(size);
	for (int i = 0; i < size; ++i) {
		result[i] = udist(mt);
	}

	return result;
}

std::vector<float> Rand::NormalFloat(float mean, float sigma, int size) {
	if (!impl_) {
		Init();
	}

	std::normal_distribution<float> ndist(mean, sigma);
	std::mt19937& mt = impl_->mt_;

	std::vector<float> result(size);
	for (int i = 0; i < size; ++i) {
		result[i] = ndist(mt);
	}

	return result;
}

std::vector<Eigen::Isometry3f> Rand::UniformIsometry3f(const float& rot_lim, const float& trans_lim, const int& size) {
	using Eigen::Quaternionf;
	using Eigen::AngleAxisf;
	using Eigen::Vector3f;
	using Eigen::Isometry3f;

	if (!impl_) {
		Init();
	}

	std::vector<float> trans_rands = UniformFloat(-trans_lim, trans_lim, 3 * size);
	std::vector<float> rot_rands = UniformFloat(-rot_lim, rot_lim, 3 * size);

	std::vector<float>::iterator rot_it = rot_rands.begin();
	std::vector<float>::iterator trans_it = trans_rands.begin();

	std::vector<Isometry3f> results(size);
	for (int i = 0; i < size; ++i) {
		Quaternionf q_ptb = AngleAxisf(kPi / 180.0f * (*rot_it++), Vector3f::UnitX())
			* AngleAxisf(kPi / 180.0f * (*rot_it++), Vector3f::UnitY())
			* AngleAxisf(kPi / 180.0f * (*rot_it++), Vector3f::UnitZ());

		Isometry3f iso(q_ptb);
		iso.pretranslate(Vector3f((*trans_it++), (*trans_it++), (*trans_it++)));

		results[i] = iso;
	}

	return results;
}

std::vector<Eigen::Isometry3f> Rand::NormalIsometry3f(float rot_mean, float rot_sigma, float trans_mean, float trans_sigma, int size) {
	using Eigen::Quaternionf;
	using Eigen::AngleAxisf;
	using Eigen::Vector3f;
	using Eigen::Isometry3f;

	if (!impl_) {
		Init();
	}

	std::vector<float> trans_rands = NormalFloat(trans_mean, trans_sigma, 3 * size);
	std::vector<float> rot_rands = NormalFloat(rot_mean, rot_sigma, 3 * size);

	std::vector<float>::iterator rot_it = rot_rands.begin();
	std::vector<float>::iterator trans_it = trans_rands.begin();

	std::vector<Isometry3f> results(size);
	for (int i = 0; i < size; ++i) {
		Quaternionf q_ptb = AngleAxisf(kPi / 180.0f * (*rot_it++), Vector3f::UnitX())
			* AngleAxisf(kPi / 180.0f * (*rot_it++), Vector3f::UnitY())
			* AngleAxisf(kPi / 180.0f * (*rot_it++), Vector3f::UnitZ());

		Isometry3f iso(q_ptb);
		iso.pretranslate(Vector3f((*trans_it++), (*trans_it++), (*trans_it++)));

		results[i] = iso;
	}

	return results;
}

Eigen::Isometry3f Rand::Isometry3fFixErr(float err_rot_deg, float err_trans) {

	if (!impl_) {
		Init();
	}

	float err_rot_rad = kDeg2Rad * err_rot_deg;

	Eigen::Vector3f trans_rand = Eigen::Vector3f::Random().normalized() * err_trans;
	Eigen::AngleAxisf rot_rand = Eigen::AngleAxisf(err_rot_rad, Eigen::Vector3f::Random().normalized());

	LOG(INFO) << trans_rand.norm();
	LOG(INFO) << rot_rand.angle() / kDeg2Rad;

	Eigen::Quaternionf quat(rot_rand);

	Eigen::Isometry3f iso(quat);
	iso.pretranslate(trans_rand);

	return iso;
}