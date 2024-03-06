#pragma once

#include <vector>
#include <memory>

#include <opencv2/core/mat.hpp>

#include <glog/logging.h>

#include <ceres/ceres.h>
#include <ceres/jet.h>
#include <ceres/cubic_interpolation.h>

#include "Types.hpp"

namespace phc {
	class BAImgInfo {
	public:
		BAImgInfo() = default;

		inline void SetCameraIntri(const CamIntri& intri) { intri_ = intri; }
		inline void SetRotWeightThresh(const float& t) { rot_weight_thresh_ = t; }
		inline void SetTransWeightThresh(const float& t) { trans_weight_thresh_ = t; }

		inline CamIntri GetCameraIntri() { return intri_; }
		inline float GetRotWeightThresh() const { return rot_weight_thresh_; }
		inline float GetTransWeightThresh() const { return trans_weight_thresh_; }

		void Clear();

		void AddImgWithPose(const cv::Mat& img, const Eigen::Isometry3f& T_lw);

		const ceres::BiCubicInterpolator<ceres::Grid2D<double>>& GetInterpolator(size_t idx) const;
		Eigen::Isometry3f GetTlw(size_t idx) const;
	private:
		CamIntri intri_{0.0f, 0.0f, 0.0f, 0.0f};

		float rot_weight_thresh_ = 0.0f;
		float trans_weight_thresh_ = 0.0f;

		std::vector<Eigen::Isometry3f> T_lw_;
		std::vector<cv::Mat> imgs_;
		std::vector<std::shared_ptr<ceres::Grid2D<double>>> img_grid_;
		std::vector<std::shared_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double>>>> interpolators_;
	};

	class PhotometricError {
	public:
		PhotometricError(const Eigen::Vector3f& pt_world, const std::vector<size_t>& vis_f_idx)
			: pt_world_(pt_world), visibile_f_idx_(vis_f_idx) {}

		template <typename T>
		bool operator()(const T* const ex_q, const T* const ex_t, T* residuals) const {
			Eigen::Quaternion<T> _ex_q(ex_q[3], ex_q[0], ex_q[1], ex_q[2]);
			Eigen::Matrix<T, 3, 1> _ex_t(ex_t[0], ex_t[1], ex_t[2]);

			std::vector<T> intensities;
			intensities.reserve(visibile_f_idx_.size());
			T intensity_sum(0.0);
			for (const auto &idx : visibile_f_idx_) {
				Eigen::Matrix<T, 3, 1> pt_lidar = (T_lw_[idx].cast<T>() * pt_world_.cast<T>());

				// Transform to camera frame
				Eigen::Matrix<T, 3, 1> pt_cam = _ex_q * pt_lidar;
				pt_cam += _ex_t;
				// pt_cam[0] += ex_t[0];
				// pt_cam[1] += ex_t[1];
				// pt_cam[2] += ex_t[2];

				// Projection
				T projected_x = static_cast<T>(fx_) * pt_cam[0] / pt_cam[2] + static_cast<T>(cx_);
				T projected_y = static_cast<T>(fy_) * pt_cam[1] / pt_cam[2] + static_cast<T>(cy_);

				// Evaluation with interpolation
				T pixel_val;
				interpolators_[idx]->Evaluate(projected_y, projected_x, &pixel_val);
				intensities.push_back(pixel_val);
				intensity_sum += pixel_val;
			}

			T mean(intensity_sum / static_cast<T>(intensities.size()));
			T numerator(0.0);
			for (int i = 0; i < intensities.size(); ++i) {
				T diff = intensities[i] - mean;
				numerator += diff * diff;
			}

			residuals[0] = numerator / static_cast<T>(intensities.size());

			return true;
		}

		static ceres::CostFunction* Create(const Eigen::Vector3f& pt_world, const std::vector<size_t>& vis_f_idx) {
			return (new ceres::AutoDiffCostFunction<PhotometricError, 1, 4, 3>(
				new PhotometricError(pt_world, vis_f_idx)));
		}

		static void SetSharedCamIntri(float fx, float fy, float cx, float cy) {
			fx_ = fx;
			fy_ = fy;
			cx_ = cx;
			cy_ = cy;

			LOG(INFO) << "[BASolver] Intrinsics set: " << " fx: " << fx_ << " fy: " << fy_
				<< " cx: " << cx_ << " cy: " << cy_;
		}

		static void AddImgWithPose(const cv::Mat &img, const Eigen::Isometry3f &T_lw) {
			using namespace ceres;

			// Push pose
			T_lw_.push_back(T_lw);

			// Push img
			cv::Mat img_f = img.clone();
			img_f.convertTo(img_f, CV_64FC1);
			imgs_.push_back(img_f);

			// Prepare interpolator and 2d grid
			double* ptr = imgs_.back().ptr<double>(0);
			std::shared_ptr<Grid2D<double>> grid_ptr(new Grid2D<double>(ptr, 0, img_f.rows, 0, img_f.cols));
			img_grid_.push_back(grid_ptr);

			std::shared_ptr<BiCubicInterpolator<Grid2D<double>>> intp_ptr(new BiCubicInterpolator<Grid2D<double>>(*grid_ptr));
			interpolators_.push_back(intp_ptr);
		}

		static void Clear() {
			fx_ = 0.0f;
			fy_ = 0.0f;
			cx_ = 0.0f;
			cy_ = 0.0f;

			T_lw_.clear();
			imgs_.clear();
			img_grid_.clear();
			interpolators_.clear();
		}
		

	private:
		static float fx_;
		static float fy_;
		static float cx_;
		static float cy_;

		static std::vector<Eigen::Isometry3f> T_lw_;
		static std::vector<cv::Mat> imgs_;
		static std::vector<std::shared_ptr<ceres::Grid2D<double>>> img_grid_;
		static std::vector<std::shared_ptr<ceres::BiCubicInterpolator<ceres::Grid2D<double>>>> interpolators_;

		Eigen::Vector3f pt_world_;
		std::vector<size_t> visibile_f_idx_;
	};

	class PhotometricErrorRotOnly {
	public:
		PhotometricErrorRotOnly(const Eigen::Vector3f& pt_world, const std::vector<size_t>& vis_f_idx, const Eigen::Vector3f &fixed_trans)
			: pt_world_(pt_world), visibile_f_idx_(vis_f_idx), ex_t_(fixed_trans){}

		template <typename T>
		bool operator()(const T* const ex_q, T* residuals) const {
			Eigen::Quaternion<T> _ex_q(ex_q[3], ex_q[0], ex_q[1], ex_q[2]);
			Eigen::Matrix<T, 3, 1> _ex_t = ex_t_.cast<T>();

			T thresh(img_info_.GetRotWeightThresh());

			Eigen::Matrix<T, Eigen::Dynamic, 1> intensities(visibile_f_idx_.size());
			Eigen::Matrix<T, Eigen::Dynamic, 1> weights(visibile_f_idx_.size());
			int intensity_idx = 0;
			T intensity_sum(0.0);
			for (const auto& idx : visibile_f_idx_) {
				Eigen::Matrix<T, 3, 1> pt_lidar = (img_info_.GetTlw(idx).cast<T>() * pt_world_.cast<T>());

				// Transform to camera frame
				Eigen::Matrix<T, 3, 1> pt_cam = _ex_q * pt_lidar;
				pt_cam += _ex_t;

				// Euclidean distance to camera optical center
				T dist_cam = ceres::sqrt(pt_cam(0) * pt_cam(0) + pt_cam(1) * pt_cam(1) + pt_cam(2) * pt_cam(2));
				weights(intensity_idx) = static_cast<T>(dist_cam < thresh ? dist_cam : thresh);

				T fx(img_info_.GetCameraIntri().fx);
				T fy(img_info_.GetCameraIntri().fy);
				T cx(img_info_.GetCameraIntri().cx);
				T cy(img_info_.GetCameraIntri().cy);

				// Projection
				T projected_x = fx * pt_cam[0] / pt_cam[2] + cx;
				T projected_y = fy * pt_cam[1] / pt_cam[2] + cy;

				// Evaluation with interpolation
				T pixel_val;
				img_info_.GetInterpolator(idx).Evaluate(projected_y, projected_x, &pixel_val);
				intensities(intensity_idx) = pixel_val;
				intensity_sum += pixel_val;

				intensity_idx += 1;
			}

			T num_visible_imgs(static_cast<T>(visibile_f_idx_.size()));

			T mean(intensity_sum / num_visible_imgs);
			T numerator(0.0);
			for (int i = 0; i < visibile_f_idx_.size(); ++i) {
				T diff = intensities(i) - mean;
				numerator += diff * diff * weights(i);
			}

			residuals[0] = numerator / static_cast<T>(num_visible_imgs);

			return true;
		}

		static void SetBAImgInfo(const BAImgInfo& info) { img_info_ = info; }
		static BAImgInfo& MutableBAImgInfo() { return img_info_; }
		static void ClearBAImgInfo() { img_info_.Clear(); }

		static ceres::CostFunction* Create(const Eigen::Vector3f& pt_world, const std::vector<size_t>& vis_f_idx, const Eigen::Vector3f& fixed_trans) {
			return (new ceres::AutoDiffCostFunction<PhotometricErrorRotOnly, 1, 4>(
				new PhotometricErrorRotOnly(pt_world, vis_f_idx, fixed_trans)));
		}

	private:
		static BAImgInfo img_info_;

		Eigen::Vector3f ex_t_;

		Eigen::Vector3f pt_world_;
		std::vector<size_t> visibile_f_idx_;
	};

	class PhotometricErrorTransOnly{
	public:
		PhotometricErrorTransOnly(const Eigen::Vector3f& pt_world, const std::vector<size_t>& vis_f_idx, const Eigen::Quaternionf& fixed_rot)
			: pt_world_(pt_world), visibile_f_idx_(vis_f_idx), ex_q_(fixed_rot){}

		template <typename T>
		bool operator()(const T* const ex_t, T* residuals) const {
			Eigen::Quaternion<T> _ex_q = ex_q_.cast<T>();
			Eigen::Matrix<T, 3, 1> _ex_t(ex_t[0], ex_t[1], ex_t[2]);

			T thresh(img_info_.GetTransWeightThresh());

			Eigen::Matrix<T, Eigen::Dynamic, 1> intensities(visibile_f_idx_.size());
			Eigen::Matrix<T, Eigen::Dynamic, 1> weights(visibile_f_idx_.size());
			int intensity_idx = 0;
			T intensity_sum(0.0);
			for (const auto& idx : visibile_f_idx_) {
				Eigen::Matrix<T, 3, 1> pt_lidar = (img_info_.GetTlw(idx).cast<T>() * pt_world_.cast<T>());

				// Transform to camera frame
				Eigen::Matrix<T, 3, 1> pt_cam = _ex_q * pt_lidar;
				pt_cam += _ex_t;

				// Euclidean distance to camera optical center
				T dist_cam = ceres::sqrt(pt_cam(0) * pt_cam(0) + pt_cam(1) * pt_cam(1) + pt_cam(2) * pt_cam(2));
				weights(intensity_idx) = dist_cam < thresh ? T(1.0) : T(0.0);
				// weights(intensity_idx) = static_cast<T>(1.0);

				T fx(img_info_.GetCameraIntri().fx);
				T fy(img_info_.GetCameraIntri().fy);
				T cx(img_info_.GetCameraIntri().cx);
				T cy(img_info_.GetCameraIntri().cy);

				// Projection
				T projected_x = fx * pt_cam[0] / pt_cam[2] + cx;
				T projected_y = fy * pt_cam[1] / pt_cam[2] + cy;

				// Evaluation with interpolation
				T pixel_val;
				img_info_.GetInterpolator(idx).Evaluate(projected_y, projected_x, &pixel_val);
				intensities(intensity_idx) = pixel_val;
				intensity_sum += pixel_val;

				intensity_idx += 1;
			}

			int num_nonzero = 0;
			for (int i = 0; i < visibile_f_idx_.size(); ++i) {
				if (weights(i) > 0.0) {
					num_nonzero += 1;
				}
			}

			if (num_nonzero <= 1) {
				residuals[0] = T(0.0);
				return true;
			}

			T num_visible_imgs(static_cast<T>(visibile_f_idx_.size()));

			T mean(intensity_sum / num_visible_imgs);
			T numerator(0.0);
			for (int i = 0; i < visibile_f_idx_.size(); ++i) {
				T diff = intensities(i) - mean;
				numerator += diff * diff * weights(i);
			}

			residuals[0] = numerator / static_cast<T>(num_visible_imgs);

			return true;
		}

		static void SetBAImgInfo(const BAImgInfo& info) { img_info_ = info; }
		static BAImgInfo& MutableBAImgInfo() { return img_info_; }
		static void ClearBAImgInfo() { img_info_.Clear(); }

		static ceres::CostFunction* Create(const Eigen::Vector3f& pt_world, const std::vector<size_t>& vis_f_idx, const Eigen::Quaternionf& fixed_rot) {
			return (new ceres::AutoDiffCostFunction<PhotometricErrorTransOnly, 1, 3>(
				new PhotometricErrorTransOnly(pt_world, vis_f_idx, fixed_rot)));
		}
	private:
		static BAImgInfo img_info_;

		Eigen::Quaternionf ex_q_;

		Eigen::Vector3f pt_world_;
		std::vector<size_t> visibile_f_idx_;
	};

	class PhotoErrRotWeighted {
	public:
		PhotoErrRotWeighted(const Eigen::Vector3f& pt_world, const std::vector<size_t>& vis_f_idx,
			const std::vector<float> &weights, const Eigen::Vector3f& fixed_trans) :
			pt_world_(pt_world), vis_f_idx_(vis_f_idx), ex_t_(fixed_trans), weights_(weights){
			CHECK_EQ(vis_f_idx.size(), weights.size()) << "[PhotoErrRotWeighted] Frame number != weight number";
		}

		template <typename T>
		bool operator()(const T* const ex_q, T* residuals) const {
			Eigen::Quaternion<T> _ex_q(ex_q[3], ex_q[0], ex_q[1], ex_q[2]);
			Eigen::Matrix<T, 3, 1> _ex_t = ex_t_.cast<T>();

			Eigen::Matrix<T, Eigen::Dynamic, 1> intensities(vis_f_idx_.size());
			Eigen::Matrix<T, Eigen::Dynamic, 1> weights(vis_f_idx_.size());
			int intensity_idx = 0;
			T intensity_sum(0.0);
			for (const auto& idx : vis_f_idx_) {
				Eigen::Matrix<T, 3, 1> pt_lidar = (img_info_.GetTlw(idx).cast<T>() * pt_world_.cast<T>());

				// Transform to camera frame
				Eigen::Matrix<T, 3, 1> pt_cam = _ex_q * pt_lidar;
				pt_cam += _ex_t;

				// Assign weights
				weights(intensity_idx) = static_cast<T>(weights_[intensity_idx]);

				// Get intrinsics
				CamIntri intri = img_info_.GetCameraIntri();
				T fx(intri.fx);
				T fy(intri.fy);
				T cx(intri.cx);
				T cy(intri.cy);

				// Projection
				T projected_x = fx * pt_cam[0] / pt_cam[2] + cx;
				T projected_y = fy * pt_cam[1] / pt_cam[2] + cy;

				// Evaluation with interpolation
				T pixel_val;
				img_info_.GetInterpolator(idx).Evaluate(projected_y, projected_x, &pixel_val);

				// 3x3 window
				// 0, 1, 2
				// 3, 4, 5
				// 6, 7, 8
				// std::array<T, 9> window;
				// img_info_.GetInterpolator(idx).Evaluate(projected_y - T(1), projected_x - T(1), &window[0]);
				// img_info_.GetInterpolator(idx).Evaluate(projected_y - T(1), projected_x, &window[1]);
				// img_info_.GetInterpolator(idx).Evaluate(projected_y - T(1), projected_x + T(1), &window[2]);
				// img_info_.GetInterpolator(idx).Evaluate(projected_y, projected_x - T(1), &window[3]);
				// img_info_.GetInterpolator(idx).Evaluate(projected_y, projected_x, &window[4]);
				// img_info_.GetInterpolator(idx).Evaluate(projected_y, projected_x + T(1), &window[5]);
				// img_info_.GetInterpolator(idx).Evaluate(projected_y + T(1), projected_x - T(1), &window[6]);
				// img_info_.GetInterpolator(idx).Evaluate(projected_y + T(1), projected_x, &window[7]);
				// img_info_.GetInterpolator(idx).Evaluate(projected_y + T(1), projected_x + T(1), &window[8]);
				//T sum(0.0);
				//for (const T &v : window) {
				//	sum += v;
				//}
				//T pixel_val = window[4] - sum / 9.0;

				intensities(intensity_idx) = pixel_val;
				intensity_sum += pixel_val;

				intensity_idx += 1;
			}

			T num_visible_imgs(static_cast<T>(vis_f_idx_.size()));

			T mean(intensity_sum / num_visible_imgs);
			T numerator(0.0);
			for (int i = 0; i < vis_f_idx_.size(); ++i) {
				T diff = intensities(i) - mean;
				// numerator += diff * diff * weights(i);
				numerator += diff * diff;
			}

			residuals[0] = numerator / static_cast<T>(num_visible_imgs);

			return true;
		}

		static ceres::CostFunction* Create(const Eigen::Vector3f& pt_world, const std::vector<size_t>& vis_f_idx,  
			const std::vector<float>& weights, const Eigen::Vector3f& fixed_trans) {
			
			return (new ceres::AutoDiffCostFunction<PhotoErrRotWeighted, 1, 4>(
				new PhotoErrRotWeighted(pt_world, vis_f_idx, weights, fixed_trans)));
		}

		static void SetBAImgInfo(const BAImgInfo& info) { img_info_ = info; }
		static BAImgInfo& MutableBAImgInfo() { return img_info_; }
		static void ClearBAImgInfo() { img_info_.Clear(); }
	private:
		static BAImgInfo img_info_;

		Eigen::Vector3f pt_world_;
		std::vector<size_t> vis_f_idx_;
		Eigen::Vector3f ex_t_;
		std::vector<float> weights_;
	};

	class PhotoErrTransWeighted {
	public:
		PhotoErrTransWeighted(const Eigen::Vector3f& pt_world, const std::vector<size_t>& vis_f_idx,
			const std::vector<float>& weights, const Eigen::Quaternionf& fixed_rot) :
			pt_world_(pt_world), vis_f_idx_(vis_f_idx), ex_q_(fixed_rot), weights_(weights) {
			CHECK_EQ(vis_f_idx.size(), weights.size()) << "[PhotoErrRotWeighted] Frame number != weight number";
		}

		template <typename T>
		bool operator()(const T* const ex_t, T* residuals) const {
			Eigen::Quaternion<T> _ex_q = ex_q_.cast<T>();
			Eigen::Matrix<T, 3, 1> _ex_t(ex_t[0], ex_t[1], ex_t[2]);

			Eigen::Matrix<T, Eigen::Dynamic, 1> intensities(vis_f_idx_.size());
			Eigen::Matrix<T, Eigen::Dynamic, 1> weights(vis_f_idx_.size());
			int intensity_idx = 0;
			T intensity_sum(0.0);
			for (const auto& idx : vis_f_idx_) {
				Eigen::Matrix<T, 3, 1> pt_lidar = (img_info_.GetTlw(idx).cast<T>() * pt_world_.cast<T>());

				// Transform to camera frame
				Eigen::Matrix<T, 3, 1> pt_cam = _ex_q * pt_lidar;
				pt_cam += _ex_t;

				// Assign weights
				weights(intensity_idx) = static_cast<T>(weights_[intensity_idx]);

				// Get intrinsics
				CamIntri intri = img_info_.GetCameraIntri();
				T fx(intri.fx);
				T fy(intri.fy);
				T cx(intri.cx);
				T cy(intri.cy);

				// Projection
				T projected_x = fx * pt_cam[0] / pt_cam[2] + cx;
				T projected_y = fy * pt_cam[1] / pt_cam[2] + cy;

				// Evaluation with interpolation
				T pixel_val;
				img_info_.GetInterpolator(idx).Evaluate(projected_y, projected_x, &pixel_val);

				// 3x3 window
				// 0, 1, 2
				// 3, 4, 5
				// 6, 7, 8
				// std::array<T, 9> window;
				// img_info_.GetInterpolator(idx).Evaluate(projected_y - T(1), projected_x - T(1), &window[0]);
				// img_info_.GetInterpolator(idx).Evaluate(projected_y - T(1), projected_x, &window[1]);
				// img_info_.GetInterpolator(idx).Evaluate(projected_y - T(1), projected_x + T(1), &window[2]);
				// img_info_.GetInterpolator(idx).Evaluate(projected_y, projected_x - T(1), &window[3]);
				// img_info_.GetInterpolator(idx).Evaluate(projected_y, projected_x, &window[4]);
				// img_info_.GetInterpolator(idx).Evaluate(projected_y, projected_x + T(1), &window[5]);
				// img_info_.GetInterpolator(idx).Evaluate(projected_y + T(1), projected_x - T(1), &window[6]);
				// img_info_.GetInterpolator(idx).Evaluate(projected_y + T(1), projected_x, &window[7]);
				// img_info_.GetInterpolator(idx).Evaluate(projected_y + T(1), projected_x + T(1), &window[8]);

				// T sum(0.0);
				// for (const T& v : window) {
				// 	sum += v;
				// }
				// T pixel_val = window[4] - sum / 9.0;

				intensities(intensity_idx) = pixel_val;
				intensity_sum += pixel_val;

				intensity_idx += 1;
			}

			T num_visible_imgs(static_cast<T>(vis_f_idx_.size()));

			T mean(intensity_sum / num_visible_imgs);
			T numerator(0.0);
			for (int i = 0; i < vis_f_idx_.size(); ++i) {
				T diff = intensities(i) - mean;
				// numerator += diff * diff * weights(i);
				numerator += diff * diff;
			}

			residuals[0] = numerator / static_cast<T>(num_visible_imgs);

			return true;
		}

		static ceres::CostFunction* Create(const Eigen::Vector3f& pt_world, const std::vector<size_t>& vis_f_idx,
			const std::vector<float>& weights, const Eigen::Quaternionf& fixed_rot) {

			return (new ceres::AutoDiffCostFunction<PhotoErrTransWeighted, 1, 3>(
				new PhotoErrTransWeighted(pt_world, vis_f_idx, weights, fixed_rot)));
		}

		static void SetBAImgInfo(const BAImgInfo& info) { img_info_ = info; }
		static BAImgInfo& MutableBAImgInfo() { return img_info_; }
		static void ClearBAImgInfo() { img_info_.Clear(); }
	private:
		static BAImgInfo img_info_;

		Eigen::Vector3f pt_world_;
		std::vector<size_t> vis_f_idx_;
		Eigen::Quaternionf ex_q_;
		std::vector<float> weights_;
	};
}