#include "Misc.hpp"
#include "Utils.hpp"
#include "ImgPyramid.hpp"

void phc::misc::ConcatPointClouds(PtCloudXYZI_t& cloud_out, const std::vector<DataFrame>& data_window) {
	for (const DataFrame& f : data_window) {
		const Eigen::Isometry3f& T_wl = f.T_wl;
		PtCloudXYZI_t cloud_cp;
		pcl::copyPointCloud(*f.ptcloud, cloud_cp);

		// Transform to world frame and concat
		pcl::transformPointCloud(cloud_cp, cloud_cp, T_wl.matrix());
		cloud_out += cloud_cp;
	}
}

void phc::misc::ComputeCovisInfo(CovisGraph& cg, const std::vector<DataFrame>& data_window,
	const PtCloudXYZI_t& cloud, const std::vector<PtId_t>& poi,
	const CamIntri& intri, const Eigen::Isometry3f T_cl,
	float max_view_range, float score_thresh, int edge_discard,
	float pixel_val_lower_lim, float pixel_val_upper_lim) {

	using pcl::PointCloud;
	using pcl::PointXYZ;
	using pcl::PointXYZI;

	CHECK(!data_window.empty());
	CHECK(!cloud.empty());

	PointCloud<PointXYZ>::Ptr cloud_xyz(new PointCloud<PointXYZ>);
	pcl::copyPointCloud(cloud, *cloud_xyz);
	// LOG(INFO) << "Cloud size for visibility check: " << cloud.size();

	Eigen::Matrix3f intri_mat = intri.AsMat();

	visc::VisCheck vis_checker;
	vis_checker.SetInputCloud(cloud_xyz);
	vis_checker.SetMaxViewRange(max_view_range);
	vis_checker.SetVisScoreThresh(score_thresh);
	vis_checker.SetDiscardEdgeSize(edge_discard);

	visc::CamIntrinsics vis_intri{ intri.fx, intri.fy, intri.cx, intri.cy };

	// Convert to format in visc
	visc::PtIndices visc_poi;
	visc_poi.indices.reserve(poi.size());
	for (const PtId_t& id : poi) {
		visc_poi.indices.push_back(id);
	}

	for (size_t f_idx = 0; f_idx < data_window.size(); ++f_idx) {
		const DataFrame& f = data_window.at(f_idx);

		LOG(INFO) << "Computing visibility info for frame " << f_idx;

		Eigen::Isometry3f T_cw = T_cl * f.T_wl.inverse();
		vis_checker.SetCamera(vis_intri, T_cw, f.img.cols, f.img.rows);

		visc::PtIndices res;
		vis_checker.ComputeVisibilityInterestIndices(res, visc_poi);
		// vis_checker.ComputeVisibility(res);

		LOG(INFO) << "Visibility check result: " << res.indices.size() << " points";
		LOG(INFO) << "visc_poi: " << visc_poi.indices.size() << " points";

		// Jacobian
		std::vector<float> jacobian_norm;
		utils::JacobianNormCompute(cloud, res, T_cw, f.img, intri.AsMat(), jacobian_norm);
		utils::NormalizeVector(jacobian_norm);

		CHECK_EQ(jacobian_norm.size(), res.indices.size());

		// Image on pyramid level 3
		// cv::Mat lvl_img;
		// Eigen::Matrix3f lvl_intri;
		// ImgPyramid::ImgPyrDown(f.img, lvl_img, 3);
		// ImgPyramid::IntriPyrDown(intri_mat, lvl_intri, 3);

		for (int i = 0; i < res.indices.size(); ++i) {
			// Remove low jacobian points
			if (jacobian_norm[i] < 0.001) {
				continue;
			}

			pcl::index_t pt_idx = res.indices[i];

			pcl::PointXYZ pt = cloud_xyz->at(pt_idx);
			Eigen::Vector2f pixel;
			utils::ProjectPoint(Eigen::Vector3f(pt.x, pt.y, pt.z), pixel, T_cw, intri_mat);
			float pixel_val = utils::GetSubPixelValBilinear(f.img, pixel);

			// Get gradient on pyramid level
			// Eigen::Vector2f lvl_pixel;
			// utils::ProjectPoint(Eigen::Vector3f(pt.x, pt.y, pt.z), lvl_pixel, T_cw, lvl_intri);
			// Eigen::Vector2f lvl_grad = utils::SobelGradientAtSubPixel(lvl_img, lvl_pixel);
			// float grad_norm = lvl_grad.norm();
			

			// Remove under/over exposured pixels
			if (pixel_val < pixel_val_lower_lim || pixel_val > pixel_val_upper_lim) {
				continue;
			}

			// Remove pixel with small gradient
			// if (grad_norm < 50.0f) {
			// 	continue;
			// }

			cg.Insert(f_idx, pt_idx);
		}
	}
}

float phc::misc::DistThreshCompute(const CamIntri& intri, int img_width, int img_height, float err_tolr_x, float err_tolr_y) {
	CHECK_GT(img_width, 0);
	CHECK_GT(img_height, 0);

	float fov_x, fov_y;
	float res_x, res_y;
	float thresh_x, thresh_y;

	fov_x = 2 * atan2f(img_width, (2 * intri.fx));
	fov_y = 2 * atan2f(img_height, (2 * intri.fy));

	res_x = fov_x / img_width;
	res_y = fov_y / img_height;

	thresh_x = err_tolr_x / res_x;
	thresh_y = err_tolr_y / res_y;

	return (thresh_x > thresh_y ? thresh_x : thresh_y);
}

void phc::misc::ComputeWeightsOnGraph(CovisGraph& cg, const CovisDataPool& dp, float rot_thresh, float trans_thresh) {
	CHECK_GT(rot_thresh, 0.0f);
	CHECK_GT(trans_thresh, 0.0f);
	CHECK(!cg.Empty());
	CHECK(!dp.Empty());

	// LOG(INFO) << "Rotation thresh for weight computing: " << rot_thresh;
	// LOG(INFO) << "Translation thresh for weight computing: " << trans_thresh;

	std::vector<FrameId_t> f_ids;
	cg.GetAllFrames(f_ids);
	for (const auto& f_id : f_ids) {
		std::unordered_set<PtId_t> pt_ids;
		cg.FindAllVisiblePts(f_id, pt_ids);

		// Img, T_cw
		const std::pair<cv::Mat, Eigen::Isometry3f>& frame = dp.GetImgWithPose(f_id);
		for (const auto& pt_id : pt_ids) {
			const PtXYZI_t& pt = dp.GetPt(pt_id); // In world coordinate

			// Point in camera
			Eigen::Vector3f pt_cam = frame.second * Eigen::Vector3f(pt.x, pt.y, pt.z);

			float dist = pt_cam.norm();
			float rot_weight = dist < rot_thresh ? dist : rot_thresh;
			float trans_weight = dist < trans_thresh ? 1.0f : 0.0f;

			cg.SetRotWeight(f_id, pt_id, rot_weight);
			cg.SetTransWeight(f_id, pt_id, trans_weight);
		}
	}
}