#pragma once

#include <vector>
#include <iterator>

#include <pcl/common/transforms.h>
#include <pcl/common/io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <glog/logging.h>

#include "vis_check/VisCheck.hpp"

#include "CovisGraph.hpp"
#include "CovisDataPool.hpp"
#include "Types.hpp"

namespace phc {
	namespace misc {
		void ConcatPointClouds(PtCloudXYZI_t& cloud_out, const std::vector<DataFrame>& data_window);

		void ComputeCovisInfo(CovisGraph& cg, const std::vector<DataFrame>& data_window,
			const PtCloudXYZI_t& cloud, const std::vector<PtId_t>& poi,
			const CamIntri& intri, const Eigen::Isometry3f T_cl,
			float max_view_range, float score_thresh, int edge_discard,
			float pixel_val_lower_lim, float pixel_val_upper_lim);

		float DistThreshCompute(const CamIntri &intri, int img_width, int img_height, float err_tolr_x, float err_tolr_y);

		void ComputeWeightsOnGraph(CovisGraph &cg, const CovisDataPool &dp, float rot_thresh, float trans_thresh);
	}
}