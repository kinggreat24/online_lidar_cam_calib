#include <unordered_map>
#include <unordered_set>
#include <iostream>

#include "CovisGraph.hpp"
#include "CovisDataPool.hpp"

int main() {
	// Frame -> pts
	// 4 -> 4, 5, 6
	// 5 -> 4, 6
	phc::CovisGraph cg;
	cg.Insert(1, 2);
	cg.Insert(1, 3);
	cg.Insert(2, 1);
	cg.Insert(2, 2);
	cg.Insert(2, 3);

	phc::CovisGraph cg2;
	cg2.Insert(1, 2);
	cg2.Insert(1, 3);
	cg2.Insert(3, 1);
	cg2.Insert(3, 2);
	cg2.Insert(3, 3);

	cg.Merge(cg2);

	// cg.ErasePt(1);
	// cg.EraseLessObservedPt(2);

	std::vector<phc::PtId_t> pt_indices;
	cg.GetAllPts(pt_indices);

	std::vector<phc::FrameId_t> f_indices;
	cg.GetAllFrames(f_indices);

	std::cout << "All visible pts: " << std::endl;
	for (const auto &pt : pt_indices) {
		std::cout << pt << std::endl;
	}

	std::cout << "All visible frames: " << std::endl;
	for (const auto& f : f_indices) {
		std::cout << f << std::endl;
	}

	std::cout << "Map:" << std::endl;
	for (const auto& f : f_indices) {
		std::unordered_set<phc::PtId_t> pt_ids;
		cg.FindAllVisiblePts(f, pt_ids);
		for (const phc::PtId_t& pt_id : pt_ids) {
			std::cout << f << " -> " << pt_id << std::endl;
		}
	}

	return EXIT_SUCCESS;
}