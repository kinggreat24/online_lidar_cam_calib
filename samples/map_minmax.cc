#include <unordered_map>
#include <algorithm>
#include <iostream>

using pair_type = std::unordered_map<int, float>::value_type;

int main() {
	std::unordered_map<int, float> umap;
	umap[0] = 2.5f;
	umap[1] = 1.5f;
	umap[2] = 0.5f;
	umap[3] = 4.5f;

	for (auto pair : umap) {
		std::cout << pair.first << " " << pair.second << std::endl;
	}

	auto minmax = std::minmax_element(umap.begin(), umap.end(),
		[](pair_type p1, pair_type p2) {
			return p1.second < p2.second;
		});

	float min = minmax.first->second;
	float max = minmax.second->second;

	std::cout << min << " " << max << std::endl;
}