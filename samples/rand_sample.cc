#include <glog/logging.h>

#include "Rand.hpp"
#include "Utils.hpp"

using phc::utils::Rand;
using Vector6f = Eigen::Matrix<float, 6, 1>;

int main(int argc, char const* argv[]) {
	FLAGS_logtostderr = 1;
	google::InitGoogleLogging(argv[0]);

	std::vector<Eigen::Isometry3f> samples = Rand::UniformIsometry3f(1.5f, 0.2f, 10);

	for (const auto &iso : samples) {
		
		Vector6f vec = phc::utils::IsometryToXYZRPY(iso);
		std::cout << vec << std::endl;
		std::cout << "---" << std::endl;
	}

	return EXIT_SUCCESS;
}