#include <iostream>

#include <Eigen/Core>

#include "Utils.hpp"

using phc::utils::VarianceCompute;

int main() {
	Eigen::VectorXf vx(3);
	vx << 1.0f, -20.0f, 50.0f;

	std::cout << vx << std::endl;

	float var = VarianceCompute(vx);

	std::cout << var << std::endl;

	return EXIT_SUCCESS;
}