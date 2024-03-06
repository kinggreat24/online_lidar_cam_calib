#include "Types.hpp"

using phc::CamIntri;

Eigen::Matrix3f CamIntri::AsMat() const {
	Eigen::Matrix3f mat;
	mat <<
		fx, 0.0f, cx,
		0.0f, fy, cy,
		0.0f, 0.0f, 1.0f;

	return mat;
}

CamIntri CamIntri::FromMat(const Eigen::Matrix3f &mat) {
	return CamIntri{mat(0, 0), mat(1, 1), mat(0, 2), mat(1, 2)};
}

void CamIntri::Clear() {
	fx = 0.0f;
	fy = 0.0f;
	cx = 0.0f;
	cy = 0.0f;
}