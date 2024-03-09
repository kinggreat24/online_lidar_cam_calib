add_library(photo_cali STATIC
    CovisGraph.cc 
	BASolver.cc 
	Optimizer.cc 
	Types.cc
	ImgPyramid.cc
	PhotoCaliOnline.cc
	CovisDataPool.cc 
	ManualPhotoErr.cc 
	CaliErrDetector.cc 
	Misc.cc)

#https://stackoverflow.com/questions/45688522/how-to-enable-c17-in-cmake
# target_compile_features(photo_cali PUBLIC cxx_std_17)

# print path for target TBB::tbb
get_target_property(TBB_LIB_LOCATION TBB::tbb IMPORTED_LOCATION_RELEASE)
message(STATUS "TBB_LIB_LOCATION: ${TBB_LIB_LOCATION}")

target_link_libraries(photo_cali PUBLIC Eigen3::Eigen ${OpenCV_LIBS} ${PCL_LIBRARIES} glog::glog Ceres::ceres utils external_deps TBB::tbb TBB::tbbmalloc)
target_include_directories(photo_cali PUBLIC ${CMAKE_CURRENT_LIST_DIR})
