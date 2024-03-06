add_library(utils STATIC
	Config.cc
    Utils.cc 
	StopWatch.cc 
	DataRecorder.cc
	Evaluator.cc 
	Rand.cc 
	DataBag.cc)

#https://stackoverflow.com/questions/45688522/how-to-enable-c17-in-cmake
target_compile_features(utils PUBLIC cxx_std_17)

target_link_libraries(utils PUBLIC Eigen3::Eigen ${OpenCV_LIBS} ${PCL_LIBRARIES} glog::glog matplot_deps external_deps)
target_include_directories(utils PUBLIC ${CMAKE_CURRENT_LIST_DIR})