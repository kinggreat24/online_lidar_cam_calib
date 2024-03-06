cmake_minimum_required(VERSION 3.5)

set(CMAKE_TOOLCHAIN_FILE vcpkg/scripts/buildsystems/vcpkg.cmake)

project(phcali)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # using GCC
  message(STATUS "Detect GNU compiler")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
  set(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()

# Eigen3
find_package(Eigen3 CONFIG REQUIRED)

# Ceres
find_package(Ceres CONFIG REQUIRED)

# glog
find_package(glog CONFIG REQUIRED)

# PCL
find_package(PCL 1.13 CONFIG REQUIRED COMPONENTS common io filters visualization segmentation)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# OpenCV
find_package(OpenCV CONFIG REQUIRED)

# Matplot++
find_package(Matplot++ CONFIG REQUIRED)
add_library(matplot_deps INTERFACE)
target_link_libraries(matplot_deps INTERFACE
    Matplot++::cimg Matplot++::matplot 
 )

find_package(TBB CONFIG REQUIRED)

add_subdirectory(external)
add_subdirectory(src)
add_subdirectory(utils)
add_subdirectory(samples)