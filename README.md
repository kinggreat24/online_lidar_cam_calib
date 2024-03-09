# online_lidar_cam_calib
Repo for paper Online Calibration Between Camera and LiDAR with Spatial-Temporal Photometric Consistency

Sorry guys but I am recently too busy to reformat my work into a better project. A quick release here. Hopefully I will have time to look back and give a beeter release in the future.

## Setup

```bash
git submodule update --init --recursive
./vcpkg/bootstrap-vcpkg.sh 
./vcpkg/vcpkg install
mkdir build
cd build
cmake ..
make -j8
```
