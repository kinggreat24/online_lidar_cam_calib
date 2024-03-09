add_executable(auto_cali_online auto_cali_online.cc)
target_link_libraries(auto_cali_online photo_cali utils external_deps)

add_executable(err_detection err_detection.cc)
target_link_libraries(err_detection photo_cali utils external_deps)
