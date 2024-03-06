add_executable(concat_cloud concat_cloud.cc)
target_link_libraries(concat_cloud photo_cali utils external_deps)

add_executable(cost_compute cost_compute.cc)
target_link_libraries(cost_compute photo_cali utils external_deps)

add_executable(covis_graph_test covis_graph_test.cc)
target_link_libraries(covis_graph_test photo_cali)

add_executable(interp_test interp_test.cc)
target_link_libraries(interp_test utils)

add_executable(variance_test variance_test.cc)
target_link_libraries(variance_test utils)

add_executable(opencv_mat_val_test opencv_mat_val_test.cc)
target_link_libraries(opencv_mat_val_test external_deps)

add_executable(extri_err_analysis extri_err_analysis.cc)
target_link_libraries(extri_err_analysis photo_cali utils external_deps)

add_executable(extri_grid_search_exp extri_grid_search_exp.cc)
target_link_libraries(extri_grid_search_exp photo_cali utils external_deps)

add_executable(auto_cali auto_cali.cc)
target_link_libraries(auto_cali photo_cali utils external_deps)

add_executable(matplot_check matplot_check.cc)
target_link_libraries(matplot_check matplot_deps)

add_executable(map_minmax map_minmax.cc)

add_executable(imgpyramid_test imgpyramid_test.cc)
target_link_libraries(imgpyramid_test photo_cali)

add_executable(err_proj_vis err_proj_vis.cc)
target_link_libraries(err_proj_vis photo_cali)

add_executable(err_torlance_exp err_torlance_exp.cc)
target_link_libraries(err_torlance_exp photo_cali external_deps)

add_executable(auto_cali_weighted_var auto_cali_weighted_var.cc)
target_link_libraries(auto_cali_weighted_var photo_cali utils external_deps)

add_executable(rand_sample rand_sample.cc)
target_link_libraries(rand_sample utils)

add_executable(don_sample don_sample.cc)
target_link_libraries(don_sample photo_cali utils external_deps)

add_executable(vischeck_test vischeck_test.cc)
target_link_libraries(vischeck_test photo_cali utils external_deps)

add_executable(auto_cali_online auto_cali_online.cc)
target_link_libraries(auto_cali_online photo_cali utils external_deps)

add_executable(err_detection err_detection.cc)
target_link_libraries(err_detection photo_cali utils external_deps)

add_executable(diff_frame_num_cali_exp diff_frame_num_cali_exp.cc)
target_link_libraries(diff_frame_num_cali_exp photo_cali utils external_deps)

add_executable(diff_err_detection_exp diff_err_detection_exp.cc)
target_link_libraries(diff_err_detection_exp photo_cali utils external_deps)

add_executable(diff_err_cost diff_err_cost.cc)
target_link_libraries(diff_err_cost photo_cali utils external_deps)

add_executable(covis_data_check covis_data_check.cc)
target_link_libraries(covis_data_check photo_cali utils external_deps)

add_executable(grid_search_cali grid_search_cali.cc)
target_link_libraries(grid_search_cali photo_cali utils external_deps)

add_executable(comp_exp comp_exp.cc)
target_link_libraries(comp_exp photo_cali utils external_deps)

add_executable(color_ptcloud color_ptcloud.cc)
target_link_libraries(color_ptcloud photo_cali utils external_deps)

add_executable(single_cloud_dist_color single_cloud_dist_color.cc)
target_link_libraries(single_cloud_dist_color photo_cali utils external_deps)

add_executable(diff_frame_num_cali_exp_livox diff_frame_num_cali_exp_livox.cc)
target_link_libraries(diff_frame_num_cali_exp_livox photo_cali utils external_deps)

add_executable(covis_data_livox covis_data_livox.cc)
target_link_libraries(covis_data_livox photo_cali utils external_deps)

add_executable(concat_cloud_livox concat_cloud_livox.cc)
target_link_libraries(concat_cloud_livox photo_cali utils external_deps)

add_executable(comp_exp_livox comp_exp_livox.cc)
target_link_libraries(comp_exp_livox photo_cali utils external_deps )

add_executable(jacobian_selection jacobian_selection.cc)
target_link_libraries(jacobian_selection photo_cali utils external_deps)

add_executable(odom_exp odom_exp.cc)
target_link_libraries(odom_exp photo_cali utils external_deps)