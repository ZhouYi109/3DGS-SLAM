cmake_minimum_required(VERSION 3.0.2)
project(gaussian_lic)

SET(CMAKE_BUILD_TYPE "Release")
# SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -Wno-sign-compare -Wno-unused -Wno-comment -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -msse4.2")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--no-as-needed")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  roslib
  std_msgs
  std_srvs
  message_generation
  cv_bridge
  image_transport
  pcl_conversions
  pcl_ros
  std_msgs
  nav_msgs
  sensor_msgs
  geometry_msgs
  tf
  tf_conversions
  eigen_conversions
)

option(USE_TENSORRT_DEPTH_COMPLETION "Build Gaussian-LIC with TensorRT depth completion" OFF)

set(OpenCV_DIR ~/Software/opencv/opencv-4.7.0/build)
find_package(OpenCV 4 REQUIRED)
set(Torch_DIR /root/Software/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

set(TENSORRT_ROOT "~/Software/TensorRT-8.6.1.6")
set(TENSORRT_INCLUDE_DIR "${TENSORRT_ROOT}/include")
set(TENSORRT_LIB_DIR "${TENSORRT_ROOT}/lib")

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include")
set(CUDA_LIB_DIR "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
set(CUDA_CUDART_LIBRARY "/usr/local/cuda/lib64/libcudart.so")
find_package(CUDA REQUIRED)
find_library(FFI_LIB ffi)

find_package(PkgConfig REQUIRED)
pkg_check_modules(YAML_CPP REQUIRED yaml-cpp>=0.5)

find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

add_service_files(
  FILES
  SemanticQuery.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS roscpp rospy roslib std_msgs std_srvs message_runtime cv_bridge image_transport pcl_conversions pcl_ros nav_msgs sensor_msgs geometry_msgs tf tf_conversions eigen_conversions
)

include_directories(
# include
  src
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${YAML_INCLUDE_DIRS}
  ${Python3_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS}
)

add_executable(gs_mapping
              ${PROJECT_SOURCE_DIR}/src/mapping.cpp
              ${PROJECT_SOURCE_DIR}/src/gaussian.cpp
              ${PROJECT_SOURCE_DIR}/src/tinyply.cpp
              ${PROJECT_SOURCE_DIR}/src/simple-knn/spatial.cu
              ${PROJECT_SOURCE_DIR}/src/simple-knn/simple_knn.cu
              ${PROJECT_SOURCE_DIR}/src/fused-ssim/ssim.cu
              ${PROJECT_SOURCE_DIR}/src/rasterizer/renderer.cpp
              ${PROJECT_SOURCE_DIR}/src/rasterizer/rasterizer.cpp
              ${PROJECT_SOURCE_DIR}/src/rasterizer/rasterize_points.cu
              ${PROJECT_SOURCE_DIR}/src/rasterizer/cuda_rasterizer/adam.cu
              ${PROJECT_SOURCE_DIR}/src/rasterizer/cuda_rasterizer/forward.cu
              ${PROJECT_SOURCE_DIR}/src/rasterizer/cuda_rasterizer/backward.cu
              ${PROJECT_SOURCE_DIR}/src/rasterizer/cuda_rasterizer/rasterizer_impl.cu)

set_target_properties(gs_mapping PROPERTIES
  CUDA_STANDARD 14
  CUDA_STANDARD_REQUIRED ON)

add_executable(semantic_query_cli
              ${PROJECT_SOURCE_DIR}/src/semantic_bundle.cpp
              ${PROJECT_SOURCE_DIR}/src/semantic_query_cli.cpp)

add_executable(semantic_highlight_cli
              ${PROJECT_SOURCE_DIR}/src/semantic_bundle.cpp
              ${PROJECT_SOURCE_DIR}/src/semantic_highlight_cli.cpp)

add_executable(semantic_query_server
              ${PROJECT_SOURCE_DIR}/src/semantic_bundle.cpp
              ${PROJECT_SOURCE_DIR}/src/semantic_query_server.cpp)

if(USE_TENSORRT_DEPTH_COMPLETION)
  target_sources(gs_mapping PRIVATE ${PROJECT_SOURCE_DIR}/src/depth_completer.cpp)
  target_compile_definitions(gs_mapping PRIVATE USE_TENSORRT_DEPTH_COMPLETION=1)
else()
  target_compile_definitions(gs_mapping PRIVATE USE_TENSORRT_DEPTH_COMPLETION=0)
endif()

target_link_libraries(gs_mapping
                      ${catkin_LIBRARIES}
                      ${OpenCV_LIBRARIES}
                      ${TORCH_LIBRARIES} 
                      ${YAML_CPP_LIBRARIES}
                      ${Python3_LIBRARIES}
                      ${FFI_LIB}
                      ${CUDA_CUDART_LIBRARY})

target_link_libraries(semantic_query_cli
                      ${TORCH_LIBRARIES}
                      ${YAML_CPP_LIBRARIES})

target_link_libraries(semantic_highlight_cli
                      ${TORCH_LIBRARIES}
                      ${YAML_CPP_LIBRARIES})

target_link_libraries(semantic_query_server
                      ${catkin_LIBRARIES}
                      ${TORCH_LIBRARIES}
                      ${YAML_CPP_LIBRARIES})

add_dependencies(semantic_query_server ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

catkin_install_python(PROGRAMS
  scripts/object_semantic_memory.py
  scripts/object_latent_encoder.py
  scripts/object_semantic_memory_node.py
  scripts/train_semantic_gaussian_prior.py
  scripts/export_semantic_gaussian_prior.py
  scripts/build_prior_manifest.py
  scripts/prepare_nuscenes_prior_shards.py
  scripts/build_r3live_teacher_shards.py
  scripts/prepare_r3live_cross_sequence_split.py
  scripts/publish_object_memory_real_smoke.py
  scripts/test_real_object_models.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

if(USE_TENSORRT_DEPTH_COMPLETION)
  target_link_libraries(gs_mapping
                        ${TENSORRT_LIB_DIR}/libnvinfer.so
                        ${TENSORRT_LIB_DIR}/libnvonnxparser.so
                        ${TENSORRT_LIB_DIR}/libnvparsers.so
                        ${TENSORRT_LIB_DIR}/libnvinfer_plugin.so)
endif()
