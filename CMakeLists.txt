cmake_minimum_required(VERSION 3.0.0)
project(depth_camera_extrinsic_calibration VERSION 0.1.0)

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fPIC")

include(CTest)
enable_testing()

find_package(Eigen3 REQUIRED)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})

find_package(OpenMP REQUIRED)

file(GLOB_RECURSE cpp_srcs src/*.cpp)
file(GLOB_RECURSE cpp_apps app/*.cpp)

add_executable(depth_camera_extrinsic_calibration ${cpp_srcs} ${cpp_apps})
target_link_libraries(depth_camera_extrinsic_calibration ${OpenCV_LIBS} OpenMP::OpenMP_CXX ${PCL_LIBRARIES} Eigen3::Eigen)
