cmake_minimum_required(VERSION 3.5)

project(dsinit LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall")

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
    include
    ${EIGEN3_INCLUDE_DIR}
    )

add_library(dsinit_lib
    src/image_processor.cpp
    src/imu_processor.cpp
    src/gt_loader.cpp
    src/init_solver.cpp
    src/utils/camera.cpp
    src/utils/so3.cpp
    )

add_executable(dsinit src/main.cpp)


target_link_libraries(dsinit
    dsinit_lib
    ${OpenCV_LIBS}
    )


