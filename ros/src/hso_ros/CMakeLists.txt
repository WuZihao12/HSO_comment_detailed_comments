cmake_minimum_required(VERSION 3.0.2)
project(hso_ros)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

set(ENABLE_BACKWARD true)
if (ENABLE_BACKWARD)
    add_definitions(-D USE_BACKWARD)
    set(LIBDW "dw")
else ()
    set(LIBDW "")
endif ()

set(BAG_LOAD true)
if (BAG_LOAD)
    add_definitions(-D USE_BAG_LOAD)
endif ()


find_package(catkin REQUIRED COMPONENTS
        rosbag
        cv_bridge
        image_transport
        roscpp
        sensor_msgs
        tf
        )
include_directories(
        ${catkin_INCLUDE_DIRS}
)

find_package(OpenCV 3.0 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)

include_directories(
        ${CMAKE_SOURCE_DIR}/src
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/../
        ${PROJECT_SOURCE_DIR}/../../
        ${PROJECT_SOURCE_DIR}/../../../
        ${PROJECT_SOURCE_DIR}/../../../include
        ${PROJECT_SOURCE_DIR}/../../../thirdparty
        ${PROJECT_SOURCE_DIR}/../../../thirdparty/Sophus
        ${PROJECT_SOURCE_DIR}/../../../thirdparty/fast/include
        ${Pangolin_INCLUDE_DIRS}
)
catkin_package()

set(hso_libs
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${Pangolin_LIBRARIES}
        ${PROJECT_SOURCE_DIR}/../../../lib/libhso.so
        ${PROJECT_SOURCE_DIR}/../../../thirdparty/g2o/lib/libg2o.so
        ${catkin_LIBRARIES}
        ${LIBDW}
        )

add_executable(Mono src/ros_mono.cpp)
target_link_libraries(Mono ${hso_libs})

