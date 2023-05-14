cmake_minimum_required(VERSION 3.0.2)
project(integrated_attitude_estimator)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  nav_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf
  tf2_ros
  message_generation
)

find_package(Eigen3 REQUIRED)

add_message_files(DIRECTORY msg
  FILES
  EularAngle.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
 INCLUDE_DIRS include
 LIBRARIES integrated_attitude_estimator
  CATKIN_DEPENDS nav_msgs roscpp rospy sensor_msgs std_msgs tf tf2_ros message_generation
  DEPENDS system_lib EIGEN3
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_executable(ekf_attitude_estimator src/ekf_attitude_estimator.cpp)
target_link_libraries(ekf_attitude_estimator ${catkin_LIBRARIES})