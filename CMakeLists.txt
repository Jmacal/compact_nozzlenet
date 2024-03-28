cmake_minimum_required(VERSION 3.0.2)
project(compact_nozzle_net_pkg)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  cv_bridge
  message_generation
  std_msgs
)

add_message_files(
  FILES
  NozzleStatus.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy sensor_msgs cv_bridge message_runtime std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

