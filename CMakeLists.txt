cmake_minimum_required(VERSION 3.0.2)
project(social_vrnn)

find_package(
   catkin REQUIRED
   # Specify dependencies
   COMPONENTS
   rospy
   message_generation
   visualization_msgs
   geometry_msgs
   nav_msgs
)

add_message_files(
   FILES
   lmpcc_obstacle.msg
   lmpcc_obstacle_array.msg
   svrnn_path.msg
   svrnn_path_array.msg
)

generate_messages(
   DEPENDENCIES
   geometry_msgs
   nav_msgs
)

catkin_package(
   # Specify package dependencies
   CATKIN_DEPENDS
   rospy
   message_runtime
   visualization_msgs
   geometry_msgs
   nav_msgs
)

include_directories(${catkin_INCLUDE_DIRS})