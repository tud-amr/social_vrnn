cmake_minimum_required(VERSION 3.0.2)
project(social_vrnn)

find_package(
   catkin REQUIRED
   # Specify dependencies
   COMPONENTS
   rospy
   nav_msgs
)

catkin_package(
   # Specify package dependencies
   CATKIN_DEPENDS
   rospy
   nav_msgs
)

include_directories(${catkin_INCLUDE_DIRS})