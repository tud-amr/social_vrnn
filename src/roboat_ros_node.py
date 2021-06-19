#!/usr/bin/env python

import os
import sys
import time

import rospy
from nav_msgs.msg import OccupancyGrid


class SocialVRNN_Predictor:
   """ 
   ROS interface to the Social-VRNN roboat prediction network.
   > see https://github.com/tud-amr/social_vrnn/tree/roboat-vrnn-vessel
   """

   def __init__(self, node_name, visual_node_name, args):
      self._node_name = node_name
      self._visual_node_name = visual_node_name
      self._args = args

      # Bind node
      rospy.init_node(self._node_name)
      rospy.on_shutdown(self._shutdown_callback)
      rospy.loginfo('{} has started.'.format(self._visual_node_name))

      # Setup subscribers
      rospy.Subscriber('/roboat_cloud/obstacle/map', OccupancyGrid, self._on_map_update)

   def _on_map_update(self, occupancy_grid_msg):
      rospy.loginfo('Received occupancy grid.')
      # rospy.loginfo(occupancy_grid_msg)
   

   def _shutdown_callback(self):
      rospy.loginfo('{} was terminated.'.format(self._visual_node_name))


if __name__ == '__main__':

   # Get CLI arguments
   args = rospy.myargv(sys.argv)

   # Start main logic
   SocialVRNN_Predictor('roboat_ros_node', 'Social-VRNN roboat node', args)

   # Keep node alive until shutdown
   rate = rospy.Rate(10)
   while not rospy.is_shutdown():
      rate.sleep()
