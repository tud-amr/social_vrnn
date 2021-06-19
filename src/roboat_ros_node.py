#!/usr/bin/env python

import os
import sys
import time
import numpy as np

import rospy
from nav_msgs.msg import OccupancyGrid, Odometry
from social_vrnn.msg import lmpcc_obstacle_array as LMPCC_Obstacle_Array


class SocialVRNN_Predictor:
   """ 
   ROS interface to the Social-VRNN roboat prediction network.
   > see https://github.com/tud-amr/social_vrnn/tree/roboat-vrnn-vessel
   """

   def __init__(self, node_name, visual_node_name, args):
      self._node_name = node_name
      self._visual_node_name = visual_node_name
      self._args = args

      # Class variables
      self._occupancy_grid = None

      # Bind node
      rospy.init_node(self._node_name)
      rospy.on_shutdown(self._shutdown_callback)
      rospy.loginfo('{} has started.'.format(self._visual_node_name))

      # Set up subscribers
      rospy.Subscriber('/roboat_cloud/obstacle/map', OccupancyGrid, self._store_occupancy_grid)
      rospy.Subscriber('/ellipse_objects_feed', LMPCC_Obstacle_Array, self._store_world_state)
      rospy.Subscriber('/roboat_localization/odometry_ekf/odometry_filtered', Odometry, self._store_roboat_state)

   def _store_occupancy_grid(self, occupancy_grid_msg):
      if self._occupancy_grid is not None: return
      # rospy.loginfo('m/cell: {}, width: {}, height: {}'.format(occupancy_grid_msg.info.resolution, occupancy_grid_msg.info.width, occupancy_grid_msg.info.height))
      # rospy.loginfo('Origin at x: {}, y: {}'.format(occupancy_grid_msg.info.origin.position.x, occupancy_grid_msg.info.origin.position.y))
      # rospy.loginfo('Data length: {}, type: {}'.format(len(occupancy_grid_msg.data), (occupancy_grid_msg.data[0])))
      
      # Transform occupancy grid data into Social-VRNN format
      grid_info = occupancy_grid_msg.info
      self._occupancy_grid = np.asarray(occupancy_grid_msg.data, dtype=float).reshape((grid_info.height, grid_info.width))
      self._occupancy_grid[self._occupancy_grid > 0.0] = 1.0
      self._occupancy_grid[self._occupancy_grid < 1.0] = 0.0
      # TODO: transform to correct resolution?

      rospy.loginfo('Saved occupancy grid.')
      
   def _store_world_state(self, world_state_msg):
      rospy.loginfo('Received world state.')

   def _store_roboat_state(self, roboat_state_msg):
      rospy.loginfo('Received roboat state.')
   
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
