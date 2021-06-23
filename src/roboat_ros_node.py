#!/usr/bin/env python

import sys; print("Running with {}".format(sys.version))
import os
import copy
import time
import pickle
import cv2
import math
import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation

import tensorflow as tf
from models.SocialVRNN import NetworkModel as SocialVRNN
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import rospkg
import rospy
from nav_msgs.msg import OccupancyGrid, Odometry
from social_vrnn.msg import lmpcc_obstacle_array as LMPCC_Obstacle_Array

PACKAGE_NAME = 'social_vrnn'
QUERY_AGENTS = 3


class SocialVRNN_Predictor:
   """ 
   ROS interface to the Social-VRNN roboat prediction network.
   > see https://github.com/tud-amr/social_vrnn/tree/roboat-vrnn-vessel
   """


   def __init__(self, node_name, visual_node_name):
      self._node_name = node_name
      self._visual_node_name = visual_node_name

      # Bind node
      rospy.init_node(self._node_name)
      rospy.on_shutdown(self._shutdown_callback)
      rospy.loginfo('{} has started.'.format(self._visual_node_name))

      # Set up class variables
      self._get_submap = None # function to request submap for some position
      self._roboat_pos = np.zeros((2, ), dtype=float)
      self._roboat_vel = np.zeros((2, ), dtype=float)
      self._agents_pos, self._agents_vel = {}, {}

      # Load the model
      self._set_model_args('SocialVRNN', '500')
      # self._model, self.tf_session = self._load_model(SocialVRNN, self._model_args)

      # Set up subscribers
      rospy.Subscriber('/roboat_cloud/obstacle/map', OccupancyGrid, self._store_occupancy_grid)
      rospy.Subscriber('/ellipse_objects_feed', LMPCC_Obstacle_Array, self._store_world_state)
      rospy.Subscriber('/roboat_localization/odometry_ekf/odometry_filtered', Odometry, self._store_roboat_state)


   def _set_model_args(self, model_name, train_run):
      trained_dir = os.path.join(rospkg.RosPack().get_path(PACKAGE_NAME), 'trained_models')
      model_dir = os.path.join(trained_dir, model_name, train_run)
      convnet_dir = os.path.join(trained_dir, 'autoencoder_with_ped')
      with open(os.path.join(model_dir, 'model_parameters.pkl'), 'rb') as f:
         self._model_args = pickle.load(f)["args"]
      self._model_args.model_path = model_dir
      self._model_args.pretrained_convnet_path = convnet_dir
      self._model_args.batch_size = QUERY_AGENTS
      self._model_args.truncated_backprop_length = 1
      self._model_args.keep_prob = 1.0


   def _load_model(self, model_class, model_args):
      model = model_class(model_args)
      tf_session = tf.Session()
      model.warmstart_model(model_args, tf_session)
      try: model.warmstart_convnet(model_args, tf_session)
      except: rospy.logwarn('Could not warm-start ConvNet')
      rospy.loginfo('Model loaded!')
      return model, tf_session


   def _store_occupancy_grid(self, occupancy_grid_msg):
      # Makes 1 pixel equal to 1 submap pixel
      scale_factor = occupancy_grid_msg.info.resolution / self._model_args.submap_resolution

      # Transform occupancy grid data into Social-VRNN format
      grid_info = occupancy_grid_msg.info
      occupancy_grid = np.asarray(occupancy_grid_msg.data, dtype=float).reshape((grid_info.height, grid_info.width))
      occupancy_grid[occupancy_grid > 0.0] = 1.0
      occupancy_grid[occupancy_grid < 1.0] = 0.0
      occupancy_grid = cv2.flip(occupancy_grid, 0)
      occupancy_grid = cv2.resize(
         occupancy_grid,
         (int(scale_factor * grid_info.width), int(scale_factor * grid_info.height)),
         fx=0, fy=0,
         interpolation=cv2.INTER_NEAREST
      )

      # Create custom function for requesting submap
      def get_submap(position, orientation):
         """
         origin_offset: (2,) numpy matrix
            The offset of the position in meters from the origin
            Index 0 is x, index 1 is y
         """
         # Translate to pixel coordinates
         origin_offset = position.copy().astype(float)
         origin_offset /= self._model_args.submap_resolution
         origin_offset[0] = origin_offset[0] + occupancy_grid.shape[1]/2
         origin_offset[1] = occupancy_grid.shape[0]/2 - origin_offset[1]
         origin_offset = origin_offset.astype(int)
         
         # Do bounds-check
         if (origin_offset[0] < self._model_args.submap_width/2 or
             origin_offset[0] > occupancy_grid.shape[1] - self._model_args.submap_width/2 or
             origin_offset[1] < self._model_args.submap_height/2 or
             origin_offset[1] > occupancy_grid.shape[0] - self._model_args.submap_height/2):
            rospy.logerr('Out-of-bounds submap requested!')
            return np.zeros((self._model_args.submap_height, self._model_args.submap_width), dtype=float)
         
         # Rotate to match orientation and return submap
         pad_x = [occupancy_grid.shape[1] - origin_offset[0], origin_offset[0]]
         pad_y = [occupancy_grid.shape[0] - origin_offset[1], origin_offset[1]]
         padded_og = np.pad(occupancy_grid, [pad_y, pad_x], 'constant', constant_values=1.0)
         rot_og = cv2.warpAffine(
            padded_og,
            cv2.getRotationMatrix2D(
               (padded_og.shape[1]/2, padded_og.shape[0]/2),
               -np.rad2deg(math.atan2(orientation[1], orientation[0])) + 90,
               1.0
            ),
            (padded_og.shape[1], padded_og.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=1.0
         )
         crop_og = rot_og[pad_y[0] : -pad_y[1], pad_x[0] : -pad_x[1]]
         return crop_og[
            int(origin_offset[1]-self._model_args.submap_height/2) : int(origin_offset[1]+self._model_args.submap_height/2),
            int(origin_offset[0]-self._model_args.submap_width/2) : int(origin_offset[0]+self._model_args.submap_width/2)
         ]

      self._get_submap = get_submap
      rospy.loginfo('Saved occupancy grid.')


   def _store_roboat_state(self, roboat_state_msg):
      curr_time = time.time()
      prev_x = self._roboat_pos[0]
      prev_y = self._roboat_pos[1]
      curr_x = roboat_state_msg.pose.pose.position.x
      curr_y = roboat_state_msg.pose.pose.position.y

      self._roboat_pos[0] = curr_x
      self._roboat_pos[1] = curr_y
      if hasattr(self, '_roboat_last_update'):
         self._roboat_vel[0] = (curr_x - prev_x) / (curr_time - self._roboat_last_update)
         self._roboat_vel[1] = (curr_y - prev_y) / (curr_time - self._roboat_last_update)
      self._roboat_last_update = curr_time

      if self._get_submap is not None:
         submap = self._get_submap(self._roboat_pos, self._roboat_vel)
         submap[submap > 0] = 255
         cv2.imshow('submap', submap)
         cv2.waitKey(1)

      
   def _store_world_state(self, world_state_msg):
      curr_time = time.time()
      prev_pos = copy.deepcopy(self._agents_pos)

      curr_pos, curr_vel = {}, {}
      for agent_msg in world_state_msg.lmpcc_obstacles:
         id = int(agent_msg.id)
         curr_x = agent_msg.pose.position.x
         curr_y = agent_msg.pose.position.y
         curr_pos[id] = np.array([curr_x, curr_y], dtype=float)
         if hasattr(self, '_agents_last_update') and id in prev_pos:
            curr_vel[id] = np.array([
               (curr_x - prev_pos[id][0]) / (curr_time - self._agents_last_update),
               (curr_y - prev_pos[id][1]) / (curr_time - self._agents_last_update)
            ], dtype=float)
         else: curr_vel[id] = np.zeros((2, ), dtype=float)

      self._agents_pos, self._agents_vel = curr_pos, curr_vel
      self._agents_last_update = curr_time


   def _shutdown_callback(self):
      rospy.loginfo('{} was terminated.'.format(self._visual_node_name))


if __name__ == '__main__':

   # Get CLI arguments
   args = rospy.myargv(sys.argv)

   # Start main logic
   SocialVRNN_Predictor('roboat_ros_node', 'Social-VRNN roboat node')

   # Keep node alive until shutdown
   rate = rospy.Rate(10)
   while not rospy.is_shutdown():
      rate.sleep()
