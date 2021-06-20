#!/usr/bin/env python

import os
import sys
import time
import pickle
import numpy as np

import tensorflow as tf
from models.SocialVRNN import NetworkModel as SocialVRNN
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import rospkg
import rospy
from nav_msgs.msg import OccupancyGrid, Odometry
from social_vrnn.msg import lmpcc_obstacle_array as LMPCC_Obstacle_Array

PACKAGE_NAME = 'social_vrnn'
TRAIN_RUN = '500'
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
      trained_models_dir = os.path.join(rospkg.RosPack().get_path(PACKAGE_NAME), 'trained_models')
      self._model, self.tf_session = self._load_model('SocialVRNN', SocialVRNN, trained_models_dir)
      self._occupancy_grid = None

      # Set up subscribers
      rospy.Subscriber('/roboat_cloud/obstacle/map', OccupancyGrid, self._store_occupancy_grid)
      rospy.Subscriber('/ellipse_objects_feed', LMPCC_Obstacle_Array, self._store_world_state)
      rospy.Subscriber('/roboat_localization/odometry_ekf/odometry_filtered', Odometry, self._store_roboat_state)

   def _load_model(self, model_name, model_class, trained_dir):
      # Load model arguments
      model_dir = os.path.join(trained_dir, model_name, TRAIN_RUN)
      convnet_dir = os.path.join(trained_dir, 'autoencoder_with_ped')
      with open(os.path.join(model_dir, 'model_parameters.pkl'), 'rb') as f:
         model_args = pickle.load(f)["args"]
      model_args.model_path = model_dir
      model_args.pretrained_convnet_path = convnet_dir
      model_args.batch_size = QUERY_AGENTS
      model_args.truncated_backprop_length = 1 # why?
      model_args.keep_prob = 1.0

      # Load model
      model = model_class(model_args)
      tf_session = tf.Session()
      model.warmstart_model(model_args, tf_session)
      try: model.warmstart_convnet(model_args, tf_session)
      except: rospy.logwarn("Could not warm-start ConvNet")
      rospy.loginfo("Model loaded!")
      return model, tf_session

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
      # rospy.loginfo('Received world state.')
      pass

   def _store_roboat_state(self, roboat_state_msg):
      # rospy.loginfo('Received roboat state.')
      pass
   
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
