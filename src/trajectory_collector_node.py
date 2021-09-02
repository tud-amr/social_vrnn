#!/usr/bin/env python3

import sys; print("Running with {}".format(sys.version))
import time

import numpy as np; np.set_printoptions(suppress=True)

import rospy
from nav_msgs.msg import OccupancyGrid
from social_vrnn.msg import lmpcc_obstacle_array as LMPCC_Obstacle_Array, svrnn_path as SVRNN_Path, svrnn_path_array as SVRNN_Path_Array

# AGENT_ID = 11
# SCENARIO_NAME = "crossing_right"
# AGENT_ID = 1
# SCENARIO_NAME = "crossing_left"
# AGENT_ID = 23
# SCENARIO_NAME = "head_on"
AGENT_ID = 11
SCENARIO_NAME = "take_over"


class TrajectoryCollector:
   """ 
   ROS interface to the Social-VRNN roboat prediction network.
   > see https://github.com/tud-amr/social_vrnn/tree/roboat-vrnn-vessel
   """


   def __init__(self, node_name, visual_node_name):
      self.node_name, self.visual_node_name = node_name, visual_node_name

      # Bind node
      rospy.init_node(self.node_name)
      rospy.on_shutdown(self.shutdown_callback)
      rospy.loginfo('{} has started.'.format(self.visual_node_name))

      # Set up class variables
      self.agents_pos, self.agents_vel = {}, {} # dictionary of world state
      self.prediction_paths = {}
      self.odom_matrix = None
      self.pred_matrix = None
      self.state_counter = 0

      # Set up subscribers
      rospy.Subscriber('/roboat_cloud/obstacle/map', OccupancyGrid, self.store_occupancy_grid)
      rospy.Subscriber('/ellipse_objects_feed', LMPCC_Obstacle_Array, self.store_world_state)
      rospy.Subscriber('/roboat_ros_node/predictions', SVRNN_Path_Array, self.store_predictions)


   def collect_data(self):
      if self.state_counter < 2: return
      elif self.state_counter > 2:
         rospy.loginfo("Writing data...")
         rospy.loginfo(f"odom_matrix: {self.odom_matrix.shape}")
         np.save(f'numpy/{SCENARIO_NAME}_odometry', self.odom_matrix)
         rospy.loginfo(f"pred_matrix: {self.pred_matrix.shape}")
         np.save(f'numpy/{SCENARIO_NAME}_predictions', self.pred_matrix)
         rospy.signal_shutdown("Finished execution")
         return
      rospy.loginfo_once("Starting data collection...")

      if AGENT_ID not in self.agents_pos or \
         AGENT_ID not in self.agents_vel or \
         AGENT_ID not in self.prediction_paths:
         rospy.logwarn("No agent data, ignoring current time step")
         self.agents_pos, self.agents_vel, self.prediction_paths = {}, {}, {}
         return

      odom_vector = np.concatenate((self.agents_pos[AGENT_ID], self.agents_vel[AGENT_ID])).reshape((1, -1))
      pred_vector = np.zeros((20, 2), dtype=float)
      for index, pose in enumerate(self.prediction_paths[AGENT_ID][:20]):
         pred_vector[index][0] = pose.pose.position.x
         pred_vector[index][1] = pose.pose.position.y

      if self.odom_matrix is None:
         self.odom_matrix = odom_vector
         self.pred_matrix = pred_vector.reshape((1, -1, 2))
      else:
         self.odom_matrix = np.concatenate((self.odom_matrix, odom_vector), axis=0)
         self.pred_matrix = np.concatenate((self.pred_matrix, pred_vector.reshape((1, -1, 2))), axis=0)

      self.agents_pos, self.agents_vel, self.prediction_paths = {}, {}, {}

   
   def store_predictions(self, path_array_msg):
      self.prediction_paths = {int(p.id): p.path.poses for p in path_array_msg.paths}

      
   def store_world_state(self, world_state_msg):
      curr_pos, curr_vel = {}, {}
      for agent_msg in world_state_msg.lmpcc_obstacles:
         id = int(agent_msg.id)
         curr_pos[id] = np.array(
            [
               agent_msg.pose.position.x,
               agent_msg.pose.position.y,
               agent_msg.pose.orientation.z,
            ],
            dtype=float,
         )
         curr_vel[id] = np.array(
            [
               agent_msg.velocity.linear.x,
               agent_msg.velocity.linear.y,
               agent_msg.velocity.angular.z,
            ],
            dtype=float,
         )
      if len(curr_pos) > 0:
         rospy.loginfo_once(f"Boat IDs: {set(curr_pos.keys())}")
      self.agents_pos, self.agents_vel = curr_pos, curr_vel


   def store_occupancy_grid(self, occupancy_grid_msg):
      self.state_counter += 1


   def shutdown_callback(self):
      rospy.loginfo('{} was terminated.'.format(self.visual_node_name))


if __name__ == '__main__':

   # Get CLI arguments
   args = rospy.myargv(sys.argv)

   # Start main logic
   node = TrajectoryCollector('trajectory_collector_node', 'MS data collector node')

   # Infer predictions every 'dt' seconds
   dt_seconds = 1.0
   while not rospy.is_shutdown():
      start_time = time.time()

      node.collect_data()
      
      elapsed = time.time() - start_time
      if elapsed < dt_seconds: rospy.sleep(dt_seconds - elapsed)
