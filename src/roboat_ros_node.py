#!/usr/bin/env python

import os, sys; print("Running with {}".format(sys.version))
import copy, time, pickle, math, collections

from scipy.spatial.transform import Rotation
import tensorflow as tf
import numpy as np
import cv2

from models.SocialVRNN import NetworkModel as SocialVRNN
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import rospy, rospkg
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from social_vrnn.msg import lmpcc_obstacle_array as LMPCC_Obstacle_Array, svrnn_path as SVRNN_Path, svrnn_path_array as SVRNN_Path_Array

PACKAGE_NAME = 'social_vrnn'
QUERY_AGENTS = 4 # includes roboat


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

      # Generic marker
      self._mrk = Marker()
      self._mrk.header.frame_id = 'odom'
      self._mrk.action = Marker.ADD
      self._mrk.lifetime.secs = 1.0
      self._mrk.scale.x, self._mrk.scale.y, self._mrk.scale.z = 1.0, 1.0, 1.0
      self._mrk.color.r, self._mrk.color.g, self._mrk.color.b, self._mrk.color.a = 1.0, 1.0, 1.0, 1.0
      self._mrk.pose.orientation.w = 1.0

      # Load the model
      self.model_args = self._get_model_args('SocialVRNN', '500')
      self.model, self.tf_session = self._load_model(SocialVRNN, self.model_args)

      # Set up subscribers
      rospy.Subscriber('/roboat_cloud/obstacle/map', OccupancyGrid, self._store_occupancy_grid)
      rospy.Subscriber('/ellipse_objects_feed', LMPCC_Obstacle_Array, self._store_world_state)
      rospy.Subscriber('/roboat_localization/odometry_ekf/odometry_filtered', Odometry, self._store_roboat_state)

      # Set up publishers
      self._pred_path_publisher = rospy.Publisher('/{}/predictions'.format(self._node_name), SVRNN_Path_Array, latch=True, queue_size=10)
      self._pos_mark_publisher = rospy.Publisher('/{}/markers/positions'.format(self._node_name), MarkerArray, latch=True, queue_size=10)
      self._pred_mark_publisher = rospy.Publisher('/{}/markers/predictions'.format(self._node_name), MarkerArray, latch=True, queue_size=10)


   def infer(self):
      if self._get_submap is None: return None, None
      if len(self._agents_pos) < QUERY_AGENTS - 1:
         rospy.logwarn('Too few other agents present, skipping path prediction')
         return None, None

      # Maintain past roboat velocities
      if not hasattr(self, '_roboat_vel_ls'):
         self._roboat_vel_ls = np.zeros((self.model_args.prev_horizon + 1, 2), dtype=float)
      self._roboat_vel_ls = np.roll(self._roboat_vel_ls, 1, axis=0)
      self._roboat_vel_ls[0] = self._roboat_vel
      # for i in range(self.model_args.prev_horizon + 1): self._roboat_vel_ls[i] = self._roboat_vel

      # Maintain past agents velocities
      if not hasattr(self, '_agents_vel_ls'):
         self._agents_vel_ls = collections.deque()
         for _ in range(self.model_args.prev_horizon + 1): self._agents_vel_ls.append({})
      self._agents_vel_ls.rotate(1)
      self._agents_vel_ls[0] = self._agents_vel
      # for i in range(self.model_args.prev_horizon + 1): self._agents_vel_ls[i] = self._agents_vel

      # ----------- positions and velocity matrix ----------
      # Collect the position and velocity data ordered by query priority
      positions_ct = self._roboat_pos.reshape((1, -1))
      velocities_tl = self._roboat_vel_ls.reshape((1, -1))

      # Order agents by distance from roboat
      agents_pos_np = np.full((max(self._agents_pos.keys()) + 1, 2), float('inf'), dtype=float)
      for id in self._agents_pos.keys(): agents_pos_np[id] = self._agents_pos[id]
      ord_agents_ind = np.argsort(np.linalg.norm(agents_pos_np - self._roboat_pos, axis=1))
      ord_agents_ind = ord_agents_ind[:QUERY_AGENTS - 1]

      # Set agent positions
      positions_ct = np.concatenate((positions_ct, agents_pos_np[ord_agents_ind]), axis=0)

      # Set agent velocities
      velocities_tl = np.concatenate((velocities_tl, np.zeros((len(ord_agents_ind), velocities_tl.shape[1]), dtype=float)), axis=0)
      for tl_ind in range(len(self._agents_vel_ls)):
         for ag_ind in range(len(ord_agents_ind)):
            if ord_agents_ind[ag_ind] not in self._agents_vel_ls[tl_ind]: assert tl_ind != 0; continue
            velocities_tl[ag_ind + 1][2*tl_ind:2*(tl_ind+1)] = self._agents_vel_ls[tl_ind][ord_agents_ind[ag_ind]]
      # ----------------------------------------------------

      # ------------- relative odometry matrix -------------
      # Obtain object positions ordered by id
      obj_pos = self._roboat_pos.reshape((1, -1))
      agents_pos_np = np.full((max(self._agents_pos.keys()) + 1, 2), float('inf'), dtype=float)
      for id in self._agents_pos.keys(): agents_pos_np[id] = self._agents_pos[id]
      obj_pos = np.concatenate((obj_pos, agents_pos_np[1:]), axis=0)

      # Obtain object headings ordered by id
      obj_vel = self._roboat_vel.reshape((1, -1))
      agents_vel_np = np.full((max(self._agents_vel.keys()) + 1, 2), float('inf'), dtype=float)
      for id in self._agents_vel.keys(): agents_vel_np[id] = self._agents_vel[id]
      obj_vel = np.concatenate((obj_vel, agents_vel_np[1:]), axis=0)

      # Calculate the relative odometry
      relative_odometry = np.zeros((len(positions_ct), self.model_args.n_other_agents * 4), dtype=float)
      for sub_ind, sub_pos in enumerate(positions_ct):
         sub_head = Rotation.from_euler('z', math.atan2(velocities_tl[sub_ind][1], velocities_tl[sub_ind][0]))

         # Get n-nearest neighbours
         nn_inds = np.argsort(np.linalg.norm(obj_pos - sub_pos, axis=1))[1:]
         if len(nn_inds) < self.model_args.n_other_agents: nn_inds = np.concatenate((nn_inds, np.repeat(nn_inds[-1], self.model_args.n_other_agents - len(nn_inds))))
         if len(nn_inds) > self.model_args.n_other_agents: nn_inds = nn_inds[:self.model_args.n_other_agents]

         # Calculate relative positions
         nn_pos = obj_pos[nn_inds].copy()
         nn_pos -= sub_pos
         nn_pos = sub_head.inv().apply(np.concatenate((nn_pos, np.zeros((len(nn_pos), 1))), axis=1))[:, :2]

         # Calculate relative velocities
         nn_vel = obj_vel[nn_inds].copy()
         nn_vel = sub_head.inv().apply(np.concatenate((nn_vel, np.zeros((len(nn_vel), 1))), axis=1))[:, :2]

         # Set relative odometry vector
         relative_odometry[sub_ind] = np.concatenate((nn_pos, nn_vel), axis=1).flatten()
      # ----------------------------------------------------

      # Get the submaps
      submaps = np.zeros((len(positions_ct), self.model_args.submap_width, self.model_args.submap_height), dtype=float)
      for id in range(len(positions_ct)):
         submaps[id] = np.transpose(self._get_submap(positions_ct[id], velocities_tl[id][:2]))
         # submaps[id] = np.zeros((60, 60), dtype=float)

      # Predict the future positions
      return positions_ct, self.model.predict(
         self.tf_session,
         self.model.feed_pred_dic(
            batch_vel = velocities_tl,
            batch_ped_grid = relative_odometry,
            batch_grid = submaps,
            step = 0
         ),
         True
      )[0]


   def publish(self, curr_positions, pred_positions):
      if pred_positions is None: return

      # Publish predicted positions
      pred_positions_path = SVRNN_Path_Array()
      pred_positions_mrk = MarkerArray()
      for id, pred in enumerate(pred_positions):
         path_msg = SVRNN_Path()
         path_msg.id = float(id)
         path_msg.dt = self.model_args.dt
         for mx_id in range(self.model_args.n_mixtures):
            path_mrk = copy.deepcopy(self._mrk)
            path_mrk.id = self.model_args.n_mixtures * id + mx_id
            path_mrk.type = Marker.LINE_STRIP
            path_mrk.scale.x = 0.1
            path_mrk.pose.position.x = curr_positions[id][0]
            path_mrk.pose.position.y = curr_positions[id][1]
            path_mrk.color.r = float(mx_id == 0)
            path_mrk.color.g = float(mx_id == 1)
            path_mrk.color.b = float(mx_id == 2)
            prev_x, prev_y = 0.0, 0.0
            for ts_id in range(self.model_args.prediction_horizon):
               pose = PoseStamped()
               pt = Point()
               idx = ts_id * self.model_args.output_pred_state_dim * self.model_args.n_mixtures + mx_id
               pt.x = prev_x + self.model_args.dt * pred[0][idx]
               pt.y = prev_y + self.model_args.dt * pred[0][idx + self.model_args.n_mixtures]
               path_mrk.points.append(pt)
               pose.pose.position = pt
               path_msg.path.poses.append(pose)
               prev_x, prev_y = pt.x, pt.y
            pred_positions_mrk.markers.append(path_mrk)
         pred_positions_path.paths.append(path_msg)
      self._pred_path_publisher.publish(pred_positions_path)
      self._pred_mark_publisher.publish(pred_positions_mrk)


   def _get_model_args(self, model_name, train_run):
      trained_dir = os.path.join(rospkg.RosPack().get_path(PACKAGE_NAME), 'trained_models')
      model_dir = os.path.join(trained_dir, model_name, train_run)
      convnet_dir = os.path.join(trained_dir, 'autoencoder_with_ped')
      with open(os.path.join(model_dir, 'model_parameters.pkl'), 'rb') as f:
         model_args = pickle.load(f)["args"]
      model_args.model_path = model_dir
      model_args.pretrained_convnet_path = convnet_dir
      model_args.batch_size = QUERY_AGENTS
      model_args.truncated_backprop_length = 1
      model_args.keep_prob = 1.0
      return model_args


   def _load_model(self, model_class, model_args):
      model = model_class(model_args)
      tf_session = tf.Session()
      model.warmstart_model(model_args, tf_session)
      try: model.warmstart_convnet(model_args, tf_session)
      except: rospy.logwarn('Could not warm-start ConvNet')
      rospy.loginfo('Model loaded!')
      return model, tf_session


   def _store_occupancy_grid(self, occupancy_grid_msg):
      # Reset class state
      self._roboat_pos = np.zeros((2, ), dtype=float)
      self._roboat_vel = np.zeros((2, ), dtype=float)
      self._agents_pos, self._agents_vel = {}, {}
      if hasattr(self, '_roboat_last_update'): delattr(self, '_roboat_last_update')
      if hasattr(self, '_agents_last_update'): delattr(self, '_agents_last_update')
      if hasattr(self, '_roboat_vel_ls'): delattr(self, '_roboat_vel_ls')
      if hasattr(self, '_agents_vel_ls'): delattr(self, '_agents_vel_ls')
      rospy.loginfo('Cleared history context')

      # Makes 1 pixel equal to 1 submap pixel
      scale_factor = occupancy_grid_msg.info.resolution / self.model_args.submap_resolution

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
         position: (2,) numpy matrix
            The offset of the position in meters from the origin
            Index 0 is x, index 1 is y
         orientation: (2,) numpy matrix
            The orientation of the position (does not have to be normalised)
            Index 0 is x, index 1 is y
         """
         # Translate to pixel coordinates
         origin_offset = position.copy().astype(float)
         origin_offset /= self.model_args.submap_resolution
         origin_offset[0] = origin_offset[0] + occupancy_grid.shape[1]/2
         origin_offset[1] = occupancy_grid.shape[0]/2 - origin_offset[1]
         origin_offset = origin_offset.astype(int)
         
         # Do bounds-check
         if (origin_offset[0] < self.model_args.submap_width/2 or
             origin_offset[0] > occupancy_grid.shape[1] - self.model_args.submap_width/2 or
             origin_offset[1] < self.model_args.submap_height/2 or
             origin_offset[1] > occupancy_grid.shape[0] - self.model_args.submap_height/2):
            rospy.logerr('Out-of-bounds submap requested!')
            return np.zeros((self.model_args.submap_height, self.model_args.submap_width), dtype=float)
         
         # Rotate to match orientation and return submap
         pad_x = [occupancy_grid.shape[1] - origin_offset[0], origin_offset[0]]
         pad_y = [occupancy_grid.shape[0] - origin_offset[1], origin_offset[1]]
         padded_og = np.pad(occupancy_grid, [pad_y, pad_x], 'constant', constant_values=1.0)
         rot_og = cv2.warpAffine(
            padded_og,
            cv2.getRotationMatrix2D(
               (padded_og.shape[1]/2, padded_og.shape[0]/2),
               -np.rad2deg(math.atan2(orientation[1], orientation[0])),
               1.0
            ),
            (padded_og.shape[1], padded_og.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=1.0
         )
         crop_og = rot_og[pad_y[0] : -pad_y[1], pad_x[0] : -pad_x[1]]
         return crop_og[
            int(origin_offset[1]-self.model_args.submap_height/2) : int(origin_offset[1]+self.model_args.submap_height/2),
            int(origin_offset[0]-self.model_args.submap_width/2) : int(origin_offset[0]+self.model_args.submap_width/2)
         ]

      self._get_submap = get_submap


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

      # if self._get_submap is not None:
      #    submap = self._get_submap(self._roboat_pos, self._roboat_vel)
      #    submap[submap > 0] = 255
      #    cv2.imshow('submap', submap)
      #    cv2.waitKey(1)

      
   def _store_world_state(self, world_state_msg):
      curr_time = time.time()
      prev_pos = copy.deepcopy(self._agents_pos)

      curr_pos, curr_vel = {}, {}
      for agent_msg in world_state_msg.lmpcc_obstacles:
         id = int(agent_msg.id)
         if id > 3: continue

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

      # Publish curent boat positions
      curr_positions_mrk = MarkerArray()
      for id, pos in self._agents_pos.items():
         ag_mrk = copy.deepcopy(self._mrk)
         ag_mrk.id = id
         ag_mrk.type = Marker.SPHERE
         ag_mrk.color.r = float(id % 3 == 0)
         ag_mrk.color.g = float(id % 3 == 1)
         ag_mrk.color.b = float(id % 3 == 2)
         ag_mrk.pose.position.x = pos[0]
         ag_mrk.pose.position.y = pos[1]
         curr_positions_mrk.markers.append(ag_mrk)
      self._pos_mark_publisher.publish(curr_positions_mrk)


   def _shutdown_callback(self):
      rospy.loginfo('{} was terminated.'.format(self._visual_node_name))


if __name__ == '__main__':

   # Get CLI arguments
   args = rospy.myargv(sys.argv)

   # Start main logic
   node = SocialVRNN_Predictor('roboat_ros_node', 'Social-VRNN roboat node')

   # Infer predictions every 'dt' seconds
   rospy.loginfo('Inferring predictions every {} seconds'.format(node.model_args.dt))
   while not rospy.is_shutdown():
      start_time = time.time()

      curr_positions, pred_positions = node.infer()
      node.publish(curr_positions, pred_positions)
      
      elapsed = time.time() - start_time
      if elapsed < node.model_args.dt: rospy.sleep(node.model_args.dt - elapsed)
