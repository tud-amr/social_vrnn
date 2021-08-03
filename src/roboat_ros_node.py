#!/home/jitske/Documents/roboat_ws/venv/bin/python3.6

import os, sys; print("Running with {}".format(sys.version))
import copy, time, pickle, math, collections

from scipy.spatial.transform import Rotation
import tensorflow as tf
import numpy as np; np.set_printoptions(suppress=True)

from models.SocialVRNN import NetworkModel as SocialVRNN
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import rospy, rospkg
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Point, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from social_vrnn.msg import lmpcc_obstacle_array as LMPCC_Obstacle_Array, svrnn_path as SVRNN_Path, svrnn_path_array as SVRNN_Path_Array

sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2

PACKAGE_NAME = 'social_vrnn'
MAX_QUERY_AGENTS = 100
SWITCH_AXES = False


class SocialVRNN_Predictor:
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
      self.get_submap = None # function to request submap for some position
      self.agents_pos, self.agents_vel = {}, {} # dictionary of world state

      # Generic marker template
      mrk = Marker()
      mrk.header.frame_id = 'odom'
      mrk.action = Marker.ADD
      mrk.lifetime.secs = int(1.0)
      mrk.scale.x, mrk.scale.y, mrk.scale.z = 1.0, 1.0, 1.0
      mrk.color.r, mrk.color.g, mrk.color.b, mrk.color.a = 1.0, 1.0, 1.0, 1.0
      mrk.pose.orientation.w = 1.0
      self.marker_template = mrk

      # Load the model
      self.model_args = self.get_model_args('SocialVRNN', '666')
      self.model, self.tf_session = self.load_model(SocialVRNN, self.model_args)

      # Set up subscribers
      rospy.Subscriber('/roboat_cloud/obstacle/map', OccupancyGrid, self.store_occupancy_grid)
      rospy.Subscriber('/ellipse_objects_feed', LMPCC_Obstacle_Array, self.store_world_state)
      rospy.Subscriber('/roboat_localization/odometry_ekf/odometry_filtered', Odometry, self.store_roboat_state)

      # Set up publishers
      self.pred_path_publisher = rospy.Publisher('/{}/predictions'.format(self.node_name), SVRNN_Path_Array, latch=True, queue_size=10)
      self.pos_mark_publisher = rospy.Publisher('/{}/markers/positions'.format(self.node_name), MarkerArray, latch=True, queue_size=10)
      self.pred_mark_publisher = rospy.Publisher('/{}/markers/predictions'.format(self.node_name), MarkerArray, latch=True, queue_size=10)


   def infer(self):
      if self.get_submap is None: return None, None
      if len(self.agents_pos) < 2:
         rospy.logwarn('No other agents present, skipping path prediction')
         return None, None

      # Build NumPy matrices from world state
      active_agents = np.full((MAX_QUERY_AGENTS, ), False, dtype=bool)
      for id in self.agents_pos.keys(): active_agents[id] = True

      agents_pos_np = np.zeros((MAX_QUERY_AGENTS, 2), dtype=float)
      for id, pos in self.agents_pos.items(): agents_pos_np[id] = pos

      agents_vel_np = np.zeros((MAX_QUERY_AGENTS, 2), dtype=float)
      for id, vel in self.agents_vel.items(): agents_vel_np[id] = vel

      # Maintain past agents velocities
      if not hasattr(self, '_agents_vel_ls'):
         self._agents_vel_ls = np.zeros((MAX_QUERY_AGENTS, 2 * (self.model_args.prev_horizon + 1)), dtype=float)
      self._agents_vel_ls = np.roll(self._agents_vel_ls, 2, axis=1)
      self._agents_vel_ls[:, :2] = agents_vel_np

      # Calculate the relative odometry
      relative_odometry = np.zeros((MAX_QUERY_AGENTS, 4 * self.model_args.n_other_agents), dtype=float)
      for sub_ind in np.arange(MAX_QUERY_AGENTS)[active_agents]:

         # Get n-nearest neighbours
         nn_inds = np.argsort(np.linalg.norm(agents_pos_np[active_agents] - agents_pos_np[sub_ind], axis=1))[1:]
         if len(nn_inds) < self.model_args.n_other_agents: nn_inds = np.concatenate((nn_inds, np.repeat(nn_inds[-1], self.model_args.n_other_agents - len(nn_inds))))
         if len(nn_inds) > self.model_args.n_other_agents: nn_inds = nn_inds[:self.model_args.n_other_agents]

         # Calculate relative positions
         nn_pos = agents_pos_np[active_agents][nn_inds].copy()
         nn_pos -= agents_pos_np[sub_ind]

         # Calculate relative velocities
         nn_vel = agents_vel_np[active_agents][nn_inds].copy()
         nn_vel -= agents_vel_np[sub_ind]

         # Set relative odometry vector
         relative_odometry[sub_ind] = np.concatenate((nn_pos, nn_vel), axis=1).flatten()

      # Get the submaps
      submaps = np.zeros((MAX_QUERY_AGENTS, self.model_args.submap_width, self.model_args.submap_height), dtype=float)
      for id in np.arange(MAX_QUERY_AGENTS)[active_agents]:
         if self.get_submap is None: return
         submaps[id] = np.transpose(self.get_submap(agents_pos_np[id], agents_vel_np[id]))
         # submaps[id] = np.zeros((60, 60), dtype=float) # disable occupancies (for debugging)

      # Predict the future velocities
      return copy.deepcopy(self.agents_pos), self.model.predict(
         self.tf_session,
         self.model.feed_pred_dic(
            batch_vel = self._agents_vel_ls,
            batch_ped_grid = relative_odometry,
            batch_grid = submaps,
            step = 0
         ),
         True
      )[0]


   def publish(self, agents_pos, pred_velocities):
      if pred_velocities is None: return

      # Publish predicted positions
      pred_velocities_path = SVRNN_Path_Array()
      pred_velocities_mrk = MarkerArray()
      for id in agents_pos.keys():
         pred = pred_velocities[id]
         path_msg = SVRNN_Path()
         path_msg.id = float(id)
         path_msg.dt = self.model_args.dt
         for mx_id in range(self.model_args.n_mixtures):
            path_mrk = copy.deepcopy(self.marker_template)
            path_mrk.id = self.model_args.n_mixtures * id + mx_id
            path_mrk.type = Marker.LINE_STRIP
            path_mrk.scale.x = 0.1
            path_mrk.pose.position.x = agents_pos[id][1] if SWITCH_AXES else agents_pos[id][0]
            path_mrk.pose.position.y = agents_pos[id][0] if SWITCH_AXES else agents_pos[id][1]
            path_mrk.color.r = float(mx_id == 0)
            path_mrk.color.g = float(mx_id == 1)
            path_mrk.color.b = float(mx_id == 2)
            prev_x, prev_y = 0.0, 0.0
            for ts_id in range(self.model_args.prediction_horizon):
               pose = PoseStamped()
               pt = Point()
               idx = ts_id * self.model_args.output_pred_state_dim * self.model_args.n_mixtures + mx_id
               pt.x = prev_x + self.model_args.dt * pred[0][idx + (self.model_args.n_mixtures if SWITCH_AXES else 0)]
               pt.y = prev_y + self.model_args.dt * pred[0][idx + (0 if SWITCH_AXES else self.model_args.n_mixtures)]
               pose.pose.position = pt
               path_msg.path.poses.append(pose)
               pt.z = 0.2
               path_mrk.points.append(pt)
               prev_x, prev_y = pt.x, pt.y
            pred_velocities_mrk.markers.append(path_mrk)
         pred_velocities_path.paths.append(path_msg)
      self.pred_path_publisher.publish(pred_velocities_path)
      self.pred_mark_publisher.publish(pred_velocities_mrk)


   def get_model_args(self, model_name, train_run):
      trained_dir = os.path.join(rospkg.RosPack().get_path(PACKAGE_NAME), 'trained_models')
      model_dir = os.path.join(trained_dir, model_name, train_run)
      convnet_dir = os.path.join(trained_dir, 'autoencoder_with_ped')
      with open(os.path.join(model_dir, 'model_parameters.pkl'), 'rb') as f:
         model_args = pickle.load(f)["args"]
      model_args.model_path = model_dir
      model_args.pretrained_convnet_path = convnet_dir
      model_args.batch_size = MAX_QUERY_AGENTS
      model_args.truncated_backprop_length = 1
      model_args.keep_prob = 1.0
      return model_args


   def load_model(self, model_class, model_args):
      model = model_class(model_args)
      tf_session = tf.Session()
      model.warmstart_model(model_args, tf_session)
      try: model.warmstart_convnet(model_args, tf_session)
      except: rospy.logwarn('Could not warm-start ConvNet')
      rospy.loginfo('Model loaded!')
      return model, tf_session


   def store_occupancy_grid(self, occupancy_grid_msg):
      # Reset class state
      def safedelattr(attr):
         if hasattr(self, attr): delattr(self, attr)
      safedelattr('_roboat_pos'); safedelattr('_roboat_last_update')
      self.agents_pos, self.agents_vel = {}, {}; safedelattr('_agents_last_update')
      self.get_submap = None; safedelattr('_agents_vel_ls')
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

      self.get_submap = get_submap

      
   def store_world_state(self, world_state_msg):
      curr_time = time.time()
      prev_pos = copy.deepcopy(self.agents_pos)

      curr_pos, curr_vel = {}, {}
      if hasattr(self, '_roboat_pos'):
         curr_pos[0], curr_vel[0] = self._roboat_pos.copy(), self._roboat_vel.copy()

      for agent_msg in world_state_msg.lmpcc_obstacles:
         id = int(agent_msg.id)
         # if id > 2: continue # only process small subset (for debugging)

         curr_x, curr_y = agent_msg.pose.position.x, agent_msg.pose.position.y
         if SWITCH_AXES: curr_x, curr_y = curr_y, curr_x
         curr_pos[id] = np.array([curr_x, curr_y], dtype=float)
         if hasattr(self, '_agents_last_update') and id in prev_pos:
            curr_vel[id] = np.array([
               (curr_x - prev_pos[id][0]) / (curr_time - self._agents_last_update),
               (curr_y - prev_pos[id][1]) / (curr_time - self._agents_last_update)
            ], dtype=float)
         else: curr_vel[id] = np.zeros((2, ), dtype=float)

      self.agents_pos, self.agents_vel = curr_pos, curr_vel
      self._agents_last_update = curr_time

      # Publish curent boat positions
      curr_positions_mrk = MarkerArray()
      for id, pos in self.agents_pos.items():
         ag_mrk = copy.deepcopy(self.marker_template)
         ag_mrk.id = int(id)
         ag_mrk.type = Marker.SPHERE
         ag_mrk.color.r = float(id % 3 == 0)
         ag_mrk.color.g = float(id % 3 == 1)
         ag_mrk.color.b = float(id % 3 == 2)
         ag_mrk.pose.position.x = pos[1] if SWITCH_AXES else pos[0]
         ag_mrk.pose.position.y = pos[0] if SWITCH_AXES else pos[1]
         curr_positions_mrk.markers.append(ag_mrk)
      self.pos_mark_publisher.publish(curr_positions_mrk)


   def store_roboat_state(self, roboat_state_msg):
      if not hasattr(self, '_roboat_pos'):
         self._roboat_pos = np.zeros((2, ), dtype=float)
         self._roboat_vel = np.zeros((2, ), dtype=float)

      curr_time = time.time()
      prev_x, prev_y = self._roboat_pos[0], self._roboat_pos[1]
      curr_x, curr_y = roboat_state_msg.pose.pose.position.x, roboat_state_msg.pose.pose.position.y
      if SWITCH_AXES: curr_x, curr_y = curr_y, curr_x
      self._roboat_pos[0], self._roboat_pos[1] = curr_x, curr_y
      if hasattr(self, '_roboat_last_update'):
         self._roboat_vel[0] = (curr_x - prev_x) / (curr_time - self._roboat_last_update)
         self._roboat_vel[1] = (curr_y - prev_y) / (curr_time - self._roboat_last_update)
      self._roboat_last_update = curr_time


   def shutdown_callback(self):
      rospy.loginfo('{} was terminated.'.format(self.visual_node_name))


if __name__ == '__main__':

   # Get CLI arguments
   args = rospy.myargv(sys.argv)

   # Start main logic
   node = SocialVRNN_Predictor('roboat_ros_node', 'Social-VRNN roboat node')

   # Infer predictions every 'dt' seconds
   rospy.loginfo('Inferring predictions every {} seconds'.format(node.model_args.dt))
   while not rospy.is_shutdown():
      start_time = time.time()

      agents_pos, pred_velocities = node.infer()
      node.publish(agents_pos, pred_velocities)
      
      elapsed = time.time() - start_time
      if elapsed < node.model_args.dt: rospy.sleep(node.model_args.dt - elapsed)
