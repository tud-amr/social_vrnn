#!/usr/bin/env python
import time
import rospy
import rospkg
import cv2
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Path, OccupancyGrid, Odometry
from nav_msgs.srv import GetMap
from visualization_msgs.msg import MarkerArray, Marker
# from derived_object_msgs.msg import ObjectArray
import sys
import os
import imutils


import math

print('Python %s on %s' % (sys.version, sys.platform))
print(os.getcwd())
sys.path.extend([os.getcwd()+'/data_utils'])
sys.path.extend([os.getcwd()+'/models'])

if sys.version_info[0] < 3:
	print("Using Python " + str(sys.version_info[0]))
	from data_utils import Support as sup
else:
	print("Using Python " + str(sys.version_info[0]))
	from src.data_utils import DataHandler as dh
	from src.data_utils import DataHandlerLSTM as dhlstm
	from src.data_utils import Support as sup
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as pl
from scipy.stats import multivariate_normal
from social_vrnn.msg import lmpcc_obstacle, lmpcc_obstacle_array

import pickle as pkl
import threading as th
import importlib
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Import model
from models.SocialVRNN import NetworkModel

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# from VDGNN_simple import NetworkModel

class SocialVDGNN:
	def __init__(self):

		self.model_name = rospy.get_param('~model_name', 'SocialVRNN')
		self.model_id = rospy.get_param('~id', '123')
		# ROS Topics
		self.other_agents_topic = rospy.get_param('~other_agents_topic', "/carla/objects")
		self.robot_state_topic = rospy.get_param('~robot_state_topic', "/carla/ego_vehicle/odometry")
		self.grid_topic = rospy.get_param('~grid_topic', "/move_base/local_costmap/costmap")
		self.robot_plan_topic = rospy.get_param('~robot_plan_topic', "/predicted_trajectory")
		self.reset_topic = rospy.get_param('~reset_topic', "/lmpcc/initialpose")

		self.n_peds = 6
		self.n_robots = 0
		self.add_robot = False
		self.n_query_agents = self.n_peds + self.n_robots

		self.prediction_steps = rospy.get_param('~prediction_steps', 20)

		# TODO: make the number of other agents variable
		# self.n_other_gents = 6
		# Load Model Parameters
		self.load_args()

		# Robot State variables
		self.robot_state_ = Pose()
		self.robot_plan_ = Path()
		for i in range(self.prediction_steps):
			self.robot_plan_.poses.append(PoseStamped())

		# Pedestrians State variables
		self.current_ids_ = np.zeros([self.n_query_agents, 1])
		self.current_position_ = np.zeros([self.n_query_agents, (self.model_args.prev_horizon + 1) * 2])
		self.current_velocity_ = np.zeros([self.n_query_agents, (self.model_args.prev_horizon + 1) * 2])
		self.predicted_positions = np.zeros([self.n_query_agents, self.model_args.n_mixtures, self.model_args.prediction_horizon, 2])
		self.predicted_velocities = np.zeros([self.n_query_agents, self.model_args.n_mixtures, self.model_args.prediction_horizon, 2])
		self.predicted_uncertainty = np.zeros([self.n_query_agents, self.model_args.n_mixtures, self.model_args.prediction_horizon, 2])
		self.add_noise = False

		self.other_pedestrians = []
		for i in range(self.n_peds):
			pass
			# self.other_pedestrians.append(TrackedPerson())

		self.load_model()

		if self.model_args.others_info == "angular_grid":
			self.other_agents_info = np.zeros(
				[self.n_query_agents, self.model_args.pedestrian_vector_dim])
		else:
			self.other_agents_info = np.zeros([self.n_query_agents, self.model_args.pedestrian_vector_dim*self.model_args.n_other_agents])

		self.width = int(self.model_args.submap_width / self.model_args.submap_resolution)
		self.height = int(self.model_args.submap_height / self.model_args.submap_resolution)
		self.batch_grid = np.zeros([self.n_query_agents, self.width, self.height])
		self.fig_animate = pl.figure('Animation')
		self.fig_width = 12  # width in inches
		self.fig_height = 25  # height in inches
		self.fig_size = [self.fig_width, self.fig_height]
		self.fontsize = 9
		self.params = {'backend': 'ps',
		               'axes.labelsize': self.fontsize,
		               'font.size': self.fontsize,
		               'xtick.labelsize': self.fontsize,
		               'ytick.labelsize': self.fontsize,
		               'figure.figsize': self.fig_size}
		pl.rcParams.update(self.params)
		self.ax_pos = pl.subplot()
		pl.show(block=False)

		self.colors = []
		self.colors.append([0.8500, 0.3250, 0.0980])  # orange
		self.colors.append([0.0, 0.4470, 0.7410])  # blue
		self.colors.append([0.4660, 0.6740, 0.1880])  # green
		self.colors.append([0.4940, 0.1840, 0.5560])  # purple
		self.colors.append([0.9290, 0.6940, 0.1250])  # yellow
		self.colors.append([0.3010, 0.7450, 0.9330])  # cyan
		self.colors.append([0.6350, 0.0780, 0.1840])  # chocolate
		self.colors.append([1, 0.6, 1])  # pink
		self.colors.append([0.0, 0.405, 1.0])  # grey

		self.is_first_step = True
		# ROS Subscribers
		# rospy.Subscriber(self.other_agents_topic, ObjectArray, self.other_agents_CB, queue_size=1)
		#rospy.Subscriber(self.grid_topic, OccupancyGrid, self.grid_CB, queue_size=1)
		# rospy.Subscriber(self.robot_state_topic, Odometry, self.robot_state_CB, queue_size=1)
		# rospy.Subscriber(self.robot_plan_topic, Path, self.robot_plan_CB, queue_size=1)

		# ROS Service Clients

		# ROS Publishers
		self.pub_viz = rospy.Publisher('social_vdgnn_predictions', MarkerArray, queue_size=10)
		self.obstacles_publisher = rospy.Publisher('ellipse_objects_feed', lmpcc_obstacle_array, queue_size=10)
		self.pub_robot_traj = rospy.Publisher('plannet_trajectory2', Path, queue_size=10)

		# THread control
		self.lock = th.Lock()

	def load_args(self):
		cwd = os.getcwd()
		
		model_path = os.path.join(rospkg.RosPack().get_path('social_vrnn'), 'trained_models', self.model_name, str(self.model_id))
		
		print("Loading data from: '{}'".format(model_path))
		file = open(model_path + '/model_parameters.pkl', 'rb')
		if sys.version_info[0] < 3:
			model_parameters = pkl.load(file)  # ,encoding='latin1')
		else:
			model_parameters = pkl.load(file, encoding='latin1')
		file.close()
		self.model_args = model_parameters["args"]
		self.model_args.model_path = model_path
		self.model_args.pretrained_convnet_path = os.path.normpath(cwd) + '/trained_models/autoencoder_with_ped'
		# change some args because we are doing inference
		self.model_args.truncated_backprop_length = 1
		self.model_args.batch_size = self.n_query_agents

		self.n_other_agents = self.model_args.n_other_agents
		self.other_agents = []

	def load_model(self):

		self.model_args.truncated_backprop_length = 1
		self.model_args.keep_prob = 1.0
		self.model = NetworkModel(self.model_args)

		self.sess = tf.Session()

		self.model.warmstart_model(self.model_args, self.sess)
		try:
			self.model.warmstart_convnet(self.model_args, self.sess)
		except:
			print("No convnet")

		print("Model Initialized")

	def get_local_grid(self,pos,vel):
		""" Extract occupancy grid from map around specified position and angle
		@param map_state: OccupancyGrid
		@param grid_size: int
		@param pos: [pos]
		@param angle: float
		@	return:
		"""
		angle = np.arctan2(vel[1], vel[0])

		if np.isnan(angle).any() or np.isnan(pos).any():
			local_grid = np.empty([int(np.ceil(self.model_args.submap_width / self.model_args.submap_resolution)),
			                       int(np.ceil(self.model_args.submap_height / self.model_args.submap_resolution))])
			local_grid.fill(np.nan)
			return local_grid

		center = (
			(pos[0] - self.map.info.origin.position.x) / self.map.info.resolution,
			(pos[1] - self.map.info.origin.position.y) / self.map.info.resolution
		)

		angle = (angle * 180 / np.pi)

		rotated_map = imutils.rotate_bound(self.gridmap.astype("float32"), angle)

		dst = (int(self.model_args.submap_height/self.model_args.submap_resolution),
		       int(self.model_args.submap_width/self.model_args.submap_resolution))
		local_grid = cv2.getRectSubPix(rotated_map.astype("float32"), dst, center)

		if False:
			self.ax_pos.clear()
			sup.plot_grid(self.ax_pos, np.array([0.0, 0.0]), local_grid, self.model_args.submap_resolution,
			              np.array([self.model_args.submap_width, self.model_args.submap_height])*self.model_args.submap_resolution)
			self.ax_pos.set_xlim([-self.model_args.submap_width / 2, self.model_args.submap_width / 2])
			self.ax_pos.set_ylim([-self.model_args.submap_height / 2, self.model_args.submap_height / 2])
			self.ax_pos.set_aspect('equal')
			self.fig_animate.canvas.draw()
			pl.show(block=False)

		return local_grid

	def robot_plan_CB(self, msg):
		self.robot_plan_ = msg

	def robot_state_CB(self, msg):
		self.robot_state_ = msg.pose.pose

	def grid_CB(self, data):
		# scale the grid from 0-100 to 0-1 and invert
		print("Grid data size: " + str(len(data.data)))
		self.grid[0, 0, :, :] = (np.asarray(data.data).reshape((self.width, self.height)).astype(float) / 100.0)
		self.grid[0, 0, :, :] = np.flip(self.grid[0, 0, :, :], 1)
		self.grid[0, 0, :, :] = sup.rotate_grid_around_center(self.grid[0, 0, :, :], 90)

		if False:
			self.ax_pos.clear()
			sup.plot_grid(self.ax_pos, np.array([0.0, 0.0]), self.grid, self.model_args.submap_resolution,
			              np.array([self.model_args.submap_width, self.model_args.submap_height]))
			self.ax_pos.set_xlim([-self.model_args.submap_width / 2, self.model_args.submap_width / 2])
			self.ax_pos.set_ylim([-self.model_args.submap_height / 2, self.model_args.submap_height / 2])
			self.ax_pos.set_aspect('equal')

	def other_agents_CB(self, data):
		"""
		Obtain surrouding agents information callback
		:param data: TrackedPerson
		:return:
		"""
		# Assuming that the information about the agents comes always ordered
		# Shift old states to fill the vector with new info
		# self.current_position_ = np.roll(self.current_position_, 2, axis=2)
		if data.objects:

			self.n_other_agents = len(data.objects)
			self.other_agents = data.objects

			self.current_velocity_ = np.roll(self.current_velocity_, 2, axis=1)
			# if len(data.objects) != self.n_query_agents:
			# 	print("NUmber of peds do not match callback info")

			other_poses_ordered = np.zeros((len(data.objects), 6))
			car_pose = np.array([self.robot_state_.position.x,self.robot_state_.position.y])

			# Save all the ped data (x, y, vx, vy, distance to vehicle, id)
			for i, ped in enumerate(data.objects):
				other_poses_ordered[i, :2] = np.array([ped.pose.position.x, ped.pose.position.y])
				other_poses_ordered[i, 2:4] = np.array([ped.twist.linear.x, ped.twist.linear.y])
				other_poses_ordered[i, 4] = np.linalg.norm(car_pose-other_poses_ordered[i, :2])  # : to i
				other_poses_ordered[i, 5] = ped.id

				# The vehicle is in the obstacle list, filter it by distance
				if(other_poses_ordered[i, 4] < 1.0):
					other_poses_ordered[i, 4] = 999999.0

			# Order Agents (happens correctly)
			other_poses_ordered = other_poses_ordered[other_poses_ordered[:, 4].argsort()]

			if self.is_first_step:
				self.is_first_step = False
				self.prev_ordered_agents = other_poses_ordered

			reset_seq = np.zeros([self.n_query_agents])

			for person_it in range(self.n_query_agents):

				# Check if order of agents has changed
				if (len(self.prev_ordered_agents) > person_it + 1 and len(other_poses_ordered) > person_it + 1) and\
						self.prev_ordered_agents[person_it,5] != other_poses_ordered[person_it,5]:
					self.current_position_[person_it, 0] = other_poses_ordered[person_it,0]
					self.current_position_[person_it, 1] = other_poses_ordered[person_it,1]
					self.current_ids_[person_it, 0] = other_poses_ordered[person_it,5]

					# Reset Velocities
					for prev_step in range(self.model_args.prev_horizon + 1):
						self.current_velocity_[person_it, 2*prev_step] = other_poses_ordered[person_it,2]
						self.current_velocity_[person_it, 2*prev_step+1] = other_poses_ordered[person_it,3]

					reset_seq[person_it] = 1

				else:
					if(len(other_poses_ordered) > person_it + 1):
						self.current_position_[person_it, 0] = other_poses_ordered[person_it,0]
						self.current_position_[person_it, 1] = other_poses_ordered[person_it,1]
						self.current_velocity_[person_it, 0] = other_poses_ordered[person_it,2] + np.random.normal(0, 0.1) * self.add_noise
						self.current_velocity_[person_it, 1] = other_poses_ordered[person_it,3] + np.random.normal(0, 0.1) * self.add_noise

			# Reset Hidden states
			self.model.reset_test_cells(reset_seq)

			self.prev_ordered_agents = other_poses_ordered

		else:
			self.current_velocity_[:,2:4] = self.current_velocity_[:,0:2]

	def fillBatchOtherAgents(self, person_it):

		if self.other_agents:
			other_poses_ordered = np.zeros((self.n_other_agents, 6))
			if person_it != -1:
				current_pos = self.current_position_[person_it, 0:2]
				current_vel = self.current_position_[person_it, 2:4]
			else:
				current_pos = np.array(
					[self.robot_state_.position.x,
					 self.robot_state_.position.y])
				current_vel = np.array(
					[self.robot_state_.position.z * np.cos(self.robot_state_.orientation.z),
					 self.robot_state_.position.z * np.sin(self.robot_state_.orientation.z)])

			ag_id = 0
			for k in range(self.n_other_agents):
				if self.other_agents[k].id != self.current_ids_[person_it,0]:
					other_poses_ordered[ag_id, 0] = self.other_agents[k].pose.position.x - current_pos[0]
					other_poses_ordered[ag_id, 1] = self.other_agents[k].pose.position.y - current_pos[1]
					other_poses_ordered[ag_id, 2] = self.other_agents[k].twist.linear.x - current_vel[0]
					other_poses_ordered[ag_id, 3] = self.other_agents[k].twist.linear.y - current_vel[1]
					other_poses_ordered[ag_id, 4] = np.linalg.norm(other_poses_ordered[ag_id, :2])
					other_poses_ordered[ag_id, 5] = np.arctan2(other_poses_ordered[ag_id, 1], other_poses_ordered[ag_id, 0])
					ag_id += 1

			# Adding robot
			if self.add_robot:
				other_poses_ordered[-1, 0] = self.robot_state_.position.x - current_pos[0]
				other_poses_ordered[-1, 1] = self.robot_state_.position.y - current_pos[1]
				other_poses_ordered[-1, 2] = self.robot_state_.position.z * np.cos(self.robot_state_.orientation.z) - current_vel[0]
				other_poses_ordered[-1, 3] = self.robot_state_.position.z * np.sin(self.robot_state_.orientation.z) - current_vel[1]
				other_poses_ordered[-1, 4] = np.linalg.norm(other_poses_ordered[-1, :2])
				other_poses_ordered[-1, 5] = np.arctan2(other_poses_ordered[-1, 1], other_poses_ordered[-1, 0])
				other_poses_ordered[-1, 6] = -1
				other_poses_ordered[-1] *= 0

			other_poses_ordered = other_poses_ordered[other_poses_ordered[:, 4].argsort()]
			if self.model_args.others_info == "angular_grid":
				heading = math.atan2(current_vel[1], current_vel[0])
				other_pos_local_frame = sup.positions_in_local_frame(current_pos, heading, other_poses_ordered[:, :2])
				self.other_agents_info[person_it, :] = sup.compute_radial_distance_vector(self.model_args.pedestrian_vector_dim,
				                                                                          other_pos_local_frame,
				                                                                          max_range=self.model_args.max_range_ped_grid,
				                                                                          min_angle=0,
				                                                                          max_angle=2 * np.pi,
				                                                                          normalize=True)
			else:
				for it in range(min(self.model_args.n_other_agents,self.n_other_agents)):
					self.other_agents_info[person_it, it*self.model_args.pedestrian_vector_dim:(it+1)*self.model_args.pedestrian_vector_dim] = \
						other_poses_ordered[it, : self.model_args.pedestrian_vector_dim]
				for i in range(it+1,self.model_args.n_other_agents):
					self.other_agents_info[person_it, i*self.model_args.pedestrian_vector_dim:(i+1)*self.model_args.pedestrian_vector_dim] = \
						other_poses_ordered[it, : self.model_args.pedestrian_vector_dim]

		#self.batch_grid[person_it] = self.get_local_grid(current_pos,current_vel)

	# query feed the data into the net and calculates the trajectory
	def query(self):
		return
		
		# Each agent query per batch dimension
		for person_it in range(0, self.n_query_agents):
			self.fillBatchOtherAgents(person_it)

		dict = {"batch_vel": self.current_velocity_,
		        "batch_pos": self.current_position_[:,0:2],
		        "batch_ped_grid": self.other_agents_info,
		        "batch_grid": self.batch_grid,
		        "step": 0
		        }
		feed_dict_ = self.model.feed_test_dic(**dict)

		outputs = self.model.predict(self.sess, feed_dict_, True)

		# y_model_pred, output_decoder, outs = outputs

		# publish the predicted trajectories
		self.global_trajectories = self.calculate_trajectories(outputs[0])

	# add the velocity predictions together to form points and convert them to global coordinate frame
	def calculate_trajectories(self, y_model_pred):
		global_trajectory = MarkerArray()
		pedestrians = lmpcc_obstacle_array()

		#time = np.zeros([self.model_args.prediction_horizon + 1])
		# Robot TRajectory to warm-start
		robot_trajectory = Path()
		robot_trajectory.header.frame_id = "odom"
		robot_trajectory.header.stamp = rospy.Time.now()

		for ped_id in range(0, y_model_pred.shape[0]):
			for mix_idx in range(self.model_args.n_mixtures):

				self.predicted_positions[ped_id,mix_idx, 0, 0] = self.current_position_[ped_id, 0]  # + y_model_pred[ped_id,0, 0] * self.model_args.dt
				self.predicted_positions[ped_id,mix_idx, 0, 1] = self.current_position_[ped_id, 1]  # + y_model_pred[ped_id,0, self.model_args.n_mixtures] * self.model_args.dt
				self.predicted_velocities[ped_id,mix_idx, 0] = self.current_velocity_[ped_id, :2]

				for pred_step in range(1, self.model_args.prediction_horizon):
					idx = (pred_step - 1) * self.model_args.output_pred_state_dim * self.model_args.n_mixtures + mix_idx
					# TODO: this can be optimized
					# TODO: fill velocity vector
					#time[pred_step + 1] = time[pred_step] + self.model_args.dt
					self.predicted_positions[ped_id, mix_idx, pred_step, 0] = self.predicted_positions[ped_id, mix_idx, pred_step - 1, 0] + \
					                                                 y_model_pred[ped_id, 0, idx] * self.model_args.dt
					self.predicted_positions[ped_id, mix_idx, pred_step, 1] = self.predicted_positions[ped_id, mix_idx, pred_step - 1, 1] + \
					                                                 y_model_pred[ped_id, 0, idx + self.model_args.n_mixtures] * self.model_args.dt
					self.predicted_velocities[ped_id, mix_idx,pred_step, 0] = y_model_pred[ped_id, 0, idx]
					self.predicted_velocities[ped_id, mix_idx,pred_step, 1] = y_model_pred[ped_id, 0, idx + self.model_args.n_mixtures]
					if self.model_args.output_pred_state_dim > 2:
						self.predicted_uncertainty[ped_id, mix_idx, pred_step, 0] = y_model_pred[ped_id, 0, idx + 2 * self.model_args.n_mixtures]
						self.predicted_uncertainty[ped_id, mix_idx, pred_step, 0] = y_model_pred[ped_id, 0, idx + 3 * self.model_args.n_mixtures]
				# up-sample trajectory to match lmpcc
				# the dt time should match the mpc horizon step
				# new_time, new_pos =  sup.smoothenTrajectory(time,positions,velocities,self.model_args,dt=0.2)
				ped = lmpcc_obstacle()

				# 15 is the number of stages of the mpcc. it should match
				sigma_x = 1
				sigma_y = 1
				for pred_step in range(self.model_args.prediction_horizon):
					marker = Marker()
					marker.header.frame_id = "map"
					marker.header.stamp = rospy.Time.now()
					marker.ns = "goal_marker"
					marker.id = pred_step + self.model_args.prediction_horizon * ped_id * mix_idx
					marker.type = 3
					marker.color.a = 1.0 / (1.0 + pred_step / 3.0)
					marker.color.r = self.colors[ped_id % 9][0]
					marker.color.g = self.colors[ped_id%9][1]
					marker.color.b = self.colors[ped_id%9][2]
					marker.scale.x = 1 * sigma_x
					marker.scale.y = 1 * sigma_y
					marker.scale.z = 0.1
					pose = Pose()

					pose.position.x = self.predicted_positions[ped_id, mix_idx, pred_step, 0]
					pose.position.y = self.predicted_positions[ped_id, mix_idx, pred_step, 1]
					pose.orientation.w = 1.0
					marker.pose = pose
					pose_stamped = PoseStamped()
					# Used for LMPCC
					pose_stamped.pose = pose
					if ped_id != -1:
						ped.trajectory.poses.append(pose_stamped)
						# TODo: Use uncertainty info
						if self.model_args.output_pred_state_dim > 2:
							sigma_x += np.square(
								self.predicted_uncertainty[ped_id, mix_idx, pred_step, 0]) * self.model_args.dt * self.model_args.dt
							sigma_y += np.square(
								self.predicted_uncertainty[ped_id, mix_idx, pred_step, 1]) * self.model_args.dt * self.model_args.dt
							ped.major_semiaxis.append(sigma_x)
							ped.minor_semiaxis.append(sigma_y)
						else:
							ped.major_semiaxis.append(1)
							ped.minor_semiaxis.append(1)
					else:
						robot_trajectory.poses.append(pose_stamped)
					global_trajectory.markers.append(marker)
				if ped_id != -1:
					ped.pose = ped.trajectory.poses[0].pose
					pedestrians.lmpcc_obstacles.append(ped)

		# self.pub_robot_traj.publish(robot_trajectory)
		self.obstacles_publisher.publish(pedestrians)
		self.pub_viz.publish(global_trajectory)

		return global_trajectory


if __name__ == '__main__':
	rospy.init_node('SocialVDGNN_node')
	prediction_network = SocialVDGNN()
	rospy.sleep(1.0)

	total_run_time = 0.0
	total_runs = 0
	while not rospy.is_shutdown():
		start_time = time.time()

		prediction_network.lock.acquire()
		prediction_network.planning = True
		prediction_network.query()
		prediction_network.planning = False
		prediction_network.lock.release()
		# planning_network.fig_animate.canvas.draw()
		# cv2.imshow("image", planning_network.grid)
		# cv2.waitKey(100)
		# wait around a bit if neccesairy
		now = time.time()
		total_run_time += now - start_time
		total_runs += 1
		if now - start_time < 0.1:  # args.dt:
			rospy.sleep(0.1 - (now - start_time))
		else:
			print("not keeping up to rate")
	print("Avg run time: {} ms".format(total_run_time / (total_runs / 1000.0)))
	prediction_network.sess.close()
