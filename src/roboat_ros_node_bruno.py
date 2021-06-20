#!/usr/bin/env python
import time
import os
import pickle as pkl
import threading as th
import importlib
import math
import numpy as np
import matplotlib.pyplot as pl

import sys; print('Python %s on %s' % (sys.version, sys.platform))
path_1 = os.path.join(os.getcwd(), "data_utils")
if os.path.exists(path_1) and os.path.isdir(path_1):
	sys.path.append(path_1)
	print('Data utils added to path')

path_2 = os.path.join(os.getcwd(), "models")
if os.path.exists(path_2) and os.path.isdir(path_2):
	sys.path.append(path_2)
	print('Models added to path')

import rospy
from geometry_msgs.msg import Pose, PoseStamped
# from pedsim_msgs.msg import TrackedPersons, TrackedPerson
from nav_msgs.msg import Path, OccupancyGrid
from visualization_msgs.msg import MarkerArray, Marker

from data_utils import DataHandlerLSTM as dhlstm

from data_utils import Support as sup
from lmpcc_msgs.msg import lmpcc_obstacle, lmpcc_obstacle_array

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
import tensorflow as tf

model_name = "IntNet"

# Import model
from IntNet import NetworkModel

class SocialVDGNN:
	def __init__(self):

		self.model_name = rospy.get_param('~model_name', 'VRNNwLikelihood')
		self.model_id = rospy.get_param('~id', '46')
		# ROS Topics
		self.other_agents_topic = rospy.get_param('~other_agents_topic',"/pedsim_visualizer/tracked_persons")
		self.robot_state_topic = rospy.get_param('~robot_state_topic', "/robot_state")
		self.grid_topic = rospy.get_param('~grid_topic',"/move_base/local_costmap/costmap")
		self.robot_plan_topic = rospy.get_param('~robot_plan_topic', "/predicted_trajectory")

		self.n_query_agents = rospy.get_param('~number_of_query_agents',9)
		self.prediction_steps = rospy.get_param('~prediction_steps', 15)
		self.robot = True
		self.n_other_agents = 8
		# Load Model Parameters
		self.load_args()

		# Robot State variables
		self.robot_state_ = Pose()
		self.robot_plan_ = Path()
		for i in range(self.prediction_steps):
			self.robot_plan_.poses.append(PoseStamped())

		# Prediction State variables
		self.current_position_ = np.zeros([self.n_query_agents ,1,(self.model_args.prev_horizon+1)*2])
		self.current_velocity_ = np.zeros([self.n_query_agents ,1,(self.model_args.prev_horizon+1)*2])
		self.predicted_positions = np.zeros([self.n_query_agents,self.model_args.prediction_horizon + 1, 2])
		self.predicted_velocities = np.zeros([self.n_query_agents, self.model_args.prediction_horizon + 1, 2])

		self.other_pedestrians = []
		for i in range(self.n_other_agents):
			self.other_pedestrians.append(TrackedPerson())

		self.other_agents_info = np.zeros([self.n_query_agents ,1,self.model_args.n_other_agents,self.model_args.pedestrian_vector_dim*self.model_args.prediction_horizon])

		self.load_model()

		self.width = int(self.model_args.submap_width / self.model_args.submap_resolution )
		self.height = int(self.model_args.submap_height / self.model_args.submap_resolution)
		self.grid = np.zeros([self.n_query_agents, 1, self.width, self.height])
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
		#self.ax_pos = pl.subplot()
		pl.show(block=False)

		self.colors = []
		self.colors.append([0.8500, 0.3250, 0.0980])  # orange
		self.colors.append([0.0, 0.4470, 0.7410])  # blue
		self.colors.append([0.4660, 0.6740, 0.1880])  # green
		self.colors.append([0.4940, 0.1840, 0.5560])  # purple
		self.colors.append([0.9290, 0.6940, 0.1250])  # yellow
		self.colors.append([0.3010, 0.7450, 0.9330])  # cyan
		self.colors.append([0.6350, 0.0780, 0.1840])  # chocolate
		self.colors.append([0.505, 0.505, 0.505])  # grey
		self.colors.append([0.8, 0.6, 1])  # pink


		# ROS Subscribers
		# rospy.Subscriber(self.other_agents_topic, TrackedPersons, self.other_agents_CB, queue_size=1)
		# rospy.Subscriber(self.grid_topic, OccupancyGrid, self.grid_CB, queue_size=1)
		# rospy.Subscriber(self.robot_state_topic, Pose, self.robot_state_CB, queue_size=1)
		# rospy.Subscriber(self.robot_plan_topic, Path, self.robot_plan_CB, queue_size=1)

		# ROS Publishers
		self.pub_viz = rospy.Publisher('social_vdgnn_predictions', MarkerArray, queue_size=10)
		self.obstacles_publisher = rospy.Publisher('ellipse_objects_feed', lmpcc_obstacle_array, queue_size=10)

		# THread control
		self.lock = th.Lock()

	def load_args(self):
		cwd = os.getcwd()

		model_path = os.path.normpath(cwd + '/../') + '/trained_models/' + self.model_name + "/" + str(self.model_id)

		print("Loading data from: '{}'".format(model_path))
		file = open(model_path + '/model_parameters.pkl', 'rb')
		if sys.version_info[0] < 3:
			model_parameters = pkl.load(file)  # ,encoding='latin1')
		else:
			model_parameters = pkl.load(file, encoding='latin1')
		file.close()
		self.model_args = model_parameters["args"]

		# change some args because we are doing inference
		self.model_args.truncated_backprop_length = 1
		self.model_args.batch_size = self.n_query_agents

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

	def robot_plan_CB(self,msg):
		self.robot_plan_ = msg

	def robot_state_CB(self, msg):
		self.robot_state_ = msg

		self.current_position_[-1, 0, 0] = self.robot_state_.position.x
		self.current_position_[-1, 0, 1] = self.robot_state_.position.y
		self.current_velocity_[-1, 0, 0] = self.robot_state_.position.z*np.cos(self.robot_state_.orientation.z)
		self.current_velocity_[-1, 0, 1] = self.robot_state_.position.z*np.sin(self.robot_state_.orientation.z)

	def grid_CB(self, data):
		# scale the grid from 0-100 to 0-1 and invert
		print("Grid data size: " + str(len(data.data)))
		self.grid[0,0,:,:] = (np.asarray(data.data).reshape((self.width, self.height)).astype(float) / 100.0)
		self.grid[0,0,:,:] = np.flip(self.grid[0,0,:,:], 1)
		self.grid[0,0,:,:] = sup.rotate_grid_around_center(self.grid[0,0,:,:], 90)

		if False:
			self.ax_pos.clear()
			sup.plot_grid(self.ax_pos, np.array([0.0, 0.0]), self.grid, self.model_args.submap_resolution,
			              np.array([self.model_args.submap_width, self.model_args.submap_height]))
			self.ax_pos.set_xlim([-self.model_args.submap_width/2,self.model_args.submap_width/2])
			self.ax_pos.set_ylim([-self.model_args.submap_height/2,self.model_args.submap_height/2])
			self.ax_pos.set_aspect('equal')

	def other_agents_CB(self, data):
		"""
		Obtain surrouding agents information callback
		:param data: TrackedPerson
		:return:
		"""
		# Assuming that the information about the agents comes always ordered
		# Shift old states to fill the vector with new info
		self.current_position_ = np.roll(self.current_position_, 2, axis=2)
		self.current_velocity_ = np.roll(self.current_velocity_, 2, axis=2)
		for person_it in range(len(data.tracks)):
			ped = TrackedPerson()
			ped.pose=data.tracks[person_it].pose
			ped.track_id = data.tracks[person_it].track_id
			ped.twist = data.tracks[person_it].twist
			self.other_pedestrians[person_it] = ped

			self.current_position_[person_it, 0, 0] = ped.pose.pose.position.x
			self.current_position_[person_it, 0, 1] = ped.pose.pose.position.y
			self.current_velocity_[person_it, 0, 0] = ped.twist.twist.linear.x
			self.current_velocity_[person_it, 0, 1] = ped.twist.twist.linear.y

	def fillBatchOtherAgents(self,person_it):
		n_other_agents = self.n_other_agents

		other_poses_ordered = np.zeros((n_other_agents+1, 6))
		if person_it != -1:
			current_pos = np.array(
				[self.other_pedestrians[person_it].pose.pose.position.x, self.other_pedestrians[person_it].pose.pose.position.y])
			current_vel = np.array(
				[self.other_pedestrians[person_it].twist.twist.linear.x, self.other_pedestrians[person_it].twist.twist.linear.y])

			ag_id = 0
			for k in range(n_other_agents):
				if self.other_pedestrians[k].track_id != person_it:
					other_poses_ordered[ag_id, 0] = self.other_pedestrians[k].pose.pose.position.x
					other_poses_ordered[ag_id, 1] = self.other_pedestrians[k].pose.pose.position.y
					other_poses_ordered[ag_id, 2] = self.other_pedestrians[k].twist.twist.linear.x
					other_poses_ordered[ag_id, 3] = self.other_pedestrians[k].twist.twist.linear.y
					other_poses_ordered[ag_id, 4] = np.linalg.norm(other_poses_ordered[ag_id, :2] - current_pos)
					other_poses_ordered[ag_id, 5] = self.other_pedestrians[k].track_id
					ag_id += 1

			# Adding robot
			if self.robot:
				other_poses_ordered[-1, 0] = self.robot_state_.position.x
				other_poses_ordered[-1, 1] = self.robot_state_.position.y
				other_poses_ordered[-1, 2] = self.robot_state_.position.z*np.cos(self.robot_state_.orientation.z)
				other_poses_ordered[-1, 3] = self.robot_state_.position.z*np.sin(self.robot_state_.orientation.z)
				other_poses_ordered[-1, 4] = np.linalg.norm(other_poses_ordered[-1, :2] - current_pos)
				other_poses_ordered[-1, 5] = -1

			other_poses_ordered = other_poses_ordered[other_poses_ordered[:, 4].argsort()]
			if self.model_args.others_info == "sequence":
				for pred_step in range(self.model_args.prediction_horizon):
					current_pos = self.predicted_positions[person_it,pred_step+1]
					current_vel = self.predicted_velocities[person_it, pred_step+1]
					ag_id = 0
					for k in range(min([n_other_agents, self.model_args.n_other_agents])):
						# surrouding pedestrians info
						if other_poses_ordered[ag_id, 5] != -1:
							# cv assuption
							#next_pose = other_poses_ordered[ag_id, :2] + self.model_args.dt * other_poses_ordered[ag_id, 2:4] * (
							#		pred_step + 1)
							# relative_velocity = other_poses_ordered[ag_id, 2:4] - current_vel
							next_pose = self.predicted_positions[ag_id, pred_step + 1]
							relative_pose = next_pose - current_pos
							relative_velocity = self.predicted_velocities[ag_id, pred_step+1] - current_vel
							self.other_agents_info[person_it-1, 0, ag_id,
							self.model_args.pedestrian_vector_dim * pred_step:self.model_args.pedestrian_vector_dim * pred_step + 2] = \
								relative_pose
							self.other_agents_info[person_it-1, 0, ag_id,
							self.model_args.pedestrian_vector_dim * pred_step + 2:self.model_args.pedestrian_vector_dim * pred_step + 4] = \
								relative_velocity
							self.other_agents_info[person_it - 1, 0, ag_id, self.model_args.pedestrian_vector_dim * pred_step + 4] = \
								np.arctan2(relative_pose[1], relative_pose[0])
							self.other_agents_info[person_it - 1, 0, ag_id, self.model_args.pedestrian_vector_dim * pred_step + 5] = \
								np.linalg.norm(relative_pose)
						elif other_poses_ordered[ag_id, 5] == -1:
							# Robot info
							next_pose = np.array([self.robot_plan_.poses[pred_step].pose.position.x,self.robot_plan_.poses[pred_step].pose.position.y])
							relative_pose = next_pose - current_pos
							robot_velocity = np.array([self.robot_plan_.poses[pred_step].pose.orientation.x*np.cos(self.robot_plan_.poses[pred_step].pose.position.z),
							                           self.robot_plan_.poses[pred_step].pose.orientation.x*np.sin(self.robot_plan_.poses[pred_step].pose.position.z)])
							relative_velocity = robot_velocity - current_vel
							self.other_agents_info[person_it-1, 0, ag_id,
							self.model_args.pedestrian_vector_dim * pred_step:self.model_args.pedestrian_vector_dim * pred_step + 2] = \
								relative_pose
							self.other_agents_info[person_it-1, 0, ag_id,
							self.model_args.pedestrian_vector_dim * pred_step + 2:self.model_args.pedestrian_vector_dim * pred_step + 4] = \
								relative_velocity
							self.other_agents_info[person_it-1, 0, ag_id,self.model_args.pedestrian_vector_dim * pred_step + 4] = \
								np.arctan2(relative_pose[1],relative_pose[0])
							self.other_agents_info[person_it-1, 0, ag_id,self.model_args.pedestrian_vector_dim * pred_step + 5] = \
								np.linalg.norm(relative_pose)
						ag_id += 1
					for j in range(min([n_other_agents, self.model_args.n_other_agents]), self.model_args.n_other_agents):
						self.other_agents_info[person_it-1, 0, j, self.pedestrian_vector_dim * pred_step:self.pedestrian_vector_dim * (pred_step+1)] =\
							np.zeros([self.pedestrian_vector_dim])
		else:
			current_pos = np.array(
				[self.robot_state_.position.x,
				 self.robot_state_.position.y])
			current_vel = np.array(
				[self.robot_state_.position.z*np.cos(self.robot_state_.orientation.z),
				 self.robot_state_.position.z*np.sin(self.robot_state_.orientation.z)])

			for ag_id in range(n_other_agents):
				other_poses_ordered[ag_id, 0] = self.other_pedestrians[ag_id].pose.pose.position.x
				other_poses_ordered[ag_id, 1] = self.other_pedestrians[ag_id].pose.pose.position.y
				other_poses_ordered[ag_id, 2] = self.other_pedestrians[ag_id].twist.twist.linear.x
				other_poses_ordered[ag_id, 3] = self.other_pedestrians[ag_id].twist.twist.linear.y
				other_poses_ordered[ag_id, 4] = np.linalg.norm(other_poses_ordered[ag_id, :2] - current_pos)
				other_poses_ordered[ag_id, 5] = self.other_pedestrians[ag_id].track_id

			other_poses_ordered = other_poses_ordered[other_poses_ordered[:, 4].argsort()]
			if self.model_args.others_info == "sequence":
				for pred_step in range(self.model_args.prediction_horizon):
					current_pos = self.predicted_positions[person_it, pred_step + 1]
					current_vel = self.predicted_velocities[person_it, pred_step + 1]
					for ag_id in range(min([n_other_agents, self.model_args.n_other_agents])):
						next_pose = other_poses_ordered[ag_id, :2] + self.model_args.dt * other_poses_ordered[ag_id, 2:4] * (
									pred_step + 1)
						relative_pose = next_pose - current_pos
						relative_velocity = other_poses_ordered[ag_id, 2:4] - current_vel
						self.other_agents_info[person_it - 1, 0, ag_id,
						self.model_args.pedestrian_vector_dim * pred_step:self.model_args.pedestrian_vector_dim * pred_step + 2] = \
								relative_pose
						self.other_agents_info[person_it - 1, 0, ag_id,
						self.model_args.pedestrian_vector_dim * pred_step + 2:self.model_args.pedestrian_vector_dim * pred_step + 4] = \
								relative_velocity
						self.other_agents_info[person_it - 1, 0, ag_id, self.model_args.pedestrian_vector_dim * pred_step + 4] = \
								np.arctan2(relative_pose[1], relative_pose[0])
						self.other_agents_info[person_it - 1, 0, ag_id, self.model_args.pedestrian_vector_dim * pred_step + 5] = \
								np.linalg.norm(relative_pose)

					for j in range(min([n_other_agents, self.model_args.n_other_agents]), self.model_args.n_other_agents):
						self.other_agents_info[person_it - 1, 0, j,
						self.pedestrian_vector_dim * pred_step:self.pedestrian_vector_dim * (pred_step + 1)] = \
							np.zeros([self.pedestrian_vector_dim])

	# query feed the data into the net and calculates the trajectory
	def query(self):
		# Each agent query per batch dimension
		if self.robot:
			for person_it in range(-1,len(self.other_pedestrians),1):
				self.fillBatchOtherAgents(self.other_pedestrians[person_it].track_id)
		else:
			for person_it in range(len(self.other_pedestrians)):
				self.fillBatchOtherAgents(self.other_pedestrians[person_it].track_id)
		feed_dict_ = self.model.feed_test_dic(self.current_velocity_, self.grid, self.other_agents_info, 0)

		outputs = self.model.predict(self.sess, feed_dict_, True)

		#y_model_pred, output_decoder, outs = outputs

		# publish the predicted trajectories
		self.global_trajectories = self.calculate_trajectories(outputs[0])

	# add the velocity predictions together to form points and convert them to global coordinate frame
	def calculate_trajectories(self, y_model_pred):

		global_trajectory = MarkerArray()
		pedestrians = lmpcc_obstacle_array()

		time = np.zeros([self.model_args.prediction_horizon + 1])
		for ped_id in range(-1,y_model_pred.shape[0]-1,1):
			mix_idx = 0
			if ped_id != -1:
				self.predicted_positions[ped_id,0] = self.current_position_[ped_id,0,:2]
				self.predicted_velocities[ped_id, 0] = self.current_velocity_[ped_id,0,:2]
			else:
				self.predicted_positions[ped_id,0] = np.array([self.robot_state_.position.x,self.robot_state_.position.y])
				self.predicted_velocities[ped_id, 0] = np.array([self.robot_state_.position.z*np.cos(self.robot_state_.orientation.z), \
				                                                 self.robot_state_.position.z*np.sin(self.robot_state_.orientation.z)])

			for pred_step in range(self.model_args.prediction_horizon):
				idx = pred_step * self.model_args.output_pred_state_dim * self.model_args.n_mixtures + mix_idx
				# TODO: this can be optimized
				# TODO: fill velocity vector
				time[pred_step +1] = time[pred_step] + self.model_args.dt
				self.predicted_positions[ped_id,pred_step+1,0] = self.predicted_positions[ped_id,pred_step,0] + y_model_pred[ped_id,0, idx] * self.model_args.dt
				self.predicted_positions[ped_id,pred_step+1,1] = self.predicted_positions[ped_id,pred_step,1] + y_model_pred[ped_id,0, idx + self.model_args.n_mixtures] * self.model_args.dt
				self.predicted_velocities[ped_id,pred_step+1,0] =  y_model_pred[ped_id,0, idx]
				self.predicted_velocities[ped_id, pred_step + 1, 0] = y_model_pred[ped_id,0, idx + self.model_args.n_mixtures]
			# up-sample trajectory to match lmpcc
			# the dt time should match the mpc horizon step
			#new_time, new_pos =  sup.smoothenTrajectory(time,positions,velocities,self.model_args,dt=0.2)
			ped = lmpcc_obstacle()

			# 15 is the number of stages of the mpcc. it should match
			for pred_step in range(15):
				marker = Marker()
				marker.header.frame_id = "odom"
				marker.header.stamp = rospy.Time.now()
				marker.ns = "goal_marker"
				marker.id = pred_step + self.model_args.prediction_horizon*ped_id
				marker.type = 3
				marker.color.a = 1.0/(1.0+pred_step/3.0)
				marker.color.r = self.colors[ped_id][0]
				marker.color.g = self.colors[ped_id][1]
				marker.color.b = self.colors[ped_id][2]
				marker.scale.x = 0.3*2.0
				marker.scale.y = 0.3*2.0
				marker.scale.z = 0.1
				pose = Pose()

				pose.position.x = self.predicted_positions[ped_id,pred_step,0]
				pose.position.y = self.predicted_positions[ped_id,pred_step,1]
				pose.orientation.w = 1.0
				marker.pose = pose
				pose_stamped = PoseStamped()
				# Used for LMPCC
				pose_stamped.pose = pose
				if ped_id != -1:
					ped.trajectory.poses.append(pose_stamped)
					# TODo: Use uncertainty info
					ped.major_semiaxis.append(0.3)
					ped.minor_semiaxis.append(0.3)
					ped.pose = pose_stamped.pose
				global_trajectory.markers.append(marker)
			if ped_id != -1:
				pedestrians.lmpcc_obstacles.append(ped)

		self.obstacles_publisher.publish(pedestrians)
		self.pub_viz.publish(global_trajectory)

		return global_trajectory

if __name__ == '__main__':
	print("Imports succeeded yess!")
	sys.exit()
	rospy.init_node('SocialVDGNN_node')
	prediction_network = SocialVDGNN()
	rospy.sleep(5.0)
	while not rospy.is_shutdown():
		start_time = time.time()

		prediction_network.lock.acquire()
		prediction_network.query()
		prediction_network.lock.release()
		#planning_network.fig_animate.canvas.draw()
		#cv2.imshow("image", planning_network.grid)
		#cv2.waitKey(100)
		# wait around a bit if neccesairy
		now = time.time()
		if now - start_time < 0.05:  # args.dt:
			rospy.sleep(0.05 - (now - start_time))
		else:
			rospy.loginfo("not keeping up to rate")
	prediction_network.sess.close()
