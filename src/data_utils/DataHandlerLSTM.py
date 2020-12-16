import numpy as np
import os
import random
import sys
import math
import cv2
import pickle as pkl
from pykalman import KalmanFilter
if sys.version_info[0] < 3:
	import Support as sup
	from Trajectory import *
	import AgentContainer as ped_cont
else:
	import src.data_utils.Support as sup
	from src.data_utils.Trajectory import *
	import src.data_utils.AgentContainer as ped_cont
from copy import deepcopy
import matplotlib.pyplot as pl
import matplotlib.animation as animation
from time import sleep
import random
from scipy.stats import multivariate_normal

from matplotlib.patches import Ellipse

class DataHandlerLSTM():
	"""
	Data handler for training an LSTM pedestrian prediction model
	"""
	def __init__(self, args):

		self.data_path = args.data_path
		self.batch_size = args.batch_size
		self.tbpl = args.truncated_backprop_length
		self.prev_horizon = args.prev_horizon
		self.output_sequence_length = args.prediction_horizon
		self.prediction_horizon = args.prediction_horizon
		self.input_dim = args.input_dim
		self.input_state_dim = args.input_state_dim
		self.output_state_dim = args.output_dim
		self.submap_width = args.submap_width
		self.submap_height = args.submap_height
		self.submap_resolution = args.submap_resolution
		self.centered_grid = args.centered_grid
		self.rotated_grid = args.rotated_grid
		self.multi_pedestrian = True
		self.pedestrian_radius = args.pedestrian_radius
		self.min_length_trajectory = self.tbpl + 1 + self.output_sequence_length + args.prev_horizon
		self.pedestrian_vector_dim = args.pedestrian_vector_dim
		self.max_range_ped_grid = args.max_range_ped_grid
		self.dt = args.dt
		self.n_mixture = args.n_mixtures
		self.output_pred_state_dim = args.output_pred_state_dim
		self.args = args
		self.real_world_data = args.real_world_data
		# Normalization constants
		self.norm_const_x = 1.0
		self.norm_const_y = 1.0
		self.norm_const_heading = 1.0
		self.norm_const_vx = 1.0
		self.norm_const_vy = 1.0
		self.norm_const_omega = 1.0
		self.min_pos_x = 1000
		self.min_pos_y = 1000
		self.max_pos_x = -1000
		self.max_pos_y = -1000
		self.min_vel_x = 1000
		self.min_vel_y = 1000
		self.max_vel_x = -1000
		self.max_vel_y = -1000
		self.train_set = args.train_set
		self.avg_traj_length = 0

		# Default Distribution Parameters
		self.amplitude_ = 1.0
		self.factor_ = 5.0
		self.cutoff_ = 1.0
		self.covar_ = 0.25
		self.var = multivariate_normal(mean=0, cov=1)

		# Data structure containing all the information about agents
		self.agent_container = ped_cont.AgentContainer()
		self.test_container = ped_cont.AgentContainer()

		self.agent_traj_idx = None  # indicates which trajectory index will be the next one per agent
		self.trajectory_set = []
		self.test_trajectory_set = []
		self.data_idx = 0
		self.val_data_idx = 0

		# Patch for updating the grid around a pedestrian position
		self.pedestrian_patch = None

		# Training variables
		self.sequence_reset = np.ones([self.batch_size])  # indicates whether sequences are reset (hidden state of LSTMs needs to be reset accordingly)
		self.sequence_idx = np.zeros([self.batch_size])+self.args.prev_horizon

		# Batch with info about other agents
		if "future" in self.args.others_info:
			self.batch_vel = np.zeros(
				[self.batch_size, self.tbpl, self.input_state_dim * self.prediction_horizon])  # data fed for training
		else:
			self.batch_vel = np.zeros([self.batch_size, self.tbpl, self.input_state_dim * (self.prev_horizon + 1)])
		self.batch_x = np.zeros([self.batch_size, self.tbpl, self.input_dim*(self.prev_horizon+1)])  # data fed for training
		self.batch_grid = np.zeros([self.batch_size, self.tbpl,
																int(np.ceil(self.submap_width / self.submap_resolution)),
																int(np.ceil(self.submap_height / self.submap_resolution))])
		if self.args.others_info == "relative":
			self.pedestrian_grid = np.zeros([self.batch_size, self.tbpl, self.pedestrian_vector_dim*self.args.n_other_agents])
			self.val_pedestrian_grid = np.zeros([self.batch_size, self.tbpl, self.pedestrian_vector_dim*self.args.n_other_agents])
		elif "sequence" in self.args.others_info:
			self.pedestrian_grid = np.zeros(
				[self.batch_size, self.tbpl, self.args.n_other_agents, self.pedestrian_vector_dim*self.prediction_horizon])
		elif self.args.others_info == "prev_sequence":
			self.pedestrian_grid = np.zeros(
				[self.batch_size, self.tbpl, self.args.n_other_agents, self.pedestrian_vector_dim*(self.prev_horizon + 1)])
		elif self.args.others_info == "sequence2":
			self.pedestrian_grid = np.zeros(
				[self.batch_size, self.tbpl, self.args.n_other_agents, self.prediction_horizon,self.pedestrian_vector_dim])
		elif self.args.others_info == "ped_grid":
			self.pedestrian_grid = np.zeros([self.batch_size, self.tbpl,
			                            int(np.ceil(self.submap_width / self.submap_resolution)),
			                            int(np.ceil(self.submap_height / self.submap_resolution))])
		else:
			self.pedestrian_grid = np.zeros([self.batch_size, self.tbpl, self.pedestrian_vector_dim])
			self.val_pedestrian_grid = np.zeros([self.batch_size, self.tbpl, self.pedestrian_vector_dim])
		self.batch_goal = np.zeros([self.batch_size, self.tbpl, 2])
		self.batch_y = np.zeros([self.batch_size, self.tbpl, self.output_state_dim * self.output_sequence_length])
		self.batch_sequences = []  # Sequence stack (tuples of trajectory + grid) where parts of length self.tbpl are taken from until a sequence is finished. As soon as one is finished, a new sequence is put in the stack.
		self.batch_ids = []
		self.batch_pos = np.zeros([self.batch_size, self.tbpl, self.input_state_dim * (self.prev_horizon + 1)])
		self.batch_pos_target = np.zeros([self.batch_size, self.tbpl, 2 * self.output_sequence_length])

		# Validation variables
		self.val_sequence_reset = np.ones([self.batch_size])  # indicates whether sequences are reset (hidden state of LSTMs needs to be reset accordingly)
		self.val_sequence_idx = np.zeros([self.batch_size]) + self.args.prev_horizon
		# Batch with current query-agent velocities
		self.val_batch_vel = np.zeros([self.batch_size, self.tbpl, self.input_state_dim * (self.prev_horizon + 1)])
		# Batch with query-agent current velocities and positions
		self.val_batch_x = np.zeros([self.batch_size, self.tbpl, self.input_dim * (self.prev_horizon + 1)])  # data fed for training
		# Query agent static environment information
		self.val_batch_grid = np.zeros([self.batch_size, self.tbpl,
		                            int(np.ceil(self.submap_width / self.submap_resolution)),
		                            int(np.ceil(self.submap_height / self.submap_resolution))])

		# Batch of query-agent goal positions
		self.val_batch_goal = np.zeros([self.batch_size, self.tbpl, 2])
		# Batch with quer-agent next velocities
		self.val_batch_y = np.zeros([self.batch_size, self.tbpl, self.output_state_dim * self.output_sequence_length])
		# Sequence stack (tuples of trajectory + grid) where parts of length self.tbpl are taken from until a sequence is finished. As soon as one is finished, a new sequence is put in the stack.
		self.val_batch_sequences = []
		self.val_batch_ids = []
		# Batch of query-agent current position
		self.val_batch_pos = np.zeros([self.batch_size, self.tbpl, self.input_state_dim * (self.prev_horizon + 1)])
		# Batch with quer-agent next positions
		self.val_batch_pos_target = np.zeros([self.batch_size, self.tbpl, 2 * self.output_sequence_length])

	def addAgentTrajectoriesToSet(self,agent_container,trajectory_set, id):
		"""
		Goes through all trajectories of agent and adds them to the member set if they fulfill the criteria.
		For all the time steps within the trajectory it also computes the positions of the other agents at that
		timestep in order to make training more efficient.
		"""
		for traj_idx, traj in enumerate(agent_container.getAgentTrajectories(id)):
			traj_with_collision = False
			if len(traj) > self.min_length_trajectory:
				#if traj.getMinTime() < 100:
				traj.updateInterpolators()
				# Find other agent's trajectories which overlap with each time step
				for time_idx in range(traj.time_vec.shape[0]):
					if self.multi_pedestrian:
						query_time = traj.time_vec[time_idx]
						other_agents_positions = agent_container.getAgentPositionsForTimeExclude(query_time, id)
						other_agents_velocities = agent_container.getAgentVelocitiesForTimeExclude(query_time, id)
						"""
						for ag_id in range(other_agents_positions.shape[0]):
							if np.linalg.norm(traj.pose_vec[time_idx,:2]-other_agents_positions[ag_id]) < 0.6:
								print("Trajectory discarded. Collision between the agents")
								traj_with_collision = True
						"""
						# Remove ego agent
						traj.other_agents_positions.append(other_agents_positions)
						traj.other_agents_velocities.append(other_agents_velocities)
				# TODO: MAYBE REMOVE IN THE FUTURE
				#if id != -1:
					#if not traj_with_collision:
				trajectory_set.append((id, traj))

	def processData(self, **kwargs):
		"""
		Processes the simulation or real-world data, depending on the usage.
		"""
		data_pickle = self.args.data_path + self.args.scenario + "/data" + str(self.args.prediction_horizon) + "_" + str(
			self.args.truncated_backprop_length)+ "_" + str(
			self.args.prev_horizon) + ".pickle"
		if os.path.isfile(data_pickle):
			self.loadTrajectoryData(data_pickle)
		else:
			if "real_world" in data_pickle:
				print("Processing real-world data.")
				self._process_real_data_()
			elif "simulation" in data_pickle:
				print("Processing simulation data.")
				self._process_simulation_data_(**kwargs)
			else:
				print("Processing gym data.")
				self._process_gym_data_(**kwargs)

			self.saveTrajectoryData(data_pickle)

	def _process_gym_data_(self, **kwargs):
		"""
		Process data generated with gym-collision-avoidance simulator
		"""
		print("Loading data from: '{}'".format(self.args.data_path + self.args.dataset))

		self.load_map(**kwargs)

		self.file = open(self.args.data_path + self.args.dataset, 'rb')
		if sys.version_info[0] < 3:
			tmp_self = pkl.load(self.file)
		else:
			tmp_self = pkl.load(self.file, encoding='latin1')

		# Iterate through the data and fill the register
		timestamp = 0.0
		for traj_id in range(len(tmp_self)):
			traj = tmp_self[traj_id]
			for t_id in range(len(traj)):
				id = 0 + traj_id
				# TODO: GET TIME VECTOR FROM SIMULATOR
				timestamp = traj[t_id]["time"]
				pose = np.zeros([1, 3])
				vel = np.zeros([1, 3])
				pose[:, 0:2] = traj[t_id]["pedestrian_state"]["position"]
				vel[:, 0:2] = traj[t_id]["pedestrian_state"]["velocity"]
				goal = traj[t_id]["pedestrian_goal_position"]

				self.agent_container.addDataSample(id, timestamp, pose, vel, goal)

		# Set the initial indices for agent trajectories (which trajectory will be returned when queried)
		self.agent_traj_idx = [0] * self.agent_container.getNumberOfAgents()

		# Subsample trajectories (longer discretization time) from dt=0.1 to dt=0.3
		for id in self.agent_container.getAgentIDs():
			for traj in self.agent_container.getAgentTrajectories(id):
				traj.subsample(int(self.args.dt * 10))

		# Put all the trajectories in the trajectory set and randomize
		for id in self.agent_container.getAgentIDs():
			print("Processing agent {} / {}".format(id, self.agent_container.getNumberOfAgents()))
			# Adds trajectory if bigger than a minimum length and maximum size
			self.addAgentTrajectoriesToSet(self.agent_container, self.trajectory_set, id)

		self.compute_min_max_values()

	def load_map(self,**kwargs):
		print("Extracting the occupancy grid ...")
		# Occupancy grid data

		if not os.path.isfile(self.data_path+self.args.scenario + '/map.npy'):
			print("Creating map from png ...")
			sup.create_map_from_png(data_path=self.args.data_path+self.args.scenario,**kwargs)

		map_data = np.load(os.path.join(self.data_path+self.args.scenario, 'map.npy'), encoding='latin1', allow_pickle=True)
		self.agent_container.occupancy_grid.gridmap = map_data.item(0)['Map']  # occupancy values of cells
		self.agent_container.occupancy_grid.resolution = map_data.item(0)['Resolution']  # map resolution in [m / cell]
		self.agent_container.occupancy_grid.map_size = map_data.item(0)['Size']  # map size in [m]
		self.agent_container.occupancy_grid.center = self.agent_container.occupancy_grid.map_size / 2.0

		#FIlter all -1 values present on MAP to zero.
		for ii in range(self.agent_container.occupancy_grid.gridmap.shape[0]):
			for jj in range(self.agent_container.occupancy_grid.gridmap.shape[1]):
				if self.agent_container.occupancy_grid.gridmap[ii, jj] == -1:
					self.agent_container.occupancy_grid.gridmap[ii, jj] = 0.0

	def _process_simulation_data_(self, **kwargs):
		"""
		Import the data from the log file stored in the directory of data_path.
		This method brings all the data into a suitable format for training.
		"""
		self.load_map(**kwargs)
		# Pedestrian data
		# [id, timestep (s), timestep (ns), pos x, pos y, yaw, vel x, vel y, omega, goal x, goal y]
		pedestrian_data = np.genfromtxt(os.path.join(self.data_path+self.args.scenario, 'total_log.csv'), delimiter=",")[1:, :]

		# Iterate through the data and fill the register
		for sample_idx in range(pedestrian_data.shape[0]):
			if pedestrian_data[sample_idx, 0] != -1:
				id = pedestrian_data[sample_idx, 0]
				timestamp = np.round(pedestrian_data[sample_idx, 1],1)# + pedestrian_data[sample_idx, 2] * 1e-9  # time in seconds
				pose = np.zeros([1,3])
				vel = np.zeros([1,3])
				pose[:,0:2] = np.true_divide(pedestrian_data[sample_idx, 3:5], np.array([self.norm_const_x, self.norm_const_y]))
				vel[:,0:2] = np.true_divide(pedestrian_data[sample_idx, 5:7], np.array([self.norm_const_vx, self.norm_const_vy]))
				goal = np.true_divide(pedestrian_data[sample_idx, 7:], np.array([self.norm_const_x, self.norm_const_y]))

				self.agent_container.addDataSample(id, timestamp, pose, vel, goal)

		# Set the initial indices for agent trajectories (which trajectory will be returned when queried)
		self.agent_traj_idx = [0] * self.agent_container.getNumberOfAgents()

#     for id in self.agent_container.getAgentIDs():
#       for traj in self.agent_container.getAgentTrajectories(id):
#         if len(traj) > self.min_length_trajectory:
#           traj.smoothenTrajectory(dt=self.dt)

		# Subsample trajectories (longer discretization time) from dt=0.1 to dt=0.3
		for id in self.agent_container.getAgentIDs():
			for traj in self.agent_container.getAgentTrajectories(id):
				traj.subsample(int(self.args.dt*10))

		# Reconstruct interpolators since they were not pickled with the rest of the trajectory
		for id in self.agent_container.getAgentIDs():
			for traj_idx, traj in enumerate(self.agent_container.getAgentTrajectories(id)):
				if len(traj) > self.min_length_trajectory:
					traj.updateInterpolators()

		# Put all the trajectories in the trajectory set and randomize
		for id in self.agent_container.getAgentIDs():
			print("Processing agent {} / {}".format(id, self.agent_container.getNumberOfAgents()))
			# Adds trajectory if bigger than a minimum length and maximum size
			self.addAgentTrajectoriesToSet(self.agent_container,self.trajectory_set,id)

		self.compute_min_max_values()

	def compute_min_max_values(self):
		for traj_id in range(len(self.trajectory_set)):
			for t_id in range(1, self.trajectory_set[traj_id][1].pose_vec.shape[0]):
				self.trajectory_set[traj_id][1].vel_vec[t_id, :2] = (self.trajectory_set[traj_id][1].pose_vec[t_id, :2] -
				                                                     self.trajectory_set[traj_id][1].pose_vec[t_id - 1,
				                                                     :2]) / self.dt
				self.min_pos_x = min(self.min_pos_x,self.trajectory_set[traj_id][1].pose_vec[t_id,0])
				self.min_pos_y = min(self.min_pos_y, self.trajectory_set[traj_id][1].pose_vec[t_id, 1])
				self.max_pos_x = max(self.max_pos_x, self.trajectory_set[traj_id][1].pose_vec[t_id, 0])
				self.max_pos_y = max(self.max_pos_y, self.trajectory_set[traj_id][1].pose_vec[t_id, 1])
				self.min_vel_x = min(self.min_vel_x,self.trajectory_set[traj_id][1].vel_vec[t_id,0])
				self.min_vel_y = min(self.min_vel_y, self.trajectory_set[traj_id][1].vel_vec[t_id, 1])
				self.max_vel_x = max(self.max_vel_x, self.trajectory_set[traj_id][1].vel_vec[t_id, 0])
				self.max_vel_y = max(self.max_vel_y, self.trajectory_set[traj_id][1].vel_vec[t_id, 1])

		self.calc_scale()

		self.args.min_pos_x = self.min_pos_x
		self.args.min_pos_y = self.min_pos_y
		self.args.max_pos_x = self.max_pos_x
		self.args.max_pos_y = self.max_pos_y
		self.args.min_vel_x = self.min_vel_x
		self.args.min_vel_y = self.min_vel_y
		self.args.max_vel_x = self.max_vel_x
		self.args.max_vel_y = self.max_vel_y
		self.args.sx_vel = self.sx_vel
		self.args.sy_vel = self.sy_vel
		self.args.sx_pos = self.sx_pos
		self.args.sy_pos = self.sy_pos

	def unit_test_data_(self,map_args):
			"""
			Generates predefined trajectories to analyse the interaction
			"""
			if not os.path.isfile(self.data_path + self.args.scenario + '/unit_map.npy'):
				print("Creating unit map from png ...")
				sup.create_map_from_png(data_path=self.args.data_path + self.args.scenario, **map_args)

			map_data = np.load(os.path.join(self.data_path + self.args.scenario, 'map.npy'), encoding='latin1',
			                   allow_pickle=True)
			self.agent_container.occupancy_grid.gridmap = map_data.item(0)['Map']  # occupancy values of cells
			self.agent_container.occupancy_grid.resolution = map_data.item(0)['Resolution']  # map resolution in [m / cell]
			self.agent_container.occupancy_grid.map_size = map_data.item(0)['Size']  # map size in [m]
			self.agent_container.occupancy_grid.center = self.agent_container.occupancy_grid.map_size / 2.0

			# FIlter all -1 values present on MAP to zero.
			for ii in range(self.agent_container.occupancy_grid.gridmap.shape[0]):
				for jj in range(self.agent_container.occupancy_grid.gridmap.shape[1]):
					if self.agent_container.occupancy_grid.gridmap[ii, jj] == -1:
						self.agent_container.occupancy_grid.gridmap[ii, jj] = 0.0

			# Iterate through the data and fill the register
			pose_agent_1 = np.array([[-15.0,0.0,0.0]])
			pose_agent_2 = np.array([[-13.0, 0.0,0.0]])
			vel = np.zeros([1, 3])
			v0 = 1.0
			goal = np.array([15, 0])
			timestamp = 0
			dt =0.4
			for sample_idx in range(int(320/dt/10)):
				id = 0
				timestamp += dt #self.args.dt
				pose_agent_1[:, 0] += v0 * dt
				vel[:, 0] = v0

				self.test_container.addDataSample(id, timestamp, pose_agent_1, vel, goal)

				id = 1
				pose_agent_2[:, 0] += v0 * dt

				self.test_container.addDataSample(id, timestamp, pose_agent_2, vel, goal)

			# Second test
			pose_agent_1 = np.array([[-15.0, 0.0, 0.0]])
			for sample_idx in range(int(320/dt/10)):
				id = 3
				timestamp += dt
				pose_agent_1[:, 0] += v0 * dt
				vel[:, 0] = v0

				self.test_container.addDataSample(id, timestamp, pose_agent_1, vel, goal)

			# Third test
			pose_agent_1 = np.array([[-15.0, 0.0, 0.0]])
			pose_agent_2 = np.array([[15.0, 0.0, 0.0]])
			for sample_idx in range(int(320/dt/10)):
				id = 4
				timestamp += (sample_idx + 1) * dt
				pose_agent_1[:, 0] += v0 * dt
				vel[:, 0] = v0
				goal = np.array([15,0])

				self.test_container.addDataSample(id, timestamp, pose_agent_1, vel, goal)

				id = 5
				pose_agent_2[:, 0] -= v0 * dt
				vel[:, 0] = -v0

				self.test_container.addDataSample(id, timestamp, pose_agent_2, vel, -goal)

			# Forth test
			pose_agent_1 = np.array([[15.0, 0.0, 0.0]])
			for sample_idx in range(int(320/dt/10)):
				id = 6
				timestamp += (sample_idx + 1) * dt
				pose_agent_1[:, 0] -= v0 * dt
				vel[:, 1] = 0
				vel[:, 0] = -v0
				goal = np.array([-15,0])

				self.test_container.addDataSample(id, timestamp, pose_agent_1, vel, goal)

			# Fith test
			pose_agent_1 = np.array([[-15.0, 0.0, 0.0]])
			pose_agent_2 = np.array([[0.0, -1.0, 0.0]])
			for sample_idx in range(int(320/dt/10)):
				id = 7
				timestamp += (sample_idx + 1) * dt
				pose_agent_1[:, 0] += v0 * dt
				vel[:, 1] = 0
				vel[:, 0] = v0
				goal = np.array([15, 0])

				self.test_container.addDataSample(id, timestamp, pose_agent_1, vel, goal)

				id = 8
				vel[:, 0] = 0
				vel[:, 1] = 0
				goal = np.array([0, -1])

				self.test_container.addDataSample(id, timestamp, pose_agent_2, vel, goal)
			# Forth test
			pose_agent_1 = np.array([[-15.0, 0.0, 0.0]])
			pose_agent_2 = np.array([[0.0, 1.0, 0.0]])
			for sample_idx in range(int(320/dt/10)):
				id = 9
				timestamp += (sample_idx + 1) * dt
				pose_agent_1[:, 0] += v0 * dt
				vel[:, 1] = 0
				vel[:, 0] = v0
				goal = np.array([15, 0])

				self.test_container.addDataSample(id, timestamp, pose_agent_1, vel, goal)

				id = 10
				vel[:, 0] = 0
				vel[:, 1] = 0
				goal = np.array([0, -1])

				self.test_container.addDataSample(id, timestamp, pose_agent_2, vel, goal)
			# Forth test
			pose_agent_1 = np.array([[-15.0, 0.0, 0.0]])
			pose_agent_2 = np.array([[0.0, 0.0, 0.0]])
			for sample_idx in range(int(320/dt/10)):
				id = 11
				timestamp += (sample_idx + 1) * dt
				pose_agent_1[:, 0] += v0 * dt
				vel[:, 1] = 0
				vel[:, 0] = v0
				goal = np.array([15, 0])

				self.test_container.addDataSample(id, timestamp, pose_agent_1, vel, goal)

				id = 12
				vel[:, 0] = 0
				vel[:, 1] = 0
				goal = np.array([0, 0])

				self.test_container.addDataSample(id, timestamp, pose_agent_2, vel, goal)

			# Put all the trajectories in the trajectory set and randomize
			pool_list = []
			for id in self.test_container.getAgentIDs():
				print("Processing agent {} / {}".format(id, self.test_container.getNumberOfAgents()))
				# Adds trajectory if bigger than a minimum length and maximum size
				self.addAgentTrajectoriesToSet(self.test_container,self.test_trajectory_set,id)

			self.min_pos_x = -1.0
			self.min_pos_y = -1.0
			self.max_pos_x = 15.0
			self.max_pos_y = 15.0
			self.min_vel_x = -1.0
			self.min_vel_y = -1.0
			self.max_vel_x = 1.0
			self.max_vel_y = 1.0

			self.calc_scale()

	def cv_to_norm_px(self, coordinate, img_shape):
		return np.array([[coordinate[1, 0]], [img_shape[0] - coordinate[0, 0]]])

	def add_z_axis(self, coordinate):
		return np.vstack((coordinate, np.array([[1]])))

	def _process_real_data_(self):
		"""
		Import the real-world data from the log file stored in the directory of data_path.
		This method brings all the data into a suitable format for training.
		"""
		print("Extracting the occupancy grid ...")
		# Occupancy grid data
		self.agent_container.occupancy_grid.resolution = 1  # map resolution in [m / cell]
		# Extract static obstacles
		obst_threshold = 200

		self.batch_grid = np.zeros([
			self.batch_size, self.tbpl,
			int(np.ceil(self.submap_width / self.agent_container.occupancy_grid.resolution)),
			int(np.ceil(self.submap_height / self.agent_container.occupancy_grid.resolution))]
		)

		# Get homography matrix to transform from image to world coordinates
		H = np.genfromtxt(os.path.join(self.data_path +self.args.scenario, 'H.txt'), delimiter='  ', unpack=True).transpose()
		# Get map image
		static_obst_img = cv2.imread(os.path.join(self.data_path+self.args.scenario, 'map.png'), 0)

		tl_px = self.cv_to_norm_px(np.array([[0], [0]]), static_obst_img.shape)
		br_px = self.cv_to_norm_px(np.array([[static_obst_img.shape[0]-1], [static_obst_img.shape[1]-1]]), static_obst_img.shape)
		tl_rw = np.dot(H, np.vstack((tl_px, np.array([[1]]))))[0:2,:]
		self.agent_container.occupancy_grid.tl_rw = tl_rw
		br_rw = np.dot(H, np.vstack((br_px, np.array([[1]]))))[0:2,:]
		self.agent_container.occupancy_grid.br_rw = br_rw
		if (br_rw[0, 0] < tl_rw[0, 0] or br_rw[1, 0] > tl_rw[1, 0]):
			print('One or more of the axes are flipeed! Check your H matrix.')
			exit()

		# Dynamically set occupancy grid size
		og_x_size = (br_rw-tl_rw)[0, 0]
		og_y_size = (tl_rw-br_rw)[1, 0]
		self.agent_container.occupancy_grid.map_size = np.array([og_x_size, og_y_size])  # map size in meters, [0] = x (→), [1] = y (↑)
		self.agent_container.occupancy_grid.center = self.agent_container.occupancy_grid.map_size / 2.0

		gridmap_us = cv2.resize(
			static_obst_img, 
			(int(og_x_size/self.agent_container.occupancy_grid.resolution), int(og_y_size/self.agent_container.occupancy_grid.resolution)), 
			fx=0, fy=0, interpolation = cv2.INTER_NEAREST
		)
		gridmap_us = np.rot90(gridmap_us, k=-1)
		gridmap_us[gridmap_us > 0.0] = 1.0
		self.agent_container.occupancy_grid.gridmap = gridmap_us

		print("Extracting the pedestrian data ...")
		# Pedestrian data
		# [id, timestep (s), timestep (ns), pos x, pos y, yaw, vel x, vel y, omega, goal x, goal y]
		if os.path.exists(self.data_path +self.args.scenario+'/obsmat.txt'):
			pedestrian_data = np.genfromtxt(os.path.join(self.data_path +self.args.scenario, 'obsmat.txt'), delimiter=" ")[1:, :]
			pixel_data = False
		elif os.path.exists(self.data_path +self.args.scenario+'/obsmat_px.txt'):
			pedestrian_data = np.genfromtxt(os.path.join(self.data_path + self.args.scenario, 'obsmat_px.txt'), delimiter="  ")[1:, :]
			pixel_data = True
		else:
			print("Could not find obsmat.txt or obsmat_px.txt")

		idx_frame = 0
		idx_id = 1
		idx_posx = 2
		idx_posy = 4
		idx_posz = 3
		idx_vx = 5
		idx_vy = 7
		idx_vz = 6
		self.dt = 0.4 # seconds (equivalent to 2.5 fps)
		if os.path.split(self.data_path)[-1] == 'seq_eth':
			frames_between_annotation = 6.0
		else:
			frames_between_annotation = 10.0

		# Iterate through the data and fill the register
		for sample_idx in range(pedestrian_data.shape[0]):
			id = pedestrian_data[sample_idx, idx_id]
			timestamp = pedestrian_data[sample_idx, idx_frame] * self.dt / frames_between_annotation  # time in seconds
			pose = np.zeros([1,3])
			vel = np.zeros([1,3])
			pose[:,0] = pedestrian_data[sample_idx, idx_posx]
			if self.args.scenario == "zara_02":
				pose[:, 1] = pedestrian_data[sample_idx, idx_posy] + 14
			else:
				pose[:,1] = pedestrian_data[sample_idx, idx_posy]
			
			vel[:, 0] = pedestrian_data[sample_idx, idx_vx]
			vel[:, 1] = pedestrian_data[sample_idx, idx_vy]
			# if pixel_data:
			# 	converted_pose = sup.to_pos_frame(H, np.expand_dims(np.array((pedestrian_data[sample_idx, idx_posx], pedestrian_data[sample_idx, idx_posy])), axis=0).astype(float))
			# 	pose[:, 0] = converted_pose[0,0]
			# 	pose[:, 1] = converted_pose[0,1]
			goal = np.zeros([2])

			self.agent_container.addDataSample(id, timestamp, pose, vel, goal)

		# Set the initial indices for agent trajectories (which trajectory will be returned when queried)
		self.agent_traj_idx = [0] * self.agent_container.getNumberOfAgents()

		# Subsample trajectories (longer discretization time)
		if self.dt != self.args.dt:
			for id in self.agent_container.getAgentIDs():
				for traj in self.agent_container.getAgentTrajectories(id):
					if len(traj) > self.min_length_trajectory:
						traj.smoothenTrajectory(dt=self.args.dt) # before was 0.3
						traj.goal = np.expand_dims(traj.pose_vec[-1, :2], axis=0)
					else:
						self.agent_container.removeAgent(id)

		# Put all the trajectories in the trajectory set and randomize
		for cnt, id in enumerate(self.agent_container.getAgentIDs()):
			self.addAgentTrajectoriesToSet(self.agent_container,self.trajectory_set,id)
		#random.shuffle(self.trajectory_set)

		# add velocities to st dataset by using kalman filter
		#if self.args.scenario == "st" :
		#	for traj_id in range(len(self.trajectory_set)):
				#filtered_state_means = self.filter_data(self.trajectory_set[traj_id][1].pose_vec[:,:2])
				#self.trajectory_set[traj_id][1].pose_vec[:,:2] = filtered_state_means[:,:2]
				#self.trajectory_set[traj_id][1].vel_vec = filtered_state_means[:, 2:]

		self.compute_min_max_values()

		# groups_path = os.path.join(self.data_path+self.args.scenario, 'groups.txt')
		# with open(groups_path,"r") as f:
		# 	all_data = [x.split() for x in f.readlines()]
		# 	lines = np.array([map(float,x) for x in all_data])
		self.groups = []

		# for line in lines:
		# 	if line:
		# 		self.groups.append(np.array(line))

	def calc_scale(self, keep_ratio=True):
		self.sx_vel = 1 / (self.max_vel_x - self.min_vel_x)
		self.sy_vel = 1 / (self.max_vel_y - self.min_vel_y)
		if keep_ratio:
			if self.sx_vel > self.sy_vel:
				self.sx_vel = self.sy_vel
			else:
				self.sy_vel = self.sx_vel

		self.sx_pos = 1 / (self.max_pos_x - self.min_pos_x)
		self.sy_pos = 1 / (self.max_pos_y - self.min_pos_y)
		if keep_ratio:
			if self.sx_pos > self.sy_pos:
				self.sx_pos = self.sy_pos
			else:
				self.sy_pos = self.sx_pos

	def normalize_vel(self, data, shift=False, inPlace=True):
		if inPlace:
			data_copy = data
		else:
			data_copy = np.copy(data)

		if data.ndim == 1:
			data_copy[0] = (data[0] - self.min_vel_x * shift) * self.sx_vel
			data_copy[1] = (data[1] - self.min_vel_y * shift) * self.sy_vel
		elif data.ndim == 2:
			data_copy[:, 0] = (data[:, 0] - self.min_vel_x * shift) * self.sx_vel
			data_copy[:, 1] = (data[:, 1] - self.min_vel_y * shift) * self.sy_vel
		elif data.ndim == 3:
			data_copy[:, :, 0] = (data[:, :, 0] - self.min_vel_x * shift) * self.sx_vel
			data_copy[:, :, 1] = (data[:, :, 1] - self.min_vel_y * shift) * self.sy_vel
		elif data.ndim == 4:
			data_copy[:, :, :, 0] = (data[:, :, :, 0] - self.min_vel_x * shift) * self.sx_vel
			data_copy[:, :, :, 1] = (data[:, :, :, 1] - self.min_vel_y * shift) * self.sy_vel
		else:
			return False
		return data_copy

	def normalize_pos(self, data, shift=False, inPlace=True):
		if inPlace:
			data_copy = data
		else:
			data_copy = np.copy(data)

		if data.ndim == 1:
			data_copy[0] = (data[0] - self.min_pos_x * shift) * self.sx_pos
			data_copy[1] = (data[1] - self.min_pos_y * shift) * self.sy_pos
		elif data.ndim == 2:
			data_copy[:, 0] = (data[:, 0] - self.min_pos_x * shift) * self.sx_pos
			data_copy[:, 1] = (data[:, 1] - self.min_pos_y * shift) * self.sy_pos
		elif data.ndim == 3:
			data_copy[:, :, 0] = (data[:, :, 0] - self.min_pos_x * shift) * self.sx_pos
			data_copy[:, :, 1] = (data[:, :, 1] - self.min_pos_y * shift) * self.sy_pos
		elif data.ndim == 4:
			data_copy[:, :, :, 0] = (data[:, :, :, 0] - self.min_pos_x * shift) * self.sx_pos
			data_copy[:, :, :, 1] = (data[:, :, :, 1] - self.min_pos_y * shift) * self.sy_pos
		else:
			return False
		return data_copy

	def denormalize_pos(self, data, shift=False, inPlace=True):
		if inPlace:
			data_copy = data
		else:
			data_copy = np.copy(data)

		ndim = data.ndim
		if ndim == 1:
			data_copy[0] = data[0] / self.sx_pos + self.min_pos_x * shift
			data_copy[1] = data[1] / self.sy_pos + self.min_pos_y * shift
		elif ndim == 2:
			data_copy[:, 0] = data[:, 0] / self.sx_pos + self.min_pos_x * shift
			data_copy[:, 1] = data[:, 1] / self.sy_pos + self.min_pos_y * shift
		elif ndim == 3:
			data_copy[:, :, 0] = data[:, :, 0] / self.sx_pos + self.min_pos_x * shift
			data_copy[:, :, 1] = data[:, :, 1] / self.sy_pos + self.min_pos_y * shift
		elif ndim == 4:
			data_copy[:, :, :, 0] = data[:, :, :, 0] / self.sx_pos + self.min_pos_x * shift
			data_copy[:, :, :, 1] = data[:, :, :, 1] / self.sy_pos + self.min_pos_y * shift
		else:
			return False

		return data_copy

	def denormalize_vel(self, data, shift=False, inPlace=True):
		if inPlace:
			data_copy = data
		else:
			data_copy = np.copy(data)

		ndim = data.ndim
		if ndim == 1:
			data_copy[0] = data[0] / self.sx_vel + self.min_vel_x * shift
			data_copy[1] = data[1] / self.sy_vel + self.min_vel_y * shift
		elif ndim == 2:
			data_copy[:, 0] = data[:, 0] / self.sx_vel + self.min_vel_x * shift
			data_copy[:, 1] = data[:, 1] / self.sy_vel + self.min_vel_y * shift
		elif ndim == 3:
			data_copy[:, :, 0] = data[:, :, 0] / self.sx_vel + self.min_vel_x * shift
			data_copy[:, :, 1] = data[:, :, 1] / self.sy_vel + self.min_vel_y * shift
		elif ndim == 4:
			data_copy[:, :, :, 0] = data[:, :, :, 0] / self.sx_vel + self.min_vel_x * shift
			data_copy[:, :, :, 1] = data[:, :, :, 1] / self.sy_vel + self.min_vel_y * shift
		else:
			return False

		return data_copy

	def filter_data(self, data):
		# Kalman filter considering a constant velocity model
		Transition_Matrix = [[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]]
		Observation_Matrix = [[1, 0, 0, 0], [0, 1, 0, 0]]

		xinit = data[0, 0]
		yinit = data[0, 1]
		vxinit = (data[1, 0] - data[0, 0]) / self.dt
		vyinit = (data[1, 1] - data[0, 1]) / self.dt
		initstate = [xinit, yinit, vxinit, vyinit]
		initcovariance = 1.0e-3 * np.eye(4) #  1.0e-3 * np.eye(4)
		transistionCov = 1.0e-4 * np.eye(4) # 1.0e-4 * np.eye(4)
		observationCov = 2.0e-3 * np.eye(2) # 1.0e-2 * np.eye(2)   -6
		kf = KalmanFilter(transition_matrices=Transition_Matrix,
		                  observation_matrices=Observation_Matrix,
		                  initial_state_mean=initstate,
		                  initial_state_covariance=initcovariance,
		                  transition_covariance=transistionCov,
		                  observation_covariance=observationCov)

		(filtered_state_means, filtered_state_covariances) = kf.filter(data)

		return filtered_state_means

	def saveTrajectoryData(self, save_path):
		print("Saving data to: '{}'".format(save_path))
		if not os.path.isdir(self.args.data_path + self.args.scenario):
			os.makedirs(self.args.data_path + self.args.scenario)

		# Reconstruct interpolators since they were not pickled with the rest of the trajectory
		for id, traj in self.trajectory_set:
			traj.updateInterpolators()

		#if "test" not in self.args.scenario:
		random.shuffle(self.trajectory_set)

		data = {
			"trajectories" : self.trajectory_set,
			"agent_container" : self.agent_container,
			"min_pos_x" : self.min_pos_x,
			"min_pos_y" : self.min_pos_y,
			"max_pos_x" : self.max_pos_x,
			"max_pos_y" : self.max_pos_y,
			"min_vel_x" : self.min_vel_x,
			"min_vel_y" : self.min_vel_y,
			"max_vel_x" : self.max_vel_x,
			"max_vel_y" : self.max_vel_y
		}
		pkl.dump(data, open(save_path, 'wb'),protocol=2)

	def loadTrajectoryData(self, load_path):
		print("Loading data from: '{}'".format(load_path))
		self.file = open(load_path, 'rb')
		if sys.version_info[0] < 3:
			tmp_self = pkl.loads(self.file)#,encoding='latin1')
		else:
			tmp_self = pkl.load(self.file , encoding='latin1')
		self.trajectory_set = tmp_self["trajectories"]
		self.agent_container = tmp_self["agent_container"]

		try:
			self.min_pos_x = tmp_self["min_pos_x"]
			self.min_pos_y = tmp_self["min_pos_y"]
			self.max_pos_x = tmp_self["max_pos_x"]
			self.max_pos_y = tmp_self["max_pos_y"]
			self.min_vel_x = tmp_self["min_vel_x"]
			self.min_vel_y = tmp_self["min_vel_y"]
			self.max_vel_x = tmp_self["max_vel_x"]
			self.max_vel_y = tmp_self["max_vel_y"]
		except:
			print("Delete dataset to compute limits")
			self.max_x = 1
			self.max_y = 1
			self.min_x = -1
			self.min_y = -1

		self.calc_scale()

		# Reconstruct interpolators since they were not pickled with the rest of the trajectory
		for id, traj in self.trajectory_set:
			traj.updateInterpolators()

	def getAgentTrajectory(self, agent_id):
		"""
		Return the next agent trajectory in the queue for the agent with id agent_id.
		"""
		trajectory = self.agent_container.agent_data[agent_id].trajectories[self.agent_traj_idx[agent_id]]
		self.agent_traj_idx[agent_id] = (self.agent_traj_idx[agent_id] + 1)  % self.agent_container.getNumberOfTrajectoriesForAgent(agent_id)
		return trajectory

	def getRandomAgentTrajectory(self, agent_id):
		"""
		Return a totally random trajectory for the agent with id agent_id.
		"""
		random_traj_idx = np.random.randint(0, len(self.agent_container.agent_data[agent_id].trajectories))
		return self.agent_container.agent_data[agent_id].trajectories[random_traj_idx]

	def fillBatch(self, agent_id, batch_idx, start_idx, truncated_backprop_length, batch_x, batch_vel,batch_pos,batch_grid, pedestrian_grid, batch_goal, batch_y, trajectory,batch_pos_target, centered_grid=False, testing = False):
		"""
		Fill the data batches of batch_idx with data for all truncated backpropagation steps.
		"""
		other_agents_pos = []

		for tbp_step in range(truncated_backprop_length):

			# Input values
			query_time = trajectory.time_vec[start_idx + tbp_step]
			for prev_step in range(self.prev_horizon,-1,-1):
				current_pos = np.array([trajectory.pose_vec[start_idx + tbp_step - prev_step, 0], trajectory.pose_vec[
					                        start_idx + tbp_step - prev_step, 1]])
				current_vel = np.array([trajectory.vel_vec[start_idx + tbp_step, 0], trajectory.vel_vec[
					                        start_idx + tbp_step - prev_step - prev_step, 1]])

				if self.args.normalize_data:
					self.normalize_pos(current_pos)
					self.normalize_vel(current_vel)

				batch_x[batch_idx, tbp_step, prev_step*self.input_dim:(prev_step+1)*self.input_dim] = np.array([current_pos[0],
																										current_pos[1],
																										current_vel[0],
																										current_vel[1]])
				#batch_vel[batch_idx, tbp_step, prev_step*self.input_state_dim:(prev_step+1)*self.input_state_dim] = np.array([np.linalg.norm(current_vel),
				#                                                                                                              np.arctan2(current_vel[1],current_vel[0])])
				batch_vel[batch_idx, tbp_step, prev_step*self.input_state_dim:(prev_step+1)*self.input_state_dim] = np.array([current_vel[0],
																											current_vel[1]])
				batch_pos[batch_idx, tbp_step, prev_step*self.input_state_dim:(prev_step+1)*self.input_state_dim] = np.array([current_pos[0],
																											current_pos[1]])
				
			heading = math.atan2(current_vel[1], current_vel[0])
			if centered_grid:
				grid_center = current_pos
				grid = self.agent_container.occupancy_grid.getSubmapByCoords(grid_center[0],
																																			 grid_center[1],
																																			 self.submap_width, self.submap_height)
				#grid = self.agent_container.occupancy_grid.getFrontSubmap(grid_center,
				#																													current_vel,
				#																													self.submap_width, self.submap_height)

			if self.rotated_grid:
				grid = sup.rotate_grid_around_center(grid, heading * 180 / math.pi)  # rotation in degrees

			batch_grid[batch_idx, tbp_step, :, :] = grid

			# batch_goal[batch_idx, tbp_step, :] = trajectory.goal

			# Find positions of other pedestrians at the current timestep and order them by dstance to query agent
			other_positions = trajectory.other_agents_positions[start_idx + tbp_step]
			n_other_agents = other_positions.shape[0]
			other_velocities = trajectory.other_agents_velocities[start_idx + tbp_step]

			# Compute distance to other agents and order them
			other_poses_ordered = np.zeros((other_positions.shape[0], 6))
			other_poses_ordered[:, :2] = other_positions
			other_poses_ordered[:, 2:4] = other_velocities
			other_agents_pos.append(other_poses_ordered)

			current_pos = np.array(
				[trajectory.pose_vec[start_idx + tbp_step, 0], trajectory.pose_vec[start_idx + tbp_step, 1]])
			current_vel = np.array([trajectory.vel_vec[start_idx + tbp_step, 0], trajectory.vel_vec[start_idx + tbp_step, 1]])
			heading = math.atan2(current_vel[1], current_vel[0])

			for ag_id in range(n_other_agents):
				other_poses_ordered[ag_id,4] = np.linalg.norm(other_poses_ordered[ag_id,:2] - current_pos)
				# ordered ids
				other_poses_ordered[ag_id,5] = ag_id
			other_poses_ordered= other_poses_ordered[other_poses_ordered[:, 4].argsort()]

			if self.args.others_info == "relative":
				for ag_id in range(min(n_other_agents,self.args.n_other_agents)):
					rel_pos = np.array([other_poses_ordered[ag_id,0] - current_pos[0],other_poses_ordered[ag_id, 1] - current_pos[1]])*\
							      multivariate_normal.pdf(np.linalg.norm(np.array([other_poses_ordered[ag_id,:2] - current_pos])),
							                                                  mean=0.0,cov=5.0)
					rel_vel = np.array([other_poses_ordered[ag_id,2] - current_vel[0],other_poses_ordered[ag_id, 3] - current_vel[1]])

					pedestrian_grid[batch_idx, tbp_step, ag_id*4:ag_id*4+4] = np.concatenate([rel_pos, rel_vel])
					#pedestrian_grid[batch_idx, tbp_step, ag_id, 4] = np.linalg.norm(rel_pos)
					#pedestrian_grid[batch_idx, tbp_step, ag_id, 5] = np.arctan2(rel_pos[1], rel_pos[0])

			elif self.args.others_info == "angular_grid":
				other_pos_local_frame = sup.positions_in_local_frame(current_pos, heading, other_positions)
				radial_pedestrian_grid = sup.compute_radial_distance_vector(self.pedestrian_vector_dim, other_pos_local_frame,
																															max_range=self.max_range_ped_grid, min_angle=0, max_angle=2*np.pi,
																															normalize=True)
				pedestrian_grid[batch_idx, tbp_step, :] = radial_pedestrian_grid

			# Output values
			for pred_step in range(self.output_sequence_length):
				vx = trajectory.vel_vec[start_idx + tbp_step + 1 + pred_step, 0]
				vy = trajectory.vel_vec[start_idx + tbp_step + 1 + pred_step, 1]
				px = trajectory.pose_vec[start_idx + tbp_step + 1 + pred_step, 0]
				py = trajectory.pose_vec[start_idx + tbp_step + 1 + pred_step, 1]
				batch_y[batch_idx, tbp_step, self.output_state_dim*pred_step] = vx# - current_vel[0]
				batch_y[batch_idx, tbp_step, self.output_state_dim*pred_step + 1] = vy# - current_vel[1]
				if self.args.normalize_data:
					self.normalize_vel(batch_y[batch_idx, tbp_step, self.output_state_dim*pred_step:self.output_state_dim*pred_step+2])
				batch_pos_target[batch_idx, tbp_step, self.output_state_dim*pred_step] = px - trajectory.pose_vec[start_idx + tbp_step, 0]
				batch_pos_target[batch_idx, tbp_step, self.output_state_dim*pred_step + 1] = py - trajectory.pose_vec[start_idx + tbp_step, 1]

			current_pos = trajectory.pose_vec[start_idx + tbp_step, :2]
			#delta_pos = trajectory.pose_vec[start_idx + tbp_step + self.prediction_horizon,:2] - current_pos
			delta_pos = trajectory.goal[0] - current_pos
			#batch_goal[batch_idx, tbp_step, 0] = np.linalg.norm(delta_pos)
			#batch_goal[batch_idx, tbp_step, 1] = np.arctan2(delta_pos[1],
			#                                                delta_pos[0])
			# Used for planning
			batch_goal[batch_idx, tbp_step, 0] = delta_pos[0]/np.linalg.norm(delta_pos) #trajectory.goal[0,0]
			batch_goal[batch_idx, tbp_step, 1] = delta_pos[1]/np.linalg.norm(delta_pos) #trajectory.goal[0,1]

		return other_agents_pos

	def getBatch(self):
		"""
		Get the next batch of training data.
		"""
		# Update sequences
		# If batch sequences are empty and need to be filled
		trajectory=[]
		if len(self.batch_sequences) == 0:
			for b in range(0,min(self.batch_size,int(len(self.trajectory_set)*self.train_set))):
				id, trajectory = self.trajectory_set[self.data_idx]
				self.data_idx += 1
				self.batch_sequences.append(trajectory)
				self.batch_ids.append(id)
		# If batch sequences are filled and can be used or need to be updated.
		other_agents_pos = []
		new_epoch = False
		for ii, traj in enumerate(self.batch_sequences):
			if self.sequence_idx[ii] + self.tbpl + self.output_sequence_length + 1 >= len(traj):
				id, trajectory = self.trajectory_set[self.data_idx]
				self.data_idx = (self.data_idx + 1) % int(len(self.trajectory_set)*self.train_set)
				if self.data_idx == 0:
					new_epoch = True
				self.batch_sequences[ii] = trajectory
				self.batch_ids[ii] = id
				self.sequence_idx[ii] = self.args.prev_horizon
				self.sequence_reset[ii] = 1
			else:
				self.sequence_reset[ii] = 0

		# Fill the batch
		other_agents_pos = []
		# Th second argument is needed such that when the dataset is too small the code does not break
		for ii in range(0,min(self.batch_size,len(self.trajectory_set)-len(self.trajectory_set)%self.batch_size)):
			traj = self.batch_sequences[ii]
			agent_id = self.batch_ids[ii]
			other_agents_pos.append(self.fillBatch(agent_id, ii, int(self.sequence_idx[ii]), self.tbpl, self.batch_x, self.batch_vel, self.batch_pos,self.batch_grid, self.pedestrian_grid, self.batch_goal, self.batch_y, traj,self.batch_pos_target, centered_grid=self.centered_grid))
			self.sequence_idx[ii] += self.tbpl

		if self.args.rotated_grid:
			_, self.batch_y = sup.rotate_batch_to_local_frame(self.batch_y,self.batch_x)
			self.batch_x, self.batch_pos_target = sup.rotate_batch_to_local_frame(self.batch_pos_target, self.batch_x)

		return deepcopy(self.batch_x), deepcopy(self.batch_vel), deepcopy(self.batch_pos), deepcopy(self.batch_goal), \
		                deepcopy(self.batch_grid), deepcopy(self.pedestrian_grid), deepcopy(self.batch_y), deepcopy(self.batch_pos_target), \
		                deepcopy(other_agents_pos), new_epoch

	def getTestBatch(self):
		"""
		Get the next batch for model validation
		"""
		# Update sequences
		# If batch sequences are empty and need to be filled

		self.val_data_idx = max(self.val_data_idx,int(len(self.trajectory_set)*self.train_set))
		trajectory=[]
		if len(self.val_batch_sequences) == 0:
			for b in range(0,self.batch_size):
				id, trajectory = self.trajectory_set[self.val_data_idx]
				self.val_data_idx += 1
				self.val_batch_sequences.append(trajectory)
				self.val_batch_ids.append(id)
				if self.val_data_idx == len(self.trajectory_set):
					self.val_data_idx =int (len(self.trajectory_set) * self.train_set)

		# If batch sequences are filled and can be used or need to be updated.
		other_agents_pos = []
		for ii, traj in enumerate(self.val_batch_sequences):
			if self.val_sequence_idx[ii] + self.tbpl + self.output_sequence_length + 1 >= len(traj):
				id, trajectory = self.trajectory_set[self.val_data_idx]
				self.val_data_idx = (self.val_data_idx + 1) % int(len(self.trajectory_set)*(1-self.train_set)-1) + int(len(self.trajectory_set)*self.train_set)
				self.val_batch_sequences[ii] = trajectory
				self.val_batch_ids[ii] = id
				self.val_sequence_idx[ii] = self.args.prev_horizon
				self.val_sequence_reset[ii] = 1
			else:
				self.val_sequence_reset[ii] = 0

		# Fill the batch
		other_agents_pos = []
		# Th second argument is needed such that when the dataset is too small the code does not break
		for ii in range(0,min(self.batch_size,len(self.trajectory_set)-len(self.trajectory_set)%self.batch_size)):
			traj = self.val_batch_sequences[ii]
			agent_id = self.val_batch_ids[ii]
			other_agents_pos.append(self.fillBatch(agent_id, ii, int(self.val_sequence_idx[ii]), self.tbpl, self.val_batch_x,
			                        self.val_batch_vel, self.val_batch_pos,self.val_batch_grid, self.val_pedestrian_grid, self.val_batch_goal,
			                        self.val_batch_y, traj,self.val_batch_pos_target, centered_grid=self.centered_grid))
			self.val_sequence_idx[ii] += self.tbpl

		if self.args.rotated_grid:
			_, self.val_batch_y = sup.rotate_batch_to_local_frame(self.val_batch_y,self.val_batch_x)
			self.val_batch_x, self.val_batch_pos_target = sup.rotate_batch_to_local_frame(self.val_batch_pos_target, self.val_batch_x)

		# Create dictionary to feed into the model
		dict = {"batch_x": self.val_batch_x,
		        "batch_vel": self.val_batch_vel,
		        "batch_pos": self.val_batch_pos,
		        "batch_goal": self.val_batch_goal,
		        "batch_grid": self.val_batch_grid,
		        "batch_ped_grid": self.val_pedestrian_grid,
		        "batch_y": self.val_batch_y,
		        "batch_div": self.val_batch_y,
		        "batch_pos_target": self.val_batch_pos_target,
		        "state_noise": 0.0,
		        "grid_noise": 0.0,
		        "ped_noise":0.0,
		        "other_agents_pos": other_agents_pos
		        }

		return dict

	def getTrajectoryAsBatch(self, trajectory_idx, max_sequence_length=1000, freeze = False):
		"""
		Get a trajectory out of the trajectory set in the same format as for the standard training data
		(e.g. for validation purposes).
		"""
		id = self.trajectory_set[trajectory_idx][0]
		traj = self.trajectory_set[trajectory_idx][1]
		grid = self.agent_container.occupancy_grid.getSubmapByCoords(traj.pose_vec[0,0] * self.norm_const_x,
																																 traj.pose_vec[0,1] * self.norm_const_y,
																																 self.submap_width, self.submap_height)
		sequence_length = min(max_sequence_length, traj.pose_vec.shape[0] - self.output_sequence_length-self.prev_horizon)
		batch_x = np.zeros([1, sequence_length, (self.prev_horizon+1)*self.input_dim])
		batch_pos_target = np.zeros([1, sequence_length, 2*self.args.prediction_horizon])
		if "future" in self.args.others_info:
			batch_vel = np.zeros(
				[1, sequence_length, self.input_state_dim * self.prediction_horizon])  # data fed for training
		else:
			batch_vel = np.zeros([1, sequence_length, self.input_state_dim * (self.prev_horizon + 1)])
		batch_grid = np.zeros([1, sequence_length,
													 int(np.ceil(self.submap_width / self.agent_container.occupancy_grid.resolution)),
													 int(np.ceil(self.submap_height / self.agent_container.occupancy_grid.resolution))])
		batch_goal = np.zeros([1, sequence_length, 2])
		batch_y = np.zeros([1, sequence_length, self.output_state_dim * self.output_sequence_length])
		batch_pos = np.zeros([1, sequence_length, self.output_state_dim * self.output_sequence_length])
		if self.args.others_info == "relative":
			pedestrian_grid = np.zeros([1, sequence_length, self.pedestrian_vector_dim*self.args.n_other_agents])
		elif "sequence" in self.args.others_info:
			pedestrian_grid = np.zeros(
				[1, sequence_length, self.args.n_other_agents, self.pedestrian_vector_dim * self.prediction_horizon])
		elif self.args.others_info == "prev_sequence":
			pedestrian_grid = np.zeros(
				[1, sequence_length, self.args.n_other_agents, self.pedestrian_vector_dim*(self.prev_horizon + 1)])
		elif self.args.others_info == "sequence2":
			pedestrian_grid = np.zeros(
				[1, sequence_length, self.args.n_other_agents, self.prediction_horizon,self.pedestrian_vector_dim])
		elif self.args.others_info == "ped_grid":
			pedestrian_grid = np.zeros([1, sequence_length,
			                       int(np.ceil(self.submap_width / self.agent_container.occupancy_grid.resolution)),
			                       int(np.ceil(self.submap_height / self.agent_container.occupancy_grid.resolution))])
		else:
			pedestrian_grid = np.zeros([1, sequence_length, self.pedestrian_vector_dim])
		other_agents_pos = self.fillBatch(id, 0, self.prev_horizon, sequence_length, batch_x, batch_vel,batch_pos,batch_grid, pedestrian_grid, batch_goal, batch_y, traj,batch_pos_target, centered_grid=self.centered_grid)

		if freeze:
			pedestrian_grid = pedestrian_grid*0.0

		return batch_x, batch_vel, batch_pos,batch_goal, batch_grid, pedestrian_grid, batch_y, batch_pos_target, other_agents_pos, traj

	def getGroupOfTrajectoriesAsBatch(self, trajectory_idx, max_sequence_length=1000, n_trajs=1):
		"""
		Get a group of trajectories out o the trajectory set in the same format as for the standard training data
		(e.g. for group prediction).
		"""

		batch_x_list = []
		batch_vel_list = []
		batch_pos_list = []
		batch_goal_list = []
		batch_grid_list = []
		pedestrian_grid_list = []
		batch_y_list = []
		batch_pos_final_list = []
		other_agents_pos_list = []
		traj_list = []

		agent_id = self.trajectory_set[trajectory_idx][0]
		traj = self.trajectory_set[trajectory_idx][1]
		grid = self.agent_container.occupancy_grid.getSubmapByCoords(traj.pose_vec[0,0] * self.norm_const_x,
																																 traj.pose_vec[0,1] * self.norm_const_y,
																																 self.submap_width, self.submap_height)
		sequence_length = min(max_sequence_length, traj.pose_vec.shape[0] - self.output_sequence_length-self.prev_horizon)
		batch_x = np.zeros([1, sequence_length, (self.prev_horizon+1)*self.input_dim])
		batch_pos_final = np.zeros([1, sequence_length, self.args.output_dim])
		batch_vel = np.zeros([1, sequence_length, self.input_state_dim * (self.prev_horizon + 1)])
		batch_grid = np.zeros([1, sequence_length,
													 int(np.ceil(self.submap_width / self.agent_container.occupancy_grid.resolution)),
													 int(np.ceil(self.submap_height / self.agent_container.occupancy_grid.resolution))])
		batch_goal = np.zeros([1, sequence_length, 2])
		batch_y = np.zeros([1, sequence_length, self.output_state_dim * self.output_sequence_length])
		batch_pos = np.zeros([1, sequence_length, self.output_state_dim * self.output_sequence_length])
		pedestrian_grid = np.zeros(
				[1, sequence_length, self.args.n_other_agents, self.pedestrian_vector_dim * self.prediction_horizon])

		other_agents_pos_list.append(self.fillBatch(agent_id, 0, self.prev_horizon, sequence_length, batch_x, batch_vel, batch_pos,
		                                  batch_grid, pedestrian_grid, batch_goal, batch_y, traj, batch_pos_final,
		                                  centered_grid=self.centered_grid))

		batch_x_list.append(batch_x)
		batch_vel_list.append(batch_vel)
		batch_pos_list.append(batch_pos)
		batch_goal_list.append(batch_goal)
		batch_grid_list.append(batch_grid)
		pedestrian_grid_list.append(pedestrian_grid)
		batch_y_list.append(batch_y)
		batch_pos_final_list.append(batch_pos_final)
		traj_list.append(traj)

		for id in self.agent_container.agent_data.keys():
			if id != agent_id:
				t = self.agent_container.agent_data[id].getTrajectoryForTime(traj.time_vec[0])
				sequence_length = min(max_sequence_length,
				                      t.pose_vec.shape[0] - self.output_sequence_length - self.prev_horizon)
				batch_x = np.zeros([1, sequence_length, (self.prev_horizon + 1) * self.input_dim])
				batch_pos_final = np.zeros([1, sequence_length, self.args.output_dim])
				batch_vel = np.zeros([1, sequence_length, self.input_state_dim * (self.prev_horizon + 1)])
				batch_grid = np.zeros([1, sequence_length,
				                       int(np.ceil(self.submap_width / self.agent_container.occupancy_grid.resolution)),
				                       int(np.ceil(self.submap_height / self.agent_container.occupancy_grid.resolution))])
				batch_goal = np.zeros([1, sequence_length, 2])
				batch_y = np.zeros([1, sequence_length, self.output_state_dim * self.output_sequence_length])
				batch_pos = np.zeros([1, sequence_length, self.output_state_dim * self.output_sequence_length])
				pedestrian_grid = np.zeros(
					[1, sequence_length, self.args.n_other_agents, self.pedestrian_vector_dim * self.prediction_horizon])
				other_agents_pos_list.append(self.fillBatch(id, 0, self.prev_horizon, sequence_length, batch_x, batch_vel,batch_pos,batch_grid, pedestrian_grid, batch_goal, batch_y, t,batch_pos_final, centered_grid=self.centered_grid))
				batch_x_list.append(batch_x)
				batch_vel_list.append(batch_vel)
				batch_pos_list.append(batch_pos)
				batch_goal_list.append(batch_goal)
				batch_grid_list.append(batch_grid)
				pedestrian_grid_list.append(pedestrian_grid)
				batch_y_list.append(batch_y)
				batch_pos_final_list.append(batch_pos_final)
				traj_list.append(t)

		return batch_x_list, batch_vel_list, batch_pos_list,batch_goal_list, batch_grid_list, pedestrian_grid_list, batch_y_list, batch_pos_final_list, other_agents_pos_list, traj_list

	def getTestTrajectoryAsBatch(self, trajectory_idx, max_sequence_length=1000,freeze=False):
		"""
		Get a trajectory out of the trajectory set in the same format as for the standard training data
		(e.g. for validation purposes).
		"""
		print(trajectory_idx)
		id = self.test_trajectory_set[trajectory_idx][0]
		traj = self.test_trajectory_set[trajectory_idx][1]

		sequence_length = min(max_sequence_length, traj.pose_vec.shape[0] - self.output_sequence_length-self.prev_horizon)
		batch_x = np.zeros([1, sequence_length, (self.prev_horizon+1)*self.input_dim])  # data fed for training
		batch_pos_final = np.zeros([1, sequence_length, self.args.output_dim])
		if "future" in self.args.others_info:
			batch_vel = np.zeros(
				[1, sequence_length, self.input_state_dim * self.prediction_horizon])  # data fed for training
		else:
			batch_vel = np.zeros([1, sequence_length, self.input_state_dim * (self.prev_horizon + 1)])
		batch_grid = np.zeros([1, sequence_length,
													 int(np.ceil(self.submap_width / self.args.submap_resolution)),
													 int(np.ceil(self.submap_height / self.args.submap_resolution))])
		batch_goal = np.zeros([1, sequence_length, 2])
		batch_y = np.zeros([1, sequence_length, self.output_state_dim * self.output_sequence_length])
		batch_pos = np.zeros([1, sequence_length, self.output_state_dim * self.output_sequence_length])
		if self.args.others_info == "relative":
			pedestrian_grid = np.zeros([1,sequence_length, self.pedestrian_vector_dim*self.args.n_other_agents])
		elif "sequence" in self.args.others_info:
			pedestrian_grid = np.zeros(
				[1, sequence_length, self.args.n_other_agents, self.pedestrian_vector_dim * self.prediction_horizon])
		elif self.args.others_info == "prev_sequence":
			pedestrian_grid = np.zeros(
				[1, sequence_length, self.args.n_other_agents, self.pedestrian_vector_dim*(self.prev_horizon + 1)])
		elif self.args.others_info == "sequence2":
			pedestrian_grid = np.zeros(
				[1, sequence_length, self.args.n_other_agents, self.prediction_horizon,self.pedestrian_vector_dim])
		else:
			pedestrian_grid = np.zeros([1, sequence_length, self.pedestrian_vector_dim])
		other_agents_pos = self.fillBatch(id, 0, self.prev_horizon, sequence_length, batch_x, batch_vel,batch_pos,batch_grid, pedestrian_grid, batch_goal, batch_y, traj,batch_pos_final, centered_grid=self.centered_grid,testing=True)

		if freeze:
			pedestrian_grid = pedestrian_grid*0.0

		return batch_x, batch_vel, batch_pos,batch_goal, batch_grid, pedestrian_grid, batch_y, other_agents_pos, traj

	def getAgentDataAtTimestep(self, query_time):
		"""
		Get data from all agents which are active at a certain time.
		"""
		agent_dict = {}  # key is agent id
		for id in self.agent_container.agent_data.keys():
			traj_for_time = self.agent_container.agent_data[id].getTrajectoryForTime(query_time)
			if traj_for_time != None:
				pose = traj_for_time.getPoseAtTime(query_time)
				vel = traj_for_time.getVelAtTime(query_time)
				heading = math.atan2(vel[1], vel[0])
				grid = self.agent_container.occupancy_grid.getSubmapByCoords(pose[0], pose[1], self.submap_width, self.submap_height)
				rotated_grid = sup.rotate_grid_around_center(grid, -heading * 180 / math.pi)
				radial_pedestrian_grid = np.zeros([self.pedestrian_vector_dim])  # Initialize pedestriangrid with 0, will be evaluated later
				goal = traj_for_time.goal
				future_pos_gt = np.zeros([0, 2])
				future_vel_gt = np.zeros([0, 2])
				for future_step in range(1, self.output_sequence_length+1):
					t_future = query_time + future_step * self.dt
					if t_future <= traj_for_time.getMaxTime():
						pose_step = traj_for_time.getPoseAtTime(t_future)
						vel_step = traj_for_time.getVelAtTime(t_future)
						future_pos_gt = np.append(future_pos_gt, np.expand_dims(pose_step[:2], axis=0), axis=0)
						future_vel_gt = np.append(future_vel_gt, np.expand_dims(vel_step[:2], axis=0), axis=0)

				agent_dict[id] = [pose, vel, rotated_grid, radial_pedestrian_grid, goal, future_pos_gt, future_vel_gt]

		# For each pedestrian, find other agents position for this time and compute pedestrian grid
		for id in agent_dict.keys():
			other_agents_pos = np.zeros([0,2])
			for other_id in agent_dict.keys():
				if other_id != id:
					other_agents_pos = np.append(other_agents_pos, np.expand_dims(agent_dict[other_id][0][:2], axis=0), axis=0)
			pos = agent_dict[id][0][:2]
			vel = agent_dict[id][1][:2]
			heading = math.atan2(vel[1], vel[0])
			other_pos_local_frame = sup.positions_in_local_frame(pos, heading, other_agents_pos)
			agent_dict[id][3] = sup.compute_radial_distance_vector(self.pedestrian_vector_dim, other_pos_local_frame,
																													max_range=self.max_range_ped_grid, min_angle=0, max_angle=2*np.pi,
																													normalize=True)

		return agent_dict

	def plot_global_scenario(self,batch_grid,batch_x,batch_agent_real_traj,batch_goal,other_agents_pos,batch_agent_predicted_traj,t,seq_index):
		"""
			inputs:
				batch_grid: batch_grid numpy array (gridmap) in the global frame [Batch_size][truncated_back_propagation_time][width][height]
				batch_x: batch of initial states [x,y, vx, vy] on the global frame
				batch_agent_real_traj: agent ground truth velocity in global frame also know as batch_y
				batch_goal: batch with the trajectory goal positons
				other_agents_pos: other agent position in the global frame
				batch_agent_predicted_traj: agent predicted velocities in global frame
			"""
		global_grid = self.agent_container.occupancy_grid
		fig = pl.figure("Global Trajectory Predictions")
		ax_in = pl.subplot()

		#color definitions
		colormap = pl.get_cmap('rainbow')
		c_norm = pl.matplotlib.colors.Normalize(vmin=0, vmax=100)
		scalar_color_map = pl.cm.ScalarMappable(norm=c_norm, cmap=colormap)
		r= random.randint(0,100)

		ax_in.clear()
		#plot scenario grid
		sup.plot_grid(ax_in, np.array([0.0, 0.0]), global_grid.gridmap, global_grid.resolution, global_grid.map_size)
		ax_in.set_xlim([-global_grid.center[0], global_grid.center[0]])
		ax_in.set_ylim([-global_grid.center[1], global_grid.center[1]])
		#ax_in.set_aspect("equal")

		#plot initial and goal pose
		x_init_global_frame = batch_x[seq_index,t, 0]*self.norm_const_x
		y_init_global_frame = batch_x[seq_index, t, 1]*self.norm_const_y
		ax_in.plot(x_init_global_frame, y_init_global_frame, color='g', marker='o', label='Agent initial pos')
		ax_in.plot(batch_goal[seq_index,t, 0]*self.norm_const_x, batch_goal[seq_index,t, 1]*self.norm_const_y,
							 color='purple', marker='o', label='Agent goal')

		#plot predicted trajectory global frame
		vel_pred = np.zeros((self.args.prediction_horizon, 2))
		for mix_idx in range(self.args.n_mixtures):
			# plot predicted trajectory global frame
			for pred_step in range(self.args.prediction_horizon):
				idx = pred_step * self.args.output_pred_state_dim * self.args.n_mixtures + mix_idx
				idy = pred_step * self.args.output_pred_state_dim * self.args.n_mixtures + mix_idx + self.args.n_mixtures
				mu_x = batch_agent_predicted_traj[seq_index, t, idx]
				mu_y = batch_agent_predicted_traj[seq_index, t, idy]
				vel_pred[pred_step, :] = [mu_x, mu_y]

			traj_pred = sup.path_from_vel(initial_pos=np.array([batch_x[seq_index,t, 0]*self.norm_const_x, batch_x[seq_index,t, 1]*self.norm_const_y]),
																		pred_vel=vel_pred, dt=self.dt,n_vx=self.norm_const_vx,n_vy=self.norm_const_vy)
		x_pred = traj_pred[:, 0] #* self.norm_const_x
		y_pred = traj_pred[:, 1] #* self.norm_const_y
		ax_in.plot(x_pred, y_pred, color='r',label='Predicted trajectory')

		#plot real trajectory global frame
		color_value = scalar_color_map.to_rgba(r)
		real_traj = np.zeros((self.args.prediction_horizon, 2))
		for i in range(self.args.prediction_horizon):
			idx = i * self.args.output_dim
			idy = i * self.args.output_dim + 1
			mu_x = batch_agent_real_traj[seq_index, t, idx]
			mu_y = batch_agent_real_traj[seq_index, t, idy]
			real_traj[i, :] = [mu_x, mu_y]
		real_traj_global_frame = sup.path_from_vel(initial_pos=np.array([batch_x[seq_index, t, 0]*self.norm_const_x, batch_x[seq_index, t, 1]*self.norm_const_y]),
												pred_vel=real_traj, dt=self.dt,n_vx=self.norm_const_vx,n_vy=self.norm_const_vy)
		x = real_traj_global_frame[:, 0] #* self.norm_const_x
		y = real_traj_global_frame[:, 1] #* self.norm_const_y
		ax_in.plot(x, y, marker='x',color='b',label='Real trajectory')

		#Plot other agent positions
		if self.multi_pedestrian:
			other_pos = other_agents_pos[seq_index][t]
			try:
				ax_in.plot(other_pos[0, 0], other_pos[0, 1], color='r', marker='o', label='Other agents')
				for jj in range(1,other_pos.shape[0]):
					ax_in.plot(other_pos[jj, 0], other_pos[jj, 1], color='r', marker='o') #other agent pos positon
			except IndexError:
				print("Oops!  That was no valid number.  Try again...")
		ax_in.legend()
		fig.canvas.draw()
		pl.show(block=False)

	def plot_local_scenario(self,batch_grid,batch_x,batch_agent_real_traj,batch_goal,other_agents_pos,batch_agent_predicted_traj,t,seq_index):
		"""
			inputs:
				global_grid: map grid of the scenario
				batch_grid: batch_grid numpy array (gridmap) in the global frame [Batch_size][truncated_back_propagation_time][width][height]
				batch_agent_real_traj: agent ground truth trajectory in global frame
				batch_agent_predicted_traj: agent predicted trajectory in global frame
			"""
		#pl.close('all')
		global_grid = self.agent_container.occupancy_grid

		fig = pl.figure("Local Trajectory Predictions")
		ax_out = pl.subplot()

		#color definitions
		colormap = pl.get_cmap('rainbow')
		c_norm = pl.matplotlib.colors.Normalize(vmin=0, vmax=100)
		scalar_color_map = pl.cm.ScalarMappable(norm=c_norm, cmap=colormap)
		r= random.randint(0,100)
		color_value = scalar_color_map.to_rgba(r)
		#Convert initial position to local frame
		#ini_pos_local_frame = sup.rotate_batch_x_to_local_frame(batch_x)
		ax_out.clear()

		#plot initial and goal pose
		x_init_global_frame = batch_x[seq_index,t, 0]*self.norm_const_x
		y_init_global_frame = batch_x[seq_index, t, 1]*self.norm_const_y

		# plot initial and goal pose
		ax_out.plot(0, 0, color='g', marker='o', label='Agent initial pos')

		x_lim = [-self.submap_width/2, self.submap_width/2]
		y_lim = [-self.submap_height/2, self.submap_height/2]
		ax_out.set_xlim(x_lim)
		ax_out.set_ylim(y_lim)

		# plot predicted trajectory local frame
		vel_pred = np.zeros((self.args.prediction_horizon, 2))
		for mix_idx in range(self.args.n_mixtures):
			# plot predicted trajectory global frame
			for pred_step in range(self.args.prediction_horizon):
				idx = pred_step * self.args.output_pred_state_dim * self.args.n_mixtures + mix_idx
				idy = pred_step * self.args.output_pred_state_dim * self.args.n_mixtures + mix_idx + self.args.n_mixtures
				mu_x = batch_agent_predicted_traj[seq_index, t, idx]
				mu_y = batch_agent_predicted_traj[seq_index, t, idy]
				vel_pred[pred_step, :] = [mu_x, mu_y]

			traj_pred = sup.path_from_vel(initial_pos=np.array([0, 0]),
																		pred_vel=vel_pred, dt=self.dt,n_vx=self.norm_const_vx,n_vy=self.norm_const_vy)
			x_pred = traj_pred[:, 0] #* self.norm_const_x
			y_pred = traj_pred[:, 1] #* self.norm_const_y
			ax_out.plot(x_pred, y_pred, color='r',label='Predicted trajectory')

		# plot real trajectory local frame does it make sense? isnt batch_y in the robot frame
		real_traj = np.zeros((self.args.prediction_horizon, 2))
		for i in range(self.args.prediction_horizon):
			idx = i * self.args.output_dim
			idy = i * self.args.output_dim + 1
			mu_x = batch_agent_real_traj[seq_index, t, idx]
			mu_y = batch_agent_real_traj[seq_index, t, idy]
			real_traj[i, :] = [mu_x, mu_y]
		real_traj_global_frame = sup.path_from_vel(initial_pos=np.array([0, 0]),
																								 pred_vel=real_traj,
																							 dt=self.dt,n_vx=self.norm_const_vx,n_vy=self.norm_const_vy)

		# Query Agent Local frame plot
		if real_traj[0, 0] > 0.1:
			sup.plot_grid(ax_out, np.array([3.0, 0.0]), batch_grid[seq_index,t,:,:], global_grid.resolution,
			              np.array([self.args.submap_width, self.args.submap_height]))
		elif real_traj[0, 0] < -0.1:
			sup.plot_grid(ax_out, np.array([-3.0, 0.0]), batch_grid[seq_index,t,:,:], global_grid.resolution,
			              np.array([self.args.submap_width, self.args.submap_height]))
		else:
			sup.plot_grid(ax_out, np.array([0.0, 0.0]), batch_grid[seq_index,t,:,:], global_grid.resolution,
			              np.array([self.args.submap_width, self.args.submap_height]))

		x = real_traj_global_frame[:, 0]
		y = real_traj_global_frame[:, 1]
		ax_out.plot(x, y, marker='x', color='b', label='Agent real trajectory')
		#ax_out.plot(batch_x[seq_index,:, 0] * self.norm_const_x-x_init_global_frame*np.ones_like(batch_x[seq_index,:, 0]),
		#            batch_x[seq_index, :, 1] * self.norm_const_y-y_init_global_frame*np.ones_like(batch_x[seq_index,:, 0]), color='k')

		# Plot other agent positions
		if self.multi_pedestrian:
			other_pos = other_agents_pos[seq_index][t]
			heading = math.atan2(batch_x[seq_index, t, 3], batch_x[seq_index, t, 2])

			try:
				ax_out.plot(other_pos[0, 0]-x_init_global_frame, other_pos[0, 1]-y_init_global_frame, color='r', marker='o',label='Other agents')
				for jj in range(1,other_pos.shape[0]):
					ax_out.plot(other_pos[jj, 0]-x_init_global_frame, other_pos[jj, 1]-y_init_global_frame, color='r', marker='o')#,markersize=23)  # other agent pos positon
					#ax_out.annotate(str(jj), xy=(other_pos[jj, 0]-x_init_global_frame , other_pos[jj, 1]-y_init_global_frame))
			except IndexError:
				print("Oops!  That was no valid number.  Try again...")

		circle = pl.Circle((0, 0), self.max_range_ped_grid, color='blue',
											 fill=False)
		ax_out.add_artist(circle)
		ax_out.set_aspect("equal")
		ax_out.legend()
		fig.canvas.draw()
				#sleep(0.5)  # Time in seconds.
		pl. show(block=False)

	def add_agent_to_grid(self,batch_grid,position,velocity):
		"""
		This function is heavily based on the proxemic layer ROS implementation
		from https://github.com/DLu/navigation_layers.git
		:param batch_grid:
		:param batch_x:
		:param other_agents:
		:param args:
		:return:
		"""
		# Position relative to query-agent
		cx = position[0, 0]
		cy = position[0, 1]
		rows = batch_grid.shape[0]
		cols = batch_grid.shape[1]

		angle = np.arctan2(velocity[0,1], velocity[0,0])
		mag = np.linalg.norm(velocity[0])
		factor = 1.0 + mag * self.factor_

		# Position in map frame
		dx = 	int((cx + rows / 2*self.args.submap_resolution ) / self.args.submap_resolution)
		dy = int((cy + cols / 2*self.args.submap_resolution ) / self.args.submap_resolution)

		bx = min(max(cx - self.args.submap_width / 2,- self.args.submap_width / 2),self.args.submap_width / 2)
		by = min(max(cy - self.args.submap_height / 2,- self.args.submap_height / 2),self.args.submap_height / 2)

		# Iterate over map
		update_window = 2
		for i in range(max(dx - update_window, 0), min(dx + update_window, rows)):
			for j in range(max(dy - update_window, 0), min(dy + update_window, cols)):
				old_cost = batch_grid[i, j]

				x = bx + i * self.args.submap_resolution
				y = by + j * self.args.submap_resolution
				ma = np.arctan2(y - cy, x - cx)

				diff = self.shortest_angular_distance(angle, ma)

				if (np.abs(diff) < np.pi / 2):
					a = self.getGaussian(x, y, cx, cy, self.covar_ * factor, self.covar_, angle);
				else:
					a = self.getGaussian(x, y, cx, cy, self.covar_, self.covar_, 0);

				if (a > self.cutoff_):
					a = self.cutoff_

				batch_grid[i, j] += a

		return batch_grid

	def add_other_agents_to_grid(self,batch_grid,batch_x,other_agents):
		"""
		This function is heavily based on the proxemic layer ROS implementation
		from https://github.com/DLu/navigation_layers.git
		:param batch_grid:
		:param batch_x:
		:param other_agents:
		:param args:
		:return:
		"""
		# iterate over batch index
		for seq_index in range(batch_grid.shape[0]):
			# iterate over truncated back propagation
			for t_idx in range(batch_grid.shape[1]):
				#print(t_idx)
				#print(batch_grid.shape[1])
				sigma = 2  # assumed position error
				rows = batch_grid.shape[2]
				cols = batch_grid.shape[3]
				other_agents_pos = other_agents[seq_index][t_idx][:,:2]
				other_agents_vel = other_agents[seq_index][t_idx][:,2:4]
				# iterate over surrounding agents
				for agent_id in range(other_agents_pos.shape[0]):
					"""old implmentation
					center_x = int((other_agents_pos[i, 0]-batch_x[seq_index,t_idx,0] + rows / 2*self.args.submap_resolution ) / self.args.submap_resolution)
					center_y = int((other_agents_pos[i, 1]-batch_x[seq_index,t_idx,1] + cols / 2*self.args.submap_resolution ) / self.args.submap_resolution)
					for idx in range(max(center_x - 5, 0), min(center_x + 5, rows)):
						for idy in range(max(center_y - 5, 0), min(center_y + 5, cols)):
							z = 1.0 * np.square(idx - center_x) / np.square(sigma) + 1.0 * np.square(idy - center_y) / np.square(
								sigma)
							pdf = max(0.0, 10.0 / 2.0 / np.pi / np.square(sigma) * np.exp(-z / 2.0))
							batch_grid[seq_index,t_idx,idx, idy] += pdf
							if batch_grid[seq_index, t_idx, idx, idy] > 1.0:
								batch_grid[seq_index, t_idx, idx, idy] = 1.0
					"""

					angle = np.arctan2(other_agents_vel[agent_id,1], other_agents_vel[agent_id,0])
					mag = np.linalg.norm(other_agents_vel[agent_id])
					factor = 1.0 + mag * self.factor_

					# Position relative to query-agent
					cx = other_agents_pos[agent_id, 0]-batch_x[seq_index,t_idx,0]
					cy = other_agents_pos[agent_id, 1]-batch_x[seq_index,t_idx,1]

					# Position in map frame
					dx = 	int((cx + rows / 2*self.args.submap_resolution ) / self.args.submap_resolution)
					dy = int((cy + cols / 2*self.args.submap_resolution ) / self.args.submap_resolution)

					bx = min(max(cx - self.args.submap_width / 2,- self.args.submap_width / 2),self.args.submap_width / 2)
					by = min(max(cy - self.args.submap_height / 2,- self.args.submap_height / 2),self.args.submap_height / 2)

					# Iterate over map
					update_window = 10
					for i in range(max(dx - update_window, 0), min(dx + update_window, rows)):
						for j in range(max(dy - update_window, 0), min(dy + update_window, cols)):
							old_cost = batch_grid[seq_index,t_idx,i, j]

							x = bx + i * self.args.submap_resolution
							y = by + j * self.args.submap_resolution
							ma = np.arctan2(y - cy, x - cx)

							diff = self.shortest_angular_distance(angle, ma)

							if (np.abs(diff) < np.pi / 2):
								a = self.getGaussian(x, y, cx, cy, self.covar_ * factor, self.covar_, angle);
							else:
								a = self.getGaussian(x, y, cx, cy, self.covar_, self.covar_, 0);

							if (a > self.cutoff_):
								a = self.cutoff_

							batch_grid[seq_index,t_idx,i, j] += a

		return batch_grid

	def normalize_angle_positive(self,angle):
		""" Normalizes the angle to be 0 to 2*pi
				It takes and returns radians. """
		return angle % (2.0 * np.pi)

	def normalize_angle(self,angle):
		""" Normalizes the angle to be -pi to +pi
				It takes and returns radians."""
		a = self.normalize_angle_positive(angle)
		if a > np.pi:
			a -= 2.0 * np.pi
		return a

	def shortest_angular_distance(self,from_angle, to_angle):
		""" Given 2 angles, this returns the shortest angular
				difference.  The inputs and ouputs are of course radians.

				The result would always be -pi <= result <= pi. Adding the result
				to "from" will always get you an equivelent angle to "to".
		"""
		return self.normalize_angle(to_angle - from_angle)

	def get_radius(self,cutoff, A,var):

		return np.sqrt(-2 * var * np.log(cutoff / A))

	def getGaussian(self,x,y,x0,y0,varx,vary,skew):

		dx = x - x0
		dy = y - y0

		h = np.linalg.norm(np.array([dx,dy]))
		angle = np.arctan2(dy, dx)
		mx = np.cos(angle - skew) * h
		my = np.sin(angle - skew) * h
		f1 = np.square(mx) / (2.0 * varx)
		f2 = np.square(my) / (2.0 * vary)

		return self.amplitude_ * np.exp(-(f1 + f2))

	def add_traj_to_grid(self,batch_grid,batch_x,batch_y):

		batch_x_rot, batch_y_rot = sup.rotate_batch_to_local_frame(batch_y, batch_x)

		for seq_index in range(batch_x.shape[0]):
			for t in range(batch_x.shape[1]):
				# plot initial and goal pose de-normalized
				x_init_global_frame = batch_x[seq_index, t, 0]*self.norm_const_x
				y_init_global_frame = batch_x[seq_index, t, 1]*self.norm_const_y

				heading = np.squeeze(batch_y_rot[seq_index,t])
				real_traj_global_frame = sup.path_from_vel(initial_pos=np.array([0, 0]),
																									 pred_vel=np.squeeze(batch_y_rot[seq_index,t]),
																									 dt=self.dt, n_vx=self.norm_const_vx, n_vy=self.norm_const_vy)
				for i in range(real_traj_global_frame.shape[0]):
					x = int(max(0,int(np.abs(real_traj_global_frame[i,0]+self.submap_width/2)/self.agent_container.occupancy_grid.resolution)))
					y = int(max(0,int(np.abs(real_traj_global_frame[i, 1]+self.submap_height/2) / self.agent_container.occupancy_grid.resolution)))
					batch_grid[seq_index, t,x,y] = 1

	def plot_angular_grid(self, batch_ped_grid,t,seq_index):
		if self.multi_pedestrian:
			fig = pl.figure("Angular grid")
			ax_ped_grid = pl.subplot()
			ax_ped_grid.clear()
			n_sectors = batch_ped_grid.shape[2]
			grid = batch_ped_grid[seq_index,t,:]
			grid = np.expand_dims(grid, axis=1)
			grid_flipped = np.zeros_like(grid)
			grid_flipped[0:int(self.pedestrian_vector_dim/2)] = grid[-int(self.pedestrian_vector_dim/2):]
			grid_flipped[int(self.pedestrian_vector_dim / 2):] = grid[0:int(self.pedestrian_vector_dim / 2)]
			sup.plot_radial_distance_vector(ax_ped_grid, grid_flipped, max_range=1.0, min_angle=0.0,
																			max_angle=2 * np.pi)
			ax_ped_grid.plot(30, 30, color='r', marker='o', markersize=4)
			ax_ped_grid.arrow(0, 0, 1, 0, head_width=0.1,head_length=self.max_range_ped_grid) #agent poiting direction
		# x- and y-range only need to be [-1, 1] since the pedestrian grid is normalized
			ax_ped_grid.set_xlim([-self.max_range_ped_grid-1, self.max_range_ped_grid+1])
			ax_ped_grid.set_ylim([-self.max_range_ped_grid-1, self.max_range_ped_grid+1])

			fig.canvas.draw()
			# sleep(0.5)  # Time in seconds.
			pl.show(block=False)
			sleep(0.5)  # Time in seconds.

	def query_agent_velocity_direction(self, batch_grid, batch_x, batch_agent_real_traj,other_agents_pos,t,seq_index):
		"""
			inputs:
				global_grid: map grid of the scenario
				batch_grid: batch_grid numpy array (gridmap) in the global frame [Batch_size][truncated_back_propagation_time][width][height]
				batch_agent_real_traj: agent ground truth velocity in global frame
				batch_agent_predicted_traj: agent predicted trajectory in global frame
			"""
		#pl.close('all')
		global_grid = self.agent_container.occupancy_grid
		fig = pl.figure("Query agent direction")
		ax_in = pl.subplot()

		#color definitions
		colormap = pl.get_cmap('rainbow')
		c_norm = pl.matplotlib.colors.Normalize(vmin=0, vmax=100)
		scalar_color_map = pl.cm.ScalarMappable(norm=c_norm, cmap=colormap)
		r= random.randint(0,100)

		number_samples =1
		ax_in.clear()
		#clear plots
		#plot scenario grid
		sup.plot_grid(ax_in, np.array([0.0, 0.0]), global_grid.gridmap, global_grid.resolution, global_grid.map_size)
		ax_in.set_xlim([-global_grid.center[0], global_grid.center[0]])
		ax_in.set_ylim([-global_grid.center[1], global_grid.center[1]])

		#plot initial and goal pose
		x_init_global_frame = batch_x[seq_index,t, 0]*self.norm_const_x
		y_init_global_frame = batch_x[seq_index,t, 1]*self.norm_const_y
		ax_in.plot(x_init_global_frame, y_init_global_frame, color='g', marker='o', label='start')

		#plot real trajectory global frame
		color_value = scalar_color_map.to_rgba(r)

		#Plot velocity direction
		ax_in.arrow(x_init_global_frame, y_init_global_frame, batch_x[seq_index,t, 2]*self.norm_const_vx, batch_x[seq_index,t, 3]*self.norm_const_vy
				, head_width=0.1, head_length=self.max_range_ped_grid+0.5,alpha=0.4,label='Agent velocity direction') # angular grid detection distance
		circle = pl.Circle((x_init_global_frame,y_init_global_frame),self.max_range_ped_grid,color='blue',fill=False,label='Detection range')
		ax_in.add_artist(circle)
		#Plot other agent positions
		if self.multi_pedestrian:
			other_pos = other_agents_pos[seq_index][t]
			for jj in range(other_pos.shape[0]):
				ax_in.plot(other_pos[jj, 0], other_pos[jj, 1], color='r', marker='o') #other agent pos positon

		ax_in.set_aspect("equal")
		fig.canvas.draw()
		#sleep(0.5)  # Time in seconds.
		pl. show(block=False)
