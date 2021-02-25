import os
import numpy as np
import cv2
import pylab as pl
from matplotlib.animation import FFMpegWriter
import sys
if sys.version_info[0] < 3:
	import Support as sup
else:
	import src.data_utils.Support as sup
import math
from matplotlib.patches import Ellipse
from scipy import interpolate

def rgba2rgb(rgba):
    # rgba is a list of 4 color elements btwn [0.0, 1.0]
    # or a 2d np array (num_colors, 4)
    # returns a list of rgb values between [0.0, 1.0] accounting for alpha and background color [1, 1, 1] == WHITE
    if isinstance(rgba, list):
        alpha = rgba[3]
        r = max(min((1 - alpha) * 1.0 + alpha * rgba[0],1.0),0.0)
        g = max(min((1 - alpha) * 1.0 + alpha * rgba[1],1.0),0.0)
        b = max(min((1 - alpha) * 1.0 + alpha * rgba[2],1.0),0.0)
        return [r,g,b]
    elif rgba.ndim == 2:
        alphas = rgba[:,3]
        r = np.clip((1 - alphas) * 1.0 + alphas * rgba[:,0], 0, 1)
        g = np.clip((1 - alphas) * 1.0 + alphas * rgba[:,1], 0, 1)
        b = np.clip((1 - alphas) * 1.0 + alphas * rgba[:,2], 0, 1)
        return np.vstack([r,g,b]).T

class Recorder():

	def __init__(self,args,grid):

		# Parameters
		self.args = args
		self.gridmap = grid
		# Animation
		self.fig_animate = pl.figure('Animation')
		self.fig_width = 12  # width in inches
		self.fig_height = 12  # height in inches
		self.fig_size = [self.fig_width, self.fig_height]
		self.fontsize = 9
		self.colors = ['r', 'g', 'b','y','m','c','k']

		self.params = {'backend': 'ps',
		          'axes.labelsize': self.fontsize,
		          'font.size': self.fontsize,
		          'xtick.labelsize': self.fontsize,
		          'ytick.labelsize': self.fontsize,
		          'figure.figsize': self.fig_size}
		pl.rcParams.update(self.params)
		self.ax_pos = pl.subplot()
		#self.ax_pos_local = pl.subplot2grid((6, 12), (0, 6), colspan=6, rowspan=6)
		#self.ax_grid = pl.subplot2grid((9, 12), (7, 0), colspan=2, rowspan=2)
		#self.ax_ped_grid = pl.subplot2grid((9, 12), (7, 2), colspan=2, rowspan=2)
		#self.ax_vel = pl.subplot2grid((9, 12), (7, 4), colspan=2, rowspan=2)
		pl.rcParams.update(self.params)
		#pl.rcParams['animation.ffmpeg_path'] = "/home/bbrito/ffmpeg/ffmpeg"

		metadata = dict(title='Movie Test', artist='Matplotlib')
		self.writer = FFMpegWriter(fps=10, metadata=metadata,codec='mpeg4')

	def animate_local(self,input_list,grid_list,ped_grid_list,y_pred_list_global,y_ground_truth_list,other_agents_list,trajectories,test_args):

		fig_animate = pl.figure('Local Scenario')
		fig_width = 12  # width in inches
		fig_height = 10  # height in inches
		fig_size = [fig_width, fig_height]
		fontsize = 26
		colors = ['r', 'g', 'y']
		params = {'backend': 'ps',
		               'axes.labelsize': fontsize,
		               'font.size': fontsize,
		               'xtick.labelsize': fontsize,
		               'ytick.labelsize': fontsize,
		               'figure.figsize': fig_size}

		pl.rcParams.update(params)
		ax_pos = pl.subplot()

		for animation_idx in range(len(input_list)):
			input = input_list[animation_idx]
			grid = grid_list[animation_idx]

			traj = trajectories[animation_idx]
			gt_vel = y_ground_truth_list[animation_idx]

			if not (y_pred_list_global is None):
				model_vel_pred = y_pred_list_global[animation_idx]
			else:
				model_vel_pred = None
			other_agents_pos = other_agents_list[animation_idx]

			for step in range(input.shape[0]): # trajectory length considering data from getTrajectoryAsBatch

				self.plot_local_scenario(ax_pos,input, grid, model_vel_pred[step], gt_vel, other_agents_pos, step,
					                    rotate=False,n_samples=test_args.n_samples)

				ax_pos.axis("on")
				ax_pos.set_xlabel('x [m]',fontsize=26)
				ax_pos.set_ylabel('y [m]',fontsize=26)
				ax_pos.set_xlim([-6.0, 6.0])
				ax_pos.set_ylim([-6.0, 6.0])
				ax_pos.set_aspect('equal')
				pl.rcParams.update(params)
				ax_pos.set_aspect('equal')
				if not os.path.exists(self.args.model_path + '/results/' + self.args.scenario+"/figs/"):
					os.makedirs(self.args.model_path + '/results/' + self.args.scenario+"/figs/")
				if test_args.save_figs:
					fig_animate.savefig(self.args.model_path + '/results/' + self.args.scenario+"/figs/local_"+str(animation_idx)+"_"+str(step)+".jpg")

	def animate(self,input_list,grid_list,ped_grid_list,y_pred_list_global,y_ground_truth_list,other_agents_list,trajectories,test_args):

		#pl.show(block=False)
		if not os.path.exists(self.args.model_path + '/../videos/'):
			os.makedirs(self.args.model_path + '/../videos/')
		if test_args.unit_testing:
			video_file = self.args.model_path + '/../videos/' + str(self.args.exp_num) + "_unit_tests.mp4"
		elif test_args.freeze_other_agents:
			video_file = self.args.model_path + '/../videos/' + str(self.args.exp_num) + "_frozen.mp4"
		else:
			video_file = self.args.model_path + '/../videos/' + str(self.args.exp_num) + "_final.mp4"
		print("Recoding to: " + video_file)
		with self.writer.saving(self.fig_animate, video_file, 100):
			for animation_idx in range(len(input_list)):
				input = input_list[animation_idx]
				grid = grid_list[animation_idx]
				ped_grid = ped_grid_list[animation_idx]
				traj = trajectories[animation_idx]
				gt_vel = y_ground_truth_list[animation_idx]

				if not (y_pred_list_global is None):
					model_vel_pred = y_pred_list_global[animation_idx]
				else:
					model_vel_pred = None
				other_agents_pos = other_agents_list[animation_idx]

				for step in range(input.shape[0]): # trajectory length considering data from getTrajectoryAsBatch
					# Eucldean / position space
					self.ax_pos.clear()
					#self.ax_pos.plot(input[:, 0], input[:, 1], color='c', alpha=0.4, lw=1)
					self.ax_pos.plot(traj.pose_vec[-1, 0], traj.pose_vec[-1, 1], color='g', marker='o')
					#Real Trajectory
					self.ax_pos.plot(traj.pose_vec[:, 0], traj.pose_vec[:, 1], color='b', lw=2)

					self.ax_pos.plot(input[step, 0], input[step, 1], color='c', marker='o')
					bottom_left_submap = input[step, 0:2] - np.array([self.args.submap_width / 2.0, self.args.submap_height / 2.0])
					self.ax_pos.add_patch(pl.Rectangle(bottom_left_submap, self.args.submap_width, self.args.submap_height, fill=None))
					self.plot_local_scenario(input, grid, model_vel_pred[step], gt_vel, other_agents_pos, step,
					                    rotate=False,n_samples=test_args.n_samples)

					# Other agents
					for id in range(other_agents_pos[step].shape[0]): # number of agents
						self.ax_pos.plot(other_agents_pos[step][id, 0], other_agents_pos[step][id, 1], marker='o', color='b')
					vel_pred = np.zeros((self.args.prediction_horizon,  self.args.output_dim))
					sigmax = np.zeros((self.args.prediction_horizon, 1))
					sigmay = np.zeros((self.args.prediction_horizon, 1))
					#Predicted trajectory
					if not (y_pred_list_global is None):
						if self.args.n_mixtures == 0:
							# plot predicted trajectory global frame
							prediction_sample = model_vel_pred[step][0]
							for i in range(self.args.prediction_horizon):
								idx = i * self.args.output_pred_state_dim
								idy = i * self.args.output_pred_state_dim + 1
								mu_x = prediction_sample[0, idx]
								mu_y = prediction_sample[0, idy]
								vel_pred[i, :] = [mu_x, mu_y]

							pred_vel_global_frame = vel_pred
							traj_pred = sup.path_from_vel(initial_pos=np.array(
								[input[step, 0],
								 input[step, 1]]),
								pred_vel=pred_vel_global_frame, dt=self.args.dt)
							self.ax_pos.plot(traj_pred[:, 0], traj_pred[:, 1], color=self.colors[0], label='Predicted trajectory')
						else:
							for sample_id in range(test_args.n_samples):
								prediction_sample = model_vel_pred[step][sample_id]
								for mix_idx in range(self.args.n_mixtures):
									# plot predicted trajectory global frame
									for pred_step in range(self.args.prediction_horizon):
										idx = pred_step * self.args.output_pred_state_dim * self.args.n_mixtures + mix_idx
										mu_x = prediction_sample[0, idx]
										mu_y = prediction_sample[0, idx + self.args.n_mixtures]
										if self.args.output_pred_state_dim>2:
											sigmax[pred_step, :] = prediction_sample[0][idx + 2 * self.args.n_mixtures]
											sigmay[pred_step, :] = prediction_sample[0][idx + 3 * self.args.n_mixtures]
										vel_pred[pred_step, :] = [mu_x, mu_y]

									pred_vel_global_frame = vel_pred
									traj_pred = sup.path_from_vel(initial_pos=np.array(
										[input[step, 0],
										 input[step, 1]]),
										pred_vel=pred_vel_global_frame, dt=self.args.dt)

									self.ax_pos.plot(traj_pred[:, 0], traj_pred[:, 1], color=self.colors[mix_idx], label='Predicted trajectory')

									if self.args.output_pred_state_dim > 2:
										# prior of 0 on the uncertainty of the pedestrian velocity
										sigma_x = np.square(sigmax[0]) * self.args.dt * self.args.dt
										sigma_y = np.square(sigmay[0]) * self.args.dt * self.args.dt

										e1 = Ellipse(xy=(traj_pred[0, 0], traj_pred[0, 1]), width=np.sqrt(sigma_x) / 2,
										             height=np.sqrt(sigma_y) / 2, angle=0 / np.pi * 180)
										e1.set_alpha(0.5)
										self.ax_pos.add_patch(e1)
										#self.ax_pos.plot(traj_pred[:, 0], traj_pred[:, 1], color=self.colors[mix_idx], label='Predicted trajectory')
										for pred_step in range(1, self.args.prediction_horizon):
											sigma_x += np.square(sigmax[pred_step]) * self.args.dt * self.args.dt
											sigma_y += np.square(sigmay[pred_step]) * self.args.dt * self.args.dt
											e1 = Ellipse(xy=(traj_pred[pred_step, 0], traj_pred[pred_step, 1]), width=np.sqrt(sigma_x) / 2,
											             height=np.sqrt(sigma_y) / 2,
											             angle=0 / np.pi * 180)
											self.ax_pos.add_patch(e1)

					# Real trajectory
					model_vel_real = np.zeros((self.args.prediction_horizon, self.args.output_dim))
					for i in range(self.args.prediction_horizon):
						idx = i * self.args.output_dim
						idy = i * self.args.output_dim + 1
						mu_x = gt_vel[step, idx]
						mu_y = gt_vel[step, idy]
						model_vel_real[i, :] = [mu_x, mu_y]
					if False:
						real_vel = sup.rotate_predicted_vel_to_global_frame(model_vel_real, batch_x[step, 2:])
					else:
						real_vel = model_vel_real
					traj_real = sup.path_from_vel(initial_pos=np.array([0, 0]),
					                              pred_vel=real_vel, dt=self.args.dt)
					self.ax_pos_local.plot(traj_real[:, 0], traj_real[:, 1], color='m', label='Agent real trajectory')

					sup.plot_grid(self.ax_pos, np.array([0.0, 0.0]), self.gridmap.gridmap, self.gridmap.resolution,
					              self.gridmap.map_size)

					ax_pos.axis('equal')
					self.ax_pos.axis("on")
					self.ax_pos.set_xlabel('x [m]')
					self.ax_pos.set_ylabel('y [m]')
					self.ax_pos.set_aspect('equal')

					#self.fig_animate.canvas.draw()
					if not os.path.exists(self.args.model_path + '/results/' + self.args.scenario+"/figs/"):
						os.makedirs(self.args.model_path + '/results/' + self.args.scenario+"/figs/")
					if test_args.save_figs:
						self.fig_animate.savefig(self.args.model_path + '/results/' + self.args.scenario+"/figs/result_"+str(animation_idx)+"_"+str(step)+".jpg")
					self.writer.grab_frame()

	def animate_group_of_agents(self,trajectories,y_pred_list_global,test_args=None):

		if not os.path.exists(self.args.model_path + '/videos/'):
			os.makedirs(self.args.model_path + '/videos/')

		fig_animate = pl.figure('Global Scenario')
		fig_width = 12  # width in inches
		fig_height = 10  # height in inches
		fig_size = [fig_width, fig_height]
		fontsize = 26

		params = {'backend': 'ps',
		          'axes.labelsize': fontsize,
		          'font.size': fontsize,
		          'xtick.labelsize': fontsize,
		          'ytick.labelsize': fontsize,
		          'figure.figsize': fig_size}

		plt_colors = []
		plt_colors.append([0.8500, 0.3250, 0.0980])  # orange
		plt_colors.append([0.0, 0.4470, 0.7410])  # blue
		plt_colors.append([0.4660, 0.6740, 0.1880])  # green
		plt_colors.append([0.4940, 0.1840, 0.5560])  # purple
		plt_colors.append([0.9290, 0.6940, 0.1250])  # yellow
		plt_colors.append([0.3010, 0.7450, 0.9330])  # cyan
		plt_colors.append([0.6350, 0.0780, 0.1840])  # chocolate
		plt_colors.append([0.6350, 0.0780, 0.1840])  # chocolate
		plt_colors.append([0.6350, 0.0780, 0.1840])  # chocolate
		scenario = self.args.scenario.split('/')[0]
		ax_pos = pl.subplot()

		ax_pos.set_xlabel('x [m]')
		ax_pos.set_ylabel('y [m]')
		ax_pos.axis('on')
		ax_pos.set_aspect('equal')

		n_agents =  len(trajectories[0])
		axbackground = fig_animate.canvas.copy_from_bbox(ax_pos.bbox)

		# Create plotting variables
		current_pos_list_ = []
		pred_line_list_ = []
		real_line_list_ = []
		ellipses_list_ = []
		for ag_id in range(n_agents):
			current_pos_, = ax_pos.plot([], color=plt_colors[ag_id], marker='o')
			current_pos_list_.append(current_pos_)
			pred_line, = ax_pos.plot([], color=plt_colors[ag_id], label='Predicted trajectory Agent ' + str(ag_id))
			pred_line_list_.append(pred_line)
			real_line, = ax_pos.plot([], color=plt_colors[ag_id], alpha=0.5, label='Real trajectory Agent ' + str(ag_id))
			real_line_list_.append(real_line)
			c = rgba2rgb(plt_colors[ag_id] + [float(1)])
			ellipses = [Ellipse(xy=(0, 0), width=0.6,
			                    height=0.6,
			                    fc=c, ec=plt_colors[ag_id],
			                    fill=True) for pred_step in range(self.args.prediction_horizon)]
			for ellipse in ellipses:
				ax_pos.add_artist(ellipse)
			ellipses_list_.append(ellipses)

		ax_pos.legend()

		with self.writer.saving(fig_animate, self.args.model_path + '/videos/' +str(self.args.exp_num)+scenario+".mp4", 100):
			for animation_idx in range(len(trajectories)):
				traj = trajectories[animation_idx]

				model_vel_pred = y_pred_list_global[animation_idx]

				vel_pred = np.zeros((self.args.prediction_horizon, self.args.output_pred_state_dim))
				sigmax = np.zeros((self.args.prediction_horizon, 1))
				sigmay = np.zeros((self.args.prediction_horizon, 1))

				ax_pos.set_xlim([0.0, 20.0])
				ax_pos.set_ylim([-5.0, 5.0])

				for step in range(traj[0].time_vec.shape[0]):
					for ag_id in range(n_agents):
					  # trajectory length considering data from getTrajectoryAsBatch
						fig_animate.canvas.restore_region(axbackground)

						try:
							current_pos_list_[ag_id].set_data(traj[ag_id].pose_vec[step,0]/self.args.sx_pos,traj[ag_id].pose_vec[step, 1]/self.args.sy_pos)
						except:
							current_pos_list_[ag_id].set_data(traj[ag_id].pose_vec[-1, 0] / self.args.sx_pos,
							                                  traj[ag_id].pose_vec[-1, 1] / self.args.sy_pos)
						#Predicted trajectory
						for sample_id in range(1):
							try:
								prediction_sample = model_vel_pred[ag_id][step][sample_id]
								init_pose = traj[ag_id].pose_vec[step,:2]
							except:
								prediction_sample =  np.zeros([self.args.output_dim*self.args.prediction_horizon])
								init_pose = traj[ag_id].pose_vec[-1, :2]
							for mix_idx in range(self.args.n_mixtures):
								# plot predicted trajectory global frame
								for pred_step in range(self.args.prediction_horizon):
									idx = pred_step * self.args.output_pred_state_dim * self.args.n_mixtures + mix_idx
									mu_x = prediction_sample[idx]/self.args.sx_vel
									mu_y = prediction_sample[idx + self.args.n_mixtures]/self.args.sy_vel
									if self.args.output_pred_state_dim > 2:
										sigmax[pred_step, :] = prediction_sample[idx + 2 * self.args.n_mixtures]
										sigmay[pred_step, :] = prediction_sample[idx + 3 * self.args.n_mixtures]
									vel_pred[pred_step, :] = [mu_x, mu_y]

								pred_vel_global_frame = vel_pred
								traj_pred = sup.path_from_vel(initial_pos=init_pose/self.args.sx_pos,
								                              pred_vel=pred_vel_global_frame, dt=self.args.dt)

								pred_line_list_[ag_id].set_data(traj_pred[:, 0], traj_pred[:, 1])

								if self.args.output_pred_state_dim > 2:
									# TODO: ADAPT TO NEW CODE
									# prior of 0 on the uncertainty of the pedestrian velocity
									sigma_x = np.square(sigmax[0]) * self.args.dt * self.args.dt
									sigma_y = np.square(sigmay[0]) * self.args.dt * self.args.dt

									e1 = Ellipse(xy=(traj_pred[0, 0], traj_pred[0, 1]), width=np.sqrt(sigma_x) / 2,
									             height=np.sqrt(sigma_y) / 2, angle=0 / np.pi * 180)
									e1.set_alpha(0.5)

									for pred_step in range(1, self.args.prediction_horizon):
										sigma_x += np.square(sigmax[pred_step]) * self.args.dt * self.args.dt
										sigma_y += np.square(sigmay[pred_step]) * self.args.dt * self.args.dt

										e1 = Ellipse(xy=(traj_pred[pred_step, 0], traj_pred[pred_step, 1]), width=np.sqrt(sigma_x) / 2,
										             height=np.sqrt(sigma_y) / 2,
										             angle=0 / np.pi * 180)
										ax_pos.add_patch(e1)
								else:
									for pred_step in range(0, self.args.prediction_horizon):
										alpha = 1/ (1+pred_step)
										c = rgba2rgb(plt_colors[ag_id] + [float(alpha)])
										ellipses_list_[ag_id][pred_step].set_facecolor(c)
										ellipses_list_[ag_id][pred_step].set_center((traj_pred[pred_step, 0], traj_pred[pred_step, 1]))

						# Real trajectory
						real_line_list_[ag_id].set_data(traj[ag_id].pose_vec[:,0]/self.args.sx_pos, traj[ag_id].pose_vec[:,1]/self.args.sy_pos)

					fig_animate.canvas.blit(ax_pos.bbox)
					self.writer.grab_frame()

					if not os.path.exists(self.args.model_path + '/results/' + self.args.scenario + "/figs/"):
						os.makedirs(self.args.model_path + '/results/' + self.args.scenario + "/figs/")
					if test_args.save_figs:
						fig_animate.savefig(self.args.model_path + '/results/' + self.args.scenario + "/figs/result_" + str(
							animation_idx) + "_" + str(step) + ".jpg")

					fig_animate.canvas.flush_events()

	def animate_global(self,input_list,grid_list,y_pred_list_global,y_ground_truth_list,other_agents_list,y_ground_truth_list_exp,all_traj_likelihood,test_args=None):

		if not os.path.exists(self.args.model_path + '/results/' + self.args.scenario):
			os.makedirs(self.args.model_path + '/results/' + self.args.scenario)

		fig_animate = pl.figure('Global Scenario')
		fig_animate.tight_layout()
		fig_width = 12  # width in inches
		fig_height = 10  # height in inches
		fig_size = [fig_width, fig_height]
		fontsize = 26

		params = {'backend': 'ps',
		          'axes.labelsize': fontsize,
		          'font.size': fontsize,
		          'xtick.labelsize': fontsize,
		          'ytick.labelsize': fontsize,
		          'figure.figsize': fig_size,
		          'legend.loc': 'upper right',
		          'legend.fontsize': 14}

		pl.rcParams.update(params)

		plt_colors = []
		plt_colors.append([0.8500, 0.3250, 0.0980])  # orange
		plt_colors.append([0.0, 0.4470, 0.7410])  # blue
		scenario = self.args.scenario.split('/')[0]
		ax_pos = pl.subplot()

		ax_pos.set_xlabel('x [m]')
		ax_pos.set_ylabel('y [m]')
		ax_pos.axis('on')
		ax_pos.set_aspect('equal')

		# Load Map
		"""
		map_file = os.path.join(self.args.data_path + self.args.scenario, 'map.png')
		if os.path.exists(map_file):
			#img = np.uint8(cv2.imread(map_file) * -1 + 255)
			img = cv2.imread(map_file)
			#img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
			resolution = 0.1
			width = int(img.shape[1] * resolution)
			height = int(img.shape[0] * resolution)
			dim = (width, height)
			resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
		else:
			print('[INF] No map file')
		ax_pos.imshow(resized,extent=[0, 30, -3, 2])
		"""
		sup.plot_grid(ax_pos, np.array([0.0, 0.0]), self.gridmap.gridmap, self.gridmap.resolution,
		              self.gridmap.map_size)

		axbackground = fig_animate.canvas.copy_from_bbox(ax_pos.bbox)
		current_pos_, = ax_pos.plot([], color='b', marker='o')
		other_agents_line_ = []
		pred_line_list = []
		for i in range(self.args.n_mixtures):
			pred_line, = ax_pos.plot([], color=self.colors[2], label='Predicted trajectory Agent ' + str(i))
			pred_line_list.append(pred_line)

		real_line, = ax_pos.plot([], color='c',marker='.',alpha=0.5, label='Real trajectory Agent 1')
		c = rgba2rgb(plt_colors[1] + [float(1)])

		ellipses = []
		for i in range(self.args.n_mixtures):
			ellipse = [Ellipse(xy=(0,0), width=0.6,
				             height=0.6,
				             fc=c, ec=plt_colors[1],
				             fill=True) for pred_step in range(self.args.prediction_horizon)]
			ellipses.append(ellipse)

		for ellipse in ellipses:
			for el in ellipse:
				ax_pos.add_artist(el)

		ax_pos.legend()

		with self.writer.saving(fig_animate, self.args.model_path + '/results/' + self.args.scenario+"/video.mp4", 100):
			for animation_idx in range(0,len(input_list)):
				input = input_list[animation_idx]
				grid = grid_list[animation_idx]

				#ax_pos.set_xlim([int(np.min(input/self.args.sx_pos))-2.0, int(np.max(input/self.args.sx_pos))+2.0])
				#ax_pos.set_ylim([int(np.min(input/self.args.sy_pos))-2.0, int(np.max(input/self.args.sy_pos))+2.0])
				ax_pos.set_xlim([-self.gridmap.map_size[0]/2, self.gridmap.map_size[0]/2])
				ax_pos.set_ylim([-self.gridmap.map_size[1]/2, self.gridmap.map_size[1]/2])

				if not (y_pred_list_global is None):
					model_vel_pred = y_pred_list_global[animation_idx]
				else:
					model_vel_pred = None
				other_agents_pos = other_agents_list[animation_idx]
				traj_likelihood = all_traj_likelihood[animation_idx]

				number_of_agents = len(other_agents_pos[0])
				for ag_id in range(number_of_agents):
					other_agent, = ax_pos.plot([], color='r', marker='o')
					other_agents_line_.append(other_agent)
				vel_pred = np.zeros((self.args.prediction_horizon, 2))
				sigmax = np.zeros((self.args.prediction_horizon))
				sigmay = np.zeros((self.args.prediction_horizon))

				for step in range(input.shape[0]): # trajectory length considering data from getTrajectoryAsBatch
					fig_animate.canvas.restore_region(axbackground)
					# Eucldean / position space
					#ax_pos.plot(input[:, 0], input[:, 1], color='b', alpha=0.4, lw=1)
					#ax_pos.plot(input[:step, 0], input[:step, 1], color='b', lw=2)
					current_pos_.set_data(input[step, 0]/self.args.sx_pos, input[step, 1]/self.args.sy_pos)
					#ax_pos.add_patch(pl.Rectangle(bottom_left_submap, self.args.submap_width, self.args.submap_height, fill=None))

					# Other agents
					for id in range(other_agents_pos[step].shape[0]): # number of agents
						other_agents_line_[id].set_data(other_agents_pos[step][id, 0], other_agents_pos[step][id, 1])
					#Predicted trajectory
					if not (y_pred_list_global is None):
						if self.args.n_mixtures == 0:
							# plot predicted trajectory global frame
							for i in range(self.args.prediction_horizon):
								idx = i * self.args.output_pred_state_dim
								idy = i * self.args.output_pred_state_dim + 1
								mu_x = model_vel_pred[step, idx]/self.args.sx_vel
								mu_y = model_vel_pred[step, idy]/self.args.sy_vel
								vel_pred[0, 2 * i:2 * i + 2] = [mu_x, mu_y]

							pred_vel_global_frame = vel_pred
							traj_pred = sup.path_from_vel(initial_pos=np.array(
								[input[step, 0],
								 input[step, 1]]),
								pred_vel=np.squeeze(pred_vel_global_frame), dt=self.args.dt)
							pred_line_list[mix_idx].set_data(traj_pred[:, 0], traj_pred[:, 1])
						else:
							for sample_id in range(1):
								prediction_sample = model_vel_pred[step][sample_id]
								most_likely_traj = np.argmax(traj_likelihood[step][0,0])
								for mix_idx in range(self.args.n_mixtures):
									# plot predicted trajectory global frame
									for pred_step in range(self.args.prediction_horizon):
										idx = pred_step * self.args.output_pred_state_dim * self.args.n_mixtures + mix_idx
										mu_x = prediction_sample[0, idx]/self.args.sx_vel
										mu_y = prediction_sample[0, idx + self.args.n_mixtures]/self.args.sy_vel
										if self.args.output_pred_state_dim > 2:
											sigmax[pred_step] = np.maximum(np.minimum(prediction_sample[0, idx + 2 * self.args.n_mixtures],0.5),0.05)
											sigmay[pred_step] = np.maximum(np.minimum(prediction_sample[0, idx + 3 * self.args.n_mixtures],0.5),0.05)
										vel_pred[pred_step, :] = [mu_x, mu_y]

									pred_vel_global_frame = vel_pred
									traj_pred = sup.path_from_vel(initial_pos=input[step,:2]/self.args.sx_pos,
									                              pred_vel=pred_vel_global_frame, dt=self.args.dt)
									# hack for prediciting positions
									#traj_pred = vel_pred+input[step,:2]
									if mix_idx != most_likely_traj:
										pred_line_list[mix_idx].set_data(traj_pred[:, 0], traj_pred[:, 1])
										pred_line_list[mix_idx].set_color(self.colors[2])
									else:
										pred_line_list[mix_idx].set_data(traj_pred[:, 0], traj_pred[:, 1])
										pred_line_list[mix_idx].set_color(self.colors[1])
									pred_line_list[mix_idx].set_label('Predicted trajectory ' + str(traj_likelihood[step][0,0,mix_idx]))

									if self.args.output_pred_state_dim > 2:
										# prior of 0 on the uncertainty of the pedestrian velocity
										sigma_x = np.square(sigmax[0]) * self.args.dt * self.args.dt
										sigma_y = np.square(sigmay[0]) * self.args.dt * self.args.dt

										alpha = 1
										c = rgba2rgb(plt_colors[0] + [float(alpha)])
										ellipses[mix_idx][0].set_facecolor(c)
										ellipses[mix_idx][0].set_center((traj_pred[0, 0], traj_pred[0, 1]))
										ellipses[mix_idx][0].width = np.sqrt(sigma_x)
										ellipses[mix_idx][0].height = np.sqrt(sigma_y)
										for pred_step in range(1, self.args.prediction_horizon):
											sigma_x += np.square(sigmax[pred_step]) * self.args.dt * self.args.dt
											sigma_y += np.square(sigmay[pred_step]) * self.args.dt * self.args.dt
											#sigma_x += sigmax[pred_step]
											#sigma_y += sigmay[pred_step]

											alpha = 1 / (1 + pred_step)
											c = rgba2rgb(plt_colors[0] + [float(alpha)])
											ellipses[mix_idx][pred_step].set_facecolor(c)
											ellipses[mix_idx][pred_step].set_center((traj_pred[pred_step, 0], traj_pred[pred_step, 1]))
											ellipses[mix_idx][pred_step].width = np.sqrt(sigma_x)
											ellipses[mix_idx][pred_step].height = np.sqrt(sigma_y)
									else:
										for pred_step in range(0, self.args.prediction_horizon):
											alpha = 1/ (1+pred_step)
											c = rgba2rgb(plt_colors[1] + [float(alpha)])
											ellipses[mix_idx][pred_step].set_facecolor(c)
											ellipses[mix_idx][pred_step].set_center((traj_pred[pred_step, 0], traj_pred[pred_step, 1]))

					# Real trajectory
					vel_real = np.zeros((self.args.prediction_horizon, self.args.output_dim))

					# plot trajectory global frame
					real_line.set_data(input[:, 0]/self.args.sx_pos, input[:, 1]/self.args.sy_pos)

					#sup.plot_grid(ax_pos, np.array([0.0, 0.0]), self.gridmap.gridmap, self.gridmap.resolution,
					#              self.gridmap.map_size)
					fig_animate.set_tight_layout(True)
					fig_animate.canvas.blit(ax_pos.bbox)

					self.writer.grab_frame()

					if not os.path.exists(self.args.model_path + '/results/' + self.args.scenario + "/figs/"):
						os.makedirs(self.args.model_path + '/results/' + self.args.scenario + "/figs/")
					if test_args.save_figs:
						fig_animate.savefig(self.args.model_path + '/results/' + self.args.scenario + "/figs/result_" + str(
							animation_idx) + "_" + str(step) + ".jpg")

					fig_animate.canvas.flush_events()

	def plot_local_scenario(self,ax_pos_local, batch_x, batch_grid, model_vel_pred, model_vel_real, other_agents_pos,step,
	            rotate=False,n_samples=1):

		"""
			inputs:
				global_grid: map grid of the scenario
				batch_grid: batch_grid numpy array (gridmap) in the global frame [Batch_size][truncated_back_propagation_time][width][height]
				batch_agent_real_traj: agent ground truth trajectory in global frame
				model_vel_pred: agent predicted trajectory in global frame
			"""
		ax_pos_local.clear()

		#plot initial and goal pose
		x_init_global_frame = batch_x[step, 0]
		y_init_global_frame = batch_x[step, 1]

		# plot initial and goal pose
		ax_pos_local.plot(0, 0, color='c', marker='o', label='Agent initial pos')

		# Local Frame Goal Position
		x_goal = batch_x[-1, 0] - x_init_global_frame
		y_goal = batch_x[-1, 1] - y_init_global_frame
		#self.ax_pos_local.plot(x_goal, y_goal, color='g', marker='o', label='Agent goal pos')

		# plot predicted trajectory local frame
		if not (model_vel_pred is None):
			vel_pred = np.zeros((self.args.prediction_horizon, self.args.output_dim))
			sigmax = np.zeros((self.args.prediction_horizon, 1))
			sigmay = np.zeros((self.args.prediction_horizon, 1))
			if self.args.n_mixtures == 0:
				# plot predicted trajectory global frame
				for i in range(self.args.prediction_horizon):
					idx = i * self.args.output_pred_state_dim
					idy = i * self.args.output_pred_state_dim + 1
					mu_x = model_vel_pred[0][0, idx]
					mu_y = model_vel_pred[0][0, idy]
					vel_pred[i, :] = [mu_x, mu_y]
				if rotate:
					pred_vel_global_frame = sup.rotate_predicted_vel_to_global_frame(vel_pred, batch_x[step, 2:])
				else:
					pred_vel_global_frame = vel_pred
				traj_pred = sup.path_from_vel(initial_pos=np.array([0,0]),
					pred_vel=pred_vel_global_frame, dt=self.args.dt)
				ax_pos_local.plot(traj_pred[:, 0], traj_pred[:, 1], color=self.colors[0], label='Predicted trajectory')
			else:
				for sample_id in range(n_samples):
					prediction_sample = model_vel_pred[sample_id]
					for mix_idx in range(self.args.n_mixtures):
						# plot predicted trajectory global frame
						for pred_step in range(self.args.prediction_horizon):
							idx = pred_step * self.args.output_pred_state_dim * self.args.n_mixtures + mix_idx
							mu_x = prediction_sample[0, idx]
							mu_y = prediction_sample[0, idx + self.args.n_mixtures]
							if self.args.output_pred_state_dim > 2:
								sigmax[pred_step, :] = prediction_sample[0, idx + 2 * self.args.n_mixtures]
								sigmay[pred_step, :] = prediction_sample[0, idx + 3 * self.args.n_mixtures]
							vel_pred[pred_step, :] = [mu_x, mu_y]

						if rotate:
							pred_vel_global_frame = sup.rotate_predicted_vel_to_global_frame(vel_pred, batch_x[step, 2:])
						else:
							pred_vel_global_frame = vel_pred
						traj_pred = sup.path_from_vel(initial_pos=np.array([0,0]),
							pred_vel=pred_vel_global_frame, dt=self.args.dt)

						ax_pos_local.plot(traj_pred[:, 0], traj_pred[:, 1], color=self.colors[mix_idx],
							                       label='Predicted trajectory')

						if self.args.output_pred_state_dim > 2:
							# prior of 0 on the uncertainty of the pedestrian velocity
							sigma_x = np.square(sigmax[0]) * self.args.dt * self.args.dt
							sigma_y = np.square(sigmay[0]) * self.args.dt * self.args.dt

							e1 = Ellipse(xy=(traj_pred[0, 0], traj_pred[0, 1]), width=np.sqrt(sigma_x) / 2,
							             height=np.sqrt(sigma_y) / 2, angle=0 / np.pi * 180)
							e1.set_alpha(0.5)

							for pred_step in range(1, self.args.prediction_horizon):
								sigma_x += np.square(sigmax[pred_step]) * self.args.dt * self.args.dt
								sigma_y += np.square(sigmay[pred_step]) * self.args.dt * self.args.dt

								e1 = Ellipse(xy=(traj_pred[pred_step, 0], traj_pred[pred_step, 1]), width=np.sqrt(sigma_x) / 2,
								             height=np.sqrt(sigma_y) / 2,
								             angle=0 / np.pi * 180)
								ax_pos_local.add_patch(e1)

		# Plot real trajectory
		gt_vel = np.zeros((self.args.prediction_horizon, self.args.output_dim))
		for i in range(self.args.prediction_horizon):
			idx = i * self.args.output_dim
			idy = i * self.args.output_dim + 1
			mu_x = model_vel_real[step, idx]
			mu_y = model_vel_real[step, idy]
			gt_vel[i, :] = [mu_x, mu_y]
		if False:
			real_vel = sup.rotate_predicted_vel_to_global_frame(gt_vel, batch_x[step, 2:])
		else:
			real_vel = gt_vel
		traj_real = sup.path_from_vel(initial_pos=np.array([0, 0]),
		                              pred_vel=real_vel, dt=self.args.dt)
		ax_pos_local.plot(traj_real[:, 0], traj_real[:, 1], color='m', label='Agent real trajectory')

		# Other agents
		for id in range(other_agents_pos[step].shape[0]):  # number of agents
			ax_pos_local.plot(other_agents_pos[step][id, 0]-x_init_global_frame
			                 , other_agents_pos[step][id, 1]-y_init_global_frame, marker='o', color='b')

		# Local grid
		""""""
		if rotate:
			heading = math.atan2(batch_x[step, 3], batch_x[step, 2])
			grid = sup.rotate_grid_around_center(batch_grid[step, :, :], -heading * 180 / math.pi)  # rotation in degrees
			sup.plot_grid(ax_pos_local, np.array([0.0, 0.0]), grid, self.gridmap.resolution,
			              np.array([self.args.submap_width, self.args.submap_height]))
		else:
			"""
			if real_vel[0,0] > 0.1:
				sup.plot_grid(ax_pos_local, np.array([3.0, 0.0]), batch_grid[step, :, :], self.gridmap.resolution,
				              np.array([self.args.submap_width, self.args.submap_height]))
			elif real_vel[0,0] < -0.1:
				sup.plot_grid(ax_pos_local, np.array([-3.0, 0.0]), batch_grid[step, :, :], self.gridmap.resolution,
				              np.array([self.args.submap_width, self.args.submap_height]))
			else:
			"""
			sup.plot_grid(ax_pos_local, np.array([0.0, 0.0]), batch_grid[step, :, :], self.gridmap.resolution,
		            np.array([self.args.submap_width, self.args.submap_height]))

	def save_global_scenario(self,input_list,grid_list,ped_grid_list,y_pred_list_global,y_ground_truth_list,other_agents_list,rotate=False,n_samples=1,format=".pdf"):
		if not os.path.exists(self.args.model_path + '/results/'+ self.args.scenario+"/figs"):
			os.makedirs(self.args.model_path + '/results/'+ self.args.scenario+"/figs")
		for animation_idx in range(len(input_list)):
			input = input_list[animation_idx]
			grid = grid_list[animation_idx]
			ped_grid = ped_grid_list[animation_idx]

			gt_vel = y_ground_truth_list[animation_idx]
			if not (y_pred_list_global is None):
				model_vel_pred = y_pred_list_global[animation_idx]
			other_agents_pos = other_agents_list[animation_idx]

			for step in range(input.shape[0]):  # trajectory length considering data from getTrajectoryAsBatch
				# Eucldean / position space
				self.ax_global_scenario.clear()
				#self.ax_global_scenario.plot(input[:, 0], input[:, 1], color='b', alpha=0.4, lw=1)
				#self.ax_global_scenario.plot(input[:step, 0], input[:step, 1], color='b', lw=2)
				self.ax_global_scenario.plot(input[step, 0], input[step, 1], color='c', marker='o')
				bottom_left_submap = input[step, 0:2] - np.array([self.args.submap_width / 2.0, self.args.submap_height / 2.0])
				#self.ax_global_scenario.add_patch(
				#	pl.Rectangle(bottom_left_submap, self.args.submap_width, self.args.submap_height, fill=None))

				# Plot real trajectory
				real_vel = np.zeros((1, self.args.prediction_horizon * self.args.output_dim))
				for i in range(self.args.prediction_horizon):
					idx = i * self.args.output_dim
					idy = i * self.args.output_dim + 1
					mu_x = gt_vel[step, idx]
					mu_y = gt_vel[step, idy]
					real_vel[0, 2 * i:2 * i + 2] = [mu_x, mu_y]

				traj_real = sup.path_from_vel(initial_pos=np.array([input[step, 0], input[step, 1]]),
				                              pred_vel=np.squeeze(real_vel), dt=self.args.dt)
				self.ax_global_scenario.plot(traj_real[:, 0], traj_real[:, 1], color='m', label='Agent real trajectory')

				# Other agents
				for id in range(other_agents_pos[step].shape[0]):  # number of agents
					self.ax_global_scenario.plot(other_agents_pos[step][id, 0], other_agents_pos[step][id, 1], marker='o', color='b',markersize=10)
				vel_pred = np.zeros((1, self.args.prediction_horizon * self.args.output_dim))
				# Predicted trajectory
				if not (y_pred_list_global is None):
					if self.args.n_mixtures == 0:
						# plot predicted trajectory global frame
						for i in range(self.args.prediction_horizon):
							idx = i * self.args.output_pred_state_dim
							idy = i * self.args.output_pred_state_dim + 1
							mu_x = model_vel_pred[step, idx]
							mu_y = model_vel_pred[step, idy]
							vel_pred[0, 2 * i:2 * i + 2] = [mu_x, mu_y]
						if rotate:
							pred_vel_global_frame = sup.rotate_predicted_vel_to_global_frame(vel_pred, input[step, 2:])
						else:
							pred_vel_global_frame = vel_pred
						traj_pred = sup.path_from_vel(initial_pos=np.array(
							[input[step, 0] * self.args.normalization_constant_x,
							 input[step, 1] * self.args.normalization_constant_y]),
							pred_vel=np.squeeze(pred_vel_global_frame), dt=self.args.dt)
						self.ax_global_scenario.plot(traj_pred[:, 0], traj_pred[:, 1], color=self.colors[0], label='Predicted trajectory')
					else:
						for sample_id in range(n_samples):
							for mix_idx in range(self.args.n_mixtures):
								# plot predicted trajectory global frame
								time = np.zeros([self.args.prediction_horizon])
								for pred_step in range(self.args.prediction_horizon):
									time[pred_step] = pred_step * self.args.dt
									idx = i * self.args.output_pred_state_dim * self.args.n_mixtures + mix_idx + self.args.output_pred_state_dim * self.args.n_mixtures*self.args.prediction_horizon*sample_id
									idy = i * self.args.output_pred_state_dim * self.args.n_mixtures + mix_idx + self.args.n_mixtures + self.args.output_pred_state_dim * self.args.n_mixtures*self.args.prediction_horizon*sample_id
									mu_x = model_vel_pred[step, idx]
									mu_y = model_vel_pred[step, idy]
									vel_pred[0, 2 * i:2 * i + 2] = [mu_x, mu_y]

								if rotate:
									pred_vel_global_frame = sup.rotate_predicted_vel_to_global_frame(vel_pred, input[step, 2:])
								else:
									pred_vel_global_frame = vel_pred
								traj_pred = sup.path_from_vel(initial_pos=np.array(
									[input[step, 0] * self.args.normalization_constant_x,
									 input[step, 1] * self.args.normalization_constant_y]),
									pred_vel=np.squeeze(pred_vel_global_frame), dt=self.args.dt)
								# sub-sample to make smoother trajectories
								_, traj_pred_smooth, _ = self.smoothenTrajectory(time, traj_pred, vel_pred)


								self.ax_global_scenario.plot(traj_pred_smooth[:, 0], traj_pred_smooth[:, 1], color=self.colors[mix_idx], label='Predicted trajectory')

								# prior of 0 on the uncertainty of the pedestrian velocity
								sigma_x = np.square(
									np.maximum(model_vel_pred[0, self.args.n_mixtures * 2 + mix_idx + self.args.output_pred_state_dim * self.args.n_mixtures*self.args.prediction_horizon*sample_id],0.75)) * self.args.dt * self.args.dt + 0.5
								sigma_y = np.square(
									np.maximum(model_vel_pred[0, self.args.n_mixtures * 3 + mix_idx + self.args.output_pred_state_dim * self.args.n_mixtures*self.args.prediction_horizon*sample_id],0.75)) * self.args.dt * self.args.dt +0.5

								e1 = Ellipse(xy=(traj_pred[0, 0], traj_pred[0, 1]), width=np.sqrt(sigma_x) / 2,
								             height=np.sqrt(sigma_y) / 2, angle=0 / np.pi * 180)
								e1.set_alpha(0.5)
								self.ax_global_scenario.add_patch(e1)

								for i in range(1, self.args.prediction_horizon):
									idx = i * self.args.output_pred_state_dim * self.args.n_mixtures + self.args.n_mixtures * 2 + mix_idx + self.args.output_pred_state_dim * self.args.n_mixtures*self.args.prediction_horizon*sample_id
									idy = i * self.args.output_pred_state_dim * self.args.n_mixtures + self.args.n_mixtures * 3 + mix_idx + self.args.output_pred_state_dim * self.args.n_mixtures*self.args.prediction_horizon*sample_id
									sigma_x += np.square(
										np.maximum(model_vel_pred[i - 1, idx],0.75)) * self.args.dt * self.args.dt
									sigma_y += np.square(
										np.maximum(model_vel_pred[i - 1, idy],0.75)) * self.args.dt * self.args.dt
									e1 = Ellipse(xy=(traj_pred[i, 0], traj_pred[i, 1]), width=np.sqrt(sigma_x) / 2,
									             height=np.sqrt(sigma_y) / 2,
									             angle=0 / np.pi * 180)
									e1.set_alpha(0.5)
									self.ax_global_scenario.add_patch(e1)

				sup.plot_grid(self.ax_global_scenario, np.array([0.0, 0.0]), self.gridmap.gridmap, self.gridmap.resolution,
				              self.gridmap.map_size)
				#self.ax_global_scenario.set_xlim([-self.gridmap.center[0], self.gridmap.center[0]])
				#self.ax_global_scenario.set_ylim([-self.gridmap.center[1], self.gridmap.center[1]])
				self.ax_global_scenario.set_xlim([input[step, 0]-7, input[step, 0]+7])
				self.ax_global_scenario.set_ylim([input[step, 1]-7, input[step, 1]+7])

				#self.ax_global_scenario.set_xlabel('x [m]')
				#self.ax_global_scenario.set_ylabel('y [m]')
				self.ax_global_scenario.axis('off')
				self.ax_global_scenario.set_aspect('equal')
				self.fig_global_scenario.savefig(self.args.model_path + '/results/'+ self.args.scenario+"/figs/"+"scene"+str(animation_idx)+"_step_"+str(step)+format, bbox_inches='tight')

	def smoothenTrajectory(self, time_vec,pose_vec,vel_vec,dt=0.05):
		"""
		Cubic interpolation with provided values in order to smoothen the trajectory and obtain
		a sequence with the specified dt.
		"""
		x_interpolator = interpolate.interp1d(time_vec, pose_vec[:, 0], kind='cubic', axis=0,fill_value='extrapolate')
		y_interpolator = interpolate.interp1d(time_vec, pose_vec[:, 1], kind='cubic', axis=0,fill_value='extrapolate')
		vx_interpolator = interpolate.interp1d(time_vec, vel_vec[:, 0], kind='cubic', axis=0,fill_value='extrapolate')
		vy_interpolator = interpolate.interp1d(time_vec, vel_vec[:, 1], kind='cubic', axis=0,fill_value='extrapolate')
		n_elem = self.args.prediction_horizon*int(self.args.dt/dt)
		new_time_vec = np.linspace(0, 0 + (n_elem - 1) * dt, n_elem)

		new_time_vec = np.zeros([n_elem])
		new_pose_vec = np.zeros([n_elem, 2])
		new_vel_vec = np.zeros([n_elem, 2])

		for ii in range(n_elem):
			t = 0 + ii * dt
			new_time_vec[ii] = t
			new_pose_vec[ii, 0] = x_interpolator(t)
			new_pose_vec[ii, 1] = y_interpolator(t)
			new_vel_vec[ii, 0] = vx_interpolator(t)
			new_vel_vec[ii, 1] = vy_interpolator(t)

		return new_time_vec, new_pose_vec, new_vel_vec

	def plot_on_video(self,input_list,grid_list,y_pred_list_global,y_ground_truth_list,other_agents_list,traj_list,all_traj_likelihood,test_args,social_trajectories=None):
		homography_file = os.path.join( self.args.data_path+ self.args.scenario, 'H.txt')
		if os.path.exists(homography_file):
			Hinv = np.linalg.inv(np.loadtxt(homography_file))
		else:
			print('[INF] No homography file')
		resolution = np.abs(1/Hinv[1,0])
		print("Resolution: " + str(resolution))
		scenario = self.args.scenario.split('/')[-1]
		video_file = self.args.data_path + self.args.scenario+'/' + scenario + '.avi'
		if os.path.exists(video_file):
			print('[INF] Using video file ' + video_file)
			cap = cv2.VideoCapture(video_file)
			if "zara_02" in self.args.scenario:
				time_offset = -22
			else:
				time_offset = 0

			if cap:
				dt = 0.4  # seconds (equivalent to 2.5 fps)
				#if 'seq_eth' in self.args.scenario:
				#	frames_between_annotation = 6.0
				#else:
				frames_between_annotation = 10.0

				frame_width = int(cap.get(3))
				frame_height = int(cap.get(4))

				# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
				if not os.path.exists(self.args.model_path + '/results/' + self.args.scenario):
					os.makedirs(self.args.model_path + '/results/' + self.args.scenario)
				if not os.path.exists(self.args.model_path + '/results/' + self.args.scenario+"/figs"):
					os.makedirs(self.args.model_path + '/results/' + self.args.scenario+"/figs")
				if social_trajectories:
					out = cv2.VideoWriter(self.args.model_path + '/results/' + self.args.scenario+'/outpy'+str(test_args.n_samples)+'_social.avi', cv2.VideoWriter_fourcc('M', 'P', '4', '2'), 10, (frame_width, frame_height))
				else:
					out = cv2.VideoWriter(self.args.model_path + '/results/' + self.args.scenario+'/outpy'+str(test_args.n_samples)+'.avi', cv2.VideoWriter_fourcc('M', 'P', '4', '2'), 10, (frame_width, frame_height))

			vel_real = np.zeros((self.args.prediction_horizon,  self.args.output_dim))
			vel_pred = np.zeros((self.args.prediction_horizon,  self.args.output_dim))
			sigmax = np.zeros((self.args.prediction_horizon, 1))
			sigmay = np.zeros((self.args.prediction_horizon, 1))
			pis = np.zeros((self.args.prediction_horizon, 3))
			for animation_idx in range(len(input_list)):
				traj = traj_list[animation_idx]

				if social_trajectories:
					social_traj = social_trajectories[animation_idx]
				gt_vel = y_ground_truth_list[animation_idx]
				input = input_list[animation_idx]
				if not (y_pred_list_global is None):
					model_vel_pred = y_pred_list_global[animation_idx]
				else:
					model_vel_pred = None
				traj_likelihood = all_traj_likelihood[animation_idx]
				# plot real trajectory global frame
				for step in range(input.shape[0]):

					time_stamp = traj.time_vec[step] / dt * frames_between_annotation + time_offset
					cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, time_stamp +self.args.prev_horizon* frames_between_annotation))
					ret, im = cap.read()
					overlay = im.copy()

					overlay_ellipses = im.copy()
					if not ret:
						print("Could not open the video")

					# Initial positions
					if self.args.normalize_data:
						x0 = input[step, 0] / self.args.sx_pos + self.args.min_pos_x
						y0 = input[step, 1] / self.args.sy_pos + self.args.min_pos_y
					else:
						x0 = input[step, 0]
						y0 = input[step, 1]

					# Real Trajectory
					for i in range(self.args.prediction_horizon):
						idx = i * self.args.output_dim
						idy = i * self.args.output_dim + 1
						mu_x = gt_vel[step, idx]
						mu_y = gt_vel[step, idy]
						vel_real[i, :] = [mu_x, mu_y]
					if self.args.rotated_grid:
						real_vel_global_frame = sup.rotate_predicted_vel_to_global_frame(vel_real, input[step, 2:])
					else:
						real_vel_global_frame = vel_real
					traj_real = sup.path_from_vel(initial_pos=np.array([x0,
					                                                    y0]),
					                              pred_vel=real_vel_global_frame, dt=self.args.dt)

					obsv_XY = sup.to_image_frame(Hinv, traj_real)
					sup.line_cv(overlay, obsv_XY, (255, 0, 0), 3) # bgr convention

					# Plot social trajectory
					if social_trajectories:
						for sample_id in range(30):
							# dt = 1 because model outputs increments on the positions
							#social_traj_pred = sup.path_from_vel(initial_pos=np.array([input[step, 0],
						  #                                                  input[step, 1]]),
						  #                            pred_vel=social_traj[step][sample_id], dt=1)
							social_traj_pred = social_traj[step][sample_id]
							obsv_XY = sup.to_image_frame(Hinv, social_traj_pred)
							sup.line_cv(overlay, obsv_XY, (255, 0, 255), 3)  # bgr convention

					# Plot real predicted traj from positions
					traj_real = traj.pose_vec[step:step+self.args.prev_horizon+self.args.prediction_horizon,:2]
					obsv_XY = sup.to_image_frame(Hinv, traj_real)
					sup.line_cv(overlay, obsv_XY, (0, 0, 0), 3)  # bgr convention

					# Predicted trajectory
					colors = [(0,0,255),(0,255,0),(0,255,255)]
					if not (y_pred_list_global is None):
						if self.args.n_mixtures == 0:
							prediction_sample = model_vel_pred[step][0]
							# plot predicted trajectory global frame
							time = np.zeros([self.args.prediction_horizon])
							for i in range(self.args.prediction_horizon):
								time[i] = i*self.args.dt
								idx = i * self.args.output_pred_state_dim
								idy = i * self.args.output_pred_state_dim + 1
								if self.args.normalize_data:
									mu_x = prediction_sample[0, idx] / self.args.sx_vel + self.args.min_vel_x
									mu_y = prediction_sample[0, idy] / self.args.sy_vel + self.args.min_vel_y
								else:
									mu_x = prediction_sample[0, idx]
									mu_y = prediction_sample[0, idy]
								vel_pred[i, :] = [mu_x, mu_y]
							if self.args.rotated_grid:
								pred_vel_global_frame = sup.rotate_predicted_vel_to_global_frame(vel_pred, input[step, 2:])
							else:
								pred_vel_global_frame = vel_pred

							if self.args.predict_positions:
								traj_pred = vel_pred
							else:
								traj_pred = sup.path_from_vel(initial_pos=np.array(
									[x0,
									 y0]),
									pred_vel=pred_vel_global_frame, dt=self.args.dt)
							# sub-sample to make smoother trajectories
							_, traj_pred_smooth, _ = self.smoothenTrajectory(time,traj_pred,vel_pred)

							obsv_XY = sup.to_image_frame(Hinv, traj_pred_smooth)
							sup.line_cv(overlay, obsv_XY, (0, 0, 255),2)
						else:
							for sample_id in range(test_args.n_samples):
								prediction_sample = model_vel_pred[step][sample_id]
								for mix_idx in range(self.args.n_mixtures): #self.args.n_mixtures
									# plot predicted trajectory global frame
									time = np.zeros([self.args.prediction_horizon])
									for pred_step in range(self.args.prediction_horizon):
										time[pred_step] = pred_step * self.args.dt
										idx = pred_step * self.args.output_pred_state_dim * self.args.n_mixtures + mix_idx
										if self.args.normalize_data:
											mu_x = prediction_sample[0, idx] / self.args.sx_vel + self.args.min_vel_x
											mu_y = prediction_sample[0, idx + self.args.n_mixtures] / self.args.sy_vel + self.args.min_vel_y
										else:
											mu_x = prediction_sample[0, idx]
											mu_y = prediction_sample[0, idx + self.args.n_mixtures]
										if self.args.output_pred_state_dim > 2:
											sigmax[pred_step, :] = prediction_sample[0][idx + 2 * self.args.n_mixtures]
											sigmay[pred_step, :] = prediction_sample[0][idx + 3 * self.args.n_mixtures]
										if self.args.output_pred_state_dim ==5:
											pis[pred_step, mix_idx] = prediction_sample[0][idx + 4 * self.args.n_mixtures]
										if math.isnan(mu_x) | math.isnan(mu_y):
											continue
										vel_pred[pred_step, :] = [mu_x, mu_y]

									if self.args.rotated_grid:
										pred_vel_global_frame = sup.rotate_predicted_vel_to_global_frame(vel_pred, input[step, 2:])
									else:
										pred_vel_global_frame = vel_pred
									traj_pred = sup.path_from_vel(initial_pos=np.array(
										[x0,
										 y0]),
										pred_vel=pred_vel_global_frame, dt=self.args.dt)
									# sub-sample to make smoother trajectories
									self.smoothenTrajectory(time, traj_pred, vel_pred)
									obsv_XY = sup.to_image_frame(Hinv, traj_pred)

									sup.line_cv(overlay, obsv_XY, colors[mix_idx], 2)

									if self.args.output_pred_state_dim > 2:
										# prior of 0 on the uncertainty of the pedestrian velocity
										sigma_x = np.square(min(max(sigmax[0],0.5),2.0)) * self.args.dt * self.args.dt +0.3
										sigma_y = np.square(min(max(sigmay[0],0.5),2.0))  * self.args.dt * self.args.dt +0.3
										if math.isnan(sigma_x) | math.isnan(sigma_y):
											continue
										axis = (int(np.sqrt(sigma_x) /resolution),int(np.sqrt(sigma_y) /resolution))
										center = (obsv_XY[0, 1], obsv_XY[0, 0])
										cv2.ellipse(overlay_ellipses, center,      (5, 5),    0, 0, 360, (255,153,51),-1)

										for pred_step in range(1, self.args.prediction_horizon):
											sigma_x = sigma_x + np.square(min(max(sigmax[pred_step],0.5),2.0)) * self.args.dt * self.args.dt
											sigma_y = sigma_y + np.square(min(max(sigmay[pred_step],0.5),2.0)) * self.args.dt * self.args.dt
											if math.isnan(sigma_x) | math.isnan(sigma_y):
												continue
											axis = (int(np.sqrt(sigma_x) / resolution),
											        int(np.sqrt(sigma_y) / resolution))
											if (sigma_x<0) or (sigma_y<0):
												print("Negative ellipse")
											center = (obsv_XY[pred_step, 1], obsv_XY[pred_step, 0])
											cv2.ellipse(overlay_ellipses, center, (5, 5), 0, 0, 360, (255,153,51),-1)

					# Plot other agents
					plot_other_agents = False
					if plot_other_agents:
						other_agents = other_agents_list[animation_idx][step]
						axis = (int(np.sqrt(0.3) / resolution),
						        int(np.sqrt(0.3) / resolution))

						for agent_id in range(other_agents.shape[0]):
							obsv_XY = sup.to_image_frame(Hinv, np.expand_dims(other_agents[agent_id,:2],axis=0))
							center = (obsv_XY[0, 1], obsv_XY[0, 0])
							cv2.ellipse(overlay_ellipses, center, (5, 5), 0, 0, 360, (153, 153, 51), -1)

					image_new = cv2.addWeighted(overlay, 0.6, im, 0.4, 0)
					image_new = cv2.addWeighted(image_new, 0.6, overlay_ellipses, 0.4, 0)

					# Adding legend
					font = cv2.FONT_HERSHEY_SIMPLEX
					cv2.line(image_new, (frame_width - 300, 30), (frame_width - 280, 30), (255, 0, 0), 4)
					cv2.putText(image_new, "Real Trajectory", (frame_width - 270, 30), font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
					cv2.line(image_new, (frame_width - 300, 50), (frame_width - 280, 50), (0, 0, 255), 4)
					if self.args.output_pred_state_dim == 5:
						cv2.putText(image_new, "Predicted Trajectory Prob " + str(np.sum(pis[:,0])/self.args.prediction_horizon*100), (frame_width - 270, 50), font, 0.5, (0, 0, 255), 2,
						            cv2.LINE_AA)
						cv2.line(image_new, (frame_width - 300, 70), (frame_width - 280, 70), (0, 255, 0), 4)
						cv2.putText(image_new, "Predicted Trajectory Prob " + str(np.sum(pis[:,1])/self.args.prediction_horizon*100), (frame_width - 270, 70), font, 0.5, (0, 255, 0), 2,
						            cv2.LINE_AA)
						cv2.line(image_new, (frame_width - 300, 90), (frame_width - 280, 90), (0, 255, 255), 4)
						cv2.putText(image_new, "Predicted Trajectory Prob " + str(np.sum(pis[:,2])/self.args.prediction_horizon*100), (frame_width - 270, 90), font, 0.5, (0, 255, 255), 2,
						            cv2.LINE_AA)
						cv2.line(image_new, (frame_width - 300, 110), (frame_width - 280, 110), (255, 153, 51), 4)
					else:
						cv2.putText(image_new, "Predicted Trajectory 1", (frame_width - 270, 50), font, 0.5, (0, 0, 255), 2,
						            cv2.LINE_AA)
						cv2.line(image_new, (frame_width - 300, 70), (frame_width - 280, 70), (0, 255, 0), 4)
						cv2.putText(image_new, "Predicted Trajectory 2", (frame_width - 270, 70), font, 0.5, (0, 255, 0), 2,
						            cv2.LINE_AA)
						cv2.line(image_new, (frame_width - 300, 90), (frame_width - 280, 90), (0, 255, 255), 4)
						cv2.putText(image_new, "Predicted Trajectory 3" , (frame_width - 270, 90), font, 0.5, (0, 255, 255), 2,
						            cv2.LINE_AA)
						cv2.line(image_new, (frame_width - 300, 110), (frame_width - 280, 110), (255, 153, 51), 4)
					cv2.putText(image_new, "1-sigma Uncertainty", (frame_width - 270, 110), font, 0.5, (255,153,51), 2,
					            cv2.LINE_AA)
					if social_trajectories:
						cv2.line(image_new, (frame_width - 300, 130), (frame_width - 280, 130), (255, 0, 255), 4)
						cv2.putText(image_new, "Social-Ways 30 Samples", (frame_width - 270, 130), font, 0.5, (255, 0, 255), 2,
						            cv2.LINE_AA)

					out.write(image_new)

					cv2.imwrite(self.args.model_path + '/results/' + self.args.scenario+"/figs/result_"+str(animation_idx)+"_"+str(step)+".jpg", image_new);

			# When everything done, release the video capture and video write objects
			cap.release()
			out.release()

			# Closes all the frames
			cv2.destroyAllWindows()

		else:
			"Video not found"

	def plot_on_image(self,input_list,grid_list,y_pred_list_global,y_ground_truth_list,other_agents_list,traj_list,test_args,social_trajectories=None):

		#self.args.dt /= 10

		scenario = self.args.scenario.split('/')[-1]

		if not os.path.exists(self.args.model_path + '/../videos/'):
			os.makedirs(self.args.model_path + '/../videos/')
		if test_args.freeze_other_agents:
			video_file = self.args.model_path + '/../videos/'+scenario + str(self.args.exp_num) + "_frozen.avi"
			map_file = self.args.data_path + self.args.scenario + '/map.png'
			homography_file = os.path.join(self.args.data_path + self.args.scenario, 'H.txt')
		elif test_args.unit_testing:
			video_file = self.args.model_path + '/../videos/'+scenario + str(self.args.exp_num) + "_unit_tests.avi"
			map_file = self.args.data_path + self.args.scenario + '/unit_map.png'
			homography_file = os.path.join(self.args.data_path + self.args.scenario, 'unit_H.txt')
		else:
			video_file = self.args.model_path + '/../videos/'+scenario + str(self.args.exp_num) + "_final.avi"
			map_file = self.args.data_path + self.args.scenario + '/map.png'
			homography_file = os.path.join(self.args.data_path + self.args.scenario, 'H.txt')

		if os.path.exists(homography_file):
			Hinv = np.linalg.inv(np.loadtxt(homography_file))
		else:
			print('[INF] No homography file')
		resolution = self.args.submap_resolution
		print("Resolution: " + str(resolution))

		im = np.uint8(cv2.imread(map_file) * -1 + 255)
		im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE) # for roboat output
		frame_height, frame_width, layers = im.shape
		scale_factor = 2
		resized_img = cv2.resize(im, (frame_width * scale_factor, frame_height * scale_factor),
		                         interpolation=cv2.INTER_AREA)

		if os.path.exists(map_file):
			print('[INF] Using video file ' + video_file)
			cap = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc('M', 'P', '4', '2'), 5,
			                      (frame_width * scale_factor, frame_height * scale_factor))

			vel_real = np.zeros((self.args.prediction_horizon,  self.args.output_dim))
			vel_pred = np.zeros((self.args.prediction_horizon,  self.args.output_dim))
			sigmax = np.zeros((self.args.prediction_horizon, 1))
			sigmay = np.zeros((self.args.prediction_horizon, 1))
			pis = np.zeros((self.args.prediction_horizon, 3))
			for animation_idx in range(len(input_list)):
				traj = traj_list[animation_idx]

				if social_trajectories:
					social_traj = social_trajectories[animation_idx]
				gt_vel = y_ground_truth_list[animation_idx]
				input = input_list[animation_idx]
				if not (y_pred_list_global is None):
					model_vel_pred = y_pred_list_global[animation_idx]
				else:
					model_vel_pred = None
				# plot real trajectory global frame
				for step in range(input.shape[0]):

					overlay = resized_img.copy()

					# Initial positions
					if self.args.normalize_data:
						x0 = input[step, 0] / self.args.sx_pos + self.args.min_pos_x
						y0 = input[step, 1] / self.args.sy_pos + self.args.min_pos_y
					else:
						x0 = input[step, 0]
						y0 = input[step, 1]

					# Real Trajectory
					for i in range(self.args.prediction_horizon):
						idx = i * self.args.output_dim
						idy = i * self.args.output_dim + 1
						mu_x = gt_vel[step, idx]
						mu_y = gt_vel[step, idy]
						vel_real[i, :] = [mu_x, mu_y]

					real_vel_global_frame = vel_real
					traj_real = sup.path_from_vel(initial_pos=np.array([x0,
					                                                    y0]),
					                              pred_vel=real_vel_global_frame, dt=self.args.dt)

					obsv_XY = sup.to_image_frame(Hinv, traj_real)*scale_factor
					sup.line_cv(overlay, obsv_XY, (128, 128, 0), 3) # bgr convention

					# Plot social trajectory
					if social_trajectories:
						for sample_id in range(3):
							social_traj_pred = sup.path_from_vel(initial_pos=np.array([input[step, 0],
						                                                    input[step, 1]]),
						                              pred_vel=social_traj[step][sample_id], dt=1)
							obsv_XY = sup.to_image_frame(Hinv, social_traj_pred)*scale_factor
							sup.line_cv(overlay, obsv_XY, (255, 0, 0), 3)  # bgr convention

					# Plot real predicted traj from positions
					traj_real = traj.pose_vec[step+self.args.prev_horizon:step+self.args.prev_horizon+self.args.prediction_horizon,:2]
					obsv_XY = sup.to_image_frame(Hinv, traj_real)*scale_factor
					sup.line_cv(overlay, obsv_XY, (255, 0, 0), 3)  # bgr convention

					traj_real = traj.pose_vec[step:,:2]
					obsv_XY2 = sup.to_image_frame(Hinv, traj_real)*scale_factor
					obsv_XY3 = obsv_XY2[self.args.prev_horizon:self.args.prev_horizon+self.args.prediction_horizon]
					delta = obsv_XY - obsv_XY3

					if np.max(np.abs(delta))>0:
						print("problem")
					sup.line_cv(overlay, obsv_XY2, (0, 0, 0), 3)  # bgr convention

					# Predicted trajectory
					colors = [(0,0,255),(0,255,0),(0,255,255)]
					if not (y_pred_list_global is None):
						if self.args.n_mixtures == 0:
							prediction_sample = model_vel_pred[step][0]
							# plot predicted trajectory global frame
							time = np.zeros([self.args.prediction_horizon])
							for i in range(self.args.prediction_horizon):
								time[i] = i*self.args.dt
								idx = i * self.args.output_pred_state_dim
								idy = i * self.args.output_pred_state_dim + 1
								if self.args.normalize_data:
									mu_x = prediction_sample[0, idx] / self.args.sx_vel + self.args.min_vel_x
									mu_y = prediction_sample[0, idy] / self.args.sy_vel + self.args.min_vel_y
								else:
									mu_x = prediction_sample[0, idx]
									mu_y = prediction_sample[0, idy]
								vel_pred[i, :] = [mu_x, mu_y]

							pred_vel_global_frame = vel_pred

							if self.args.predict_positions:
								traj_pred = vel_pred
							else:
								traj_pred = sup.path_from_vel(initial_pos=np.array(
									[x0,
									 y0]),
									pred_vel=pred_vel_global_frame, dt=self.args.dt)
							# sub-sample to make smoother trajectories
							#_, traj_pred_smooth, _ = self.smoothenTrajectory(time,traj_pred,vel_pred)

							obsv_XY = sup.to_image_frame(Hinv, traj_pred)*scale_factor
							sup.line_cv(overlay, obsv_XY, (0, 0, 255),2)
						else:
							for sample_id in range(test_args.n_samples):
								prediction_sample = model_vel_pred[step][sample_id]
								for mix_idx in range(self.args.n_mixtures): #self.args.n_mixtures
									# plot predicted trajectory global frame
									time = np.zeros([self.args.prediction_horizon])
									for pred_step in range(self.args.prediction_horizon):
										time[pred_step] = pred_step * self.args.dt
										idx = pred_step * self.args.output_pred_state_dim * self.args.n_mixtures + mix_idx
										if self.args.normalize_data:
											mu_x = prediction_sample[0, idx] / self.args.sx_vel + self.args.min_vel_x
											mu_y = prediction_sample[0, idx + self.args.n_mixtures] / self.args.sy_vel + self.args.min_vel_y
										else:
											mu_x = prediction_sample[0, idx] + input[step, 2]
											mu_y = prediction_sample[0, idx + self.args.n_mixtures] + input[step, 3]
										if self.args.output_pred_state_dim > 2:
											sigmax[pred_step, :] = prediction_sample[0][idx + 2 * self.args.n_mixtures]
											sigmay[pred_step, :] = prediction_sample[0][idx + 3 * self.args.n_mixtures]
										if self.args.output_pred_state_dim ==5:
											pis[pred_step, mix_idx] = prediction_sample[0][idx + 4 * self.args.n_mixtures]
										if math.isnan(mu_x) | math.isnan(mu_y):
											continue
										vel_pred[pred_step, :] = [mu_x, mu_y]

									pred_vel_global_frame = vel_pred
									traj_pred = sup.path_from_vel(initial_pos=np.array(
										[x0,
										 y0]),
										pred_vel=pred_vel_global_frame, dt=self.args.dt)
									# sub-sample to make smoother trajectories
									#self.smoothenTrajectory(time, traj_pred, vel_pred)
									obsv_XY = sup.to_image_frame(Hinv, traj_pred)*scale_factor

									sup.line_cv(overlay, obsv_XY, colors[mix_idx], 2)

									try:
										if self.args.output_pred_state_dim > 2:
											# prior of 0 on the uncertainty of the pedestrian velocity
											sigma_x = np.square(sigmax[0]) * self.args.dt * self.args.dt +6.0
											sigma_y = np.square(sigmay[0]) * self.args.dt * self.args.dt +3.0
											if math.isnan(sigma_x) | math.isnan(sigma_y):
												continue
											axis = (int(np.sqrt(sigma_x) /resolution*scale_factor),int(np.sqrt(sigma_y) /resolution*scale_factor))
											y = obsv_XY[0, 1]*scale_factor
											x = obsv_XY[0, 0]*scale_factor
											center = (y , x)
											cv2.ellipse(overlay, center,       axis,    0, 0, 360, (255,153,51),-1)

											for pred_step in range(1, self.args.prediction_horizon):
												sigma_x += np.square(sigmax[pred_step]) * self.args.dt * self.args.dt
												sigma_y += np.square(sigmay[pred_step]) * self.args.dt * self.args.dt
												if math.isnan(sigma_x) | math.isnan(sigma_y):
													continue
												axis = (int(np.sqrt(sigma_x) / resolution*scale_factor),
												        int(np.sqrt(sigma_y) / resolution*scale_factor))
												if (sigma_x<0) or (sigma_y<0):
													print("Negative ellipse")
													axis = (5, 5)
												y = obsv_XY[pred_step, 1] * scale_factor
												x = obsv_XY[pred_step, 0] * scale_factor
												center = (y, x)
												cv2.ellipse(overlay, center, axis, 0, 0, 360, (255,153,51),-1)
									except:
										print("Failed to add ellipse")

					# Plot other agents
					plot_other_agents = True
					if plot_other_agents:
						other_agents = other_agents_list[animation_idx][step]
						axis = (int(np.sqrt(0.3) / resolution),
						        int(np.sqrt(0.3) / resolution))

						for agent_id in range(other_agents.shape[0]):
							obsv_XY = sup.to_image_frame(Hinv, np.expand_dims(other_agents[agent_id,:2],axis=0))
							center = (int(obsv_XY[0, 1]*scale_factor), int(obsv_XY[0, 0]*scale_factor))
							cv2.ellipse(overlay, center, (5, 5), 0, 0, 360, (153, 153, 51), -1)

					# Adding legend
					font = cv2.FONT_HERSHEY_SIMPLEX
					cv2.line(overlay, (int(frame_height*scale_factor*0.9-50), 30), (int(frame_height*scale_factor*0.9-40), 30), (255, 0, 0), 4)
					cv2.putText(overlay, "Real Trajectory", (int(frame_height*scale_factor*0.9-30), 30), font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
					cv2.line(overlay, (int(frame_height*scale_factor*0.9-50), 50), (int(frame_height*scale_factor*0.9-40), 50), (0, 0, 255), 4)
					cv2.putText(overlay, "Predicted Trajectory", (int(frame_height*scale_factor * 0.9 - 40), 50), font, 0.5, (0, 0, 255), 2,
					            cv2.LINE_AA)

					cap.write(overlay)

					#cv2.imwrite(self.args.model_path + '/results/' + self.args.scenario+"/figs/result_"+str(animation_idx)+"_"+str(step)+".jpg", overlay);

			# When everything done, release the video capture and video write objects
			cap.release()

			# Closes all the frames
			cv2.destroyAllWindows()

		else:
			print("map_file not found")

	def plot_GA3C(self,input_list,grid_list,ped_grid_list,y_pred_list_global,y_ground_truth_list,other_agents_list,traj_list,rotate=False,n_samples=1,social_trajectories=None):
		homography_file = os.path.join( self.args.data_path+ self.args.scenario, 'H.txt')
		if os.path.exists(homography_file):
			Hinv = np.linalg.inv(np.loadtxt(homography_file))
		else:
			print('[INF] No homography file')
		resolution = np.abs(1/Hinv[0,0])/2
		print("Resolution: " + str(resolution))
		scenario = self.args.scenario.split('/')[-1]
		map_file = self.args.data_path + self.args.scenario+ '/map.png'
		im = cv2.imread(map_file) +255
		height, width, layers = im.shape
		scale_factor = 2
		resized_img = cv2.resize(im, (height*scale_factor, width*scale_factor), interpolation=cv2.INTER_AREA)

		video_file = self.args.model_path + '/results/' + self.args.scenario+'/outpy'+'.avi'
		if os.path.exists(map_file):
			print('[INF] Using map file ' + map_file)
			cap = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc('M', 'P', '4', '2'), 10, (width*scale_factor, height*scale_factor))


			vel_pred = np.zeros((self.args.prediction_horizon,  self.args.output_dim))
			sigmax = np.zeros((self.args.prediction_horizon, 1))
			sigmay = np.zeros((self.args.prediction_horizon, 1))
			pis = np.zeros((self.args.prediction_horizon, 3))

			for animation_idx in range(len(input_list)):
				traj = traj_list[animation_idx]

				gt_vel = y_ground_truth_list[animation_idx]
				input = input_list[animation_idx]
				if not (y_pred_list_global is None):
					model_vel_pred = y_pred_list_global[animation_idx]
				else:
					model_vel_pred = None

				# Real Trajectory
				traj_real = np.zeros((len(traj), self.args.output_dim))
				for i in range(len(traj)):
					x = traj[i]["pedestrian_state"]["position"][0]
					y = traj[i]["pedestrian_state"]["position"][1]
					traj_real[i, :] = [x, y]

				# plot real trajectory global frame
				for step in range(len(model_vel_pred)):

					overlay = resized_img.copy()

					# Initial positions
					x0 = traj[0]["pedestrian_state"]["position"][0]
					y0 = traj[0]["pedestrian_state"]["position"][1]
					obsv_XY = sup.to_image_frame(Hinv, np.expand_dims(np.array([x0,y0]),axis=0))
					center = (obsv_XY[0, 1], obsv_XY[0, 0])
					axis = (int(np.sqrt(0.3) / resolution),
					        int(np.sqrt(0.3) / resolution))
					cv2.ellipse(overlay, center, (5, 5), 0, 0, 360, (255, 0, 0), -1)

					# Final positions
					x0 = traj[-1]["pedestrian_state"]["position"][0]
					y0 = traj[-1]["pedestrian_state"]["position"][1]
					obsv_XY = sup.to_image_frame(Hinv, np.expand_dims(np.array([x0, y0]), axis=0))
					center = (obsv_XY[0, 1], obsv_XY[0, 0])
					axis = (int(np.sqrt(0.3) / resolution),
					        int(np.sqrt(0.3) / resolution))
					cv2.ellipse(overlay, center, (5, 5), 0, 0, 360, (0, 255, 0), -1)

					obsv_XY = sup.to_image_frame(Hinv, traj_real)
					sup.line_cv(overlay, obsv_XY, (0, 0, 0), 3) # bgr convention

					# Predicted trajectory
					colors = [(0,0,255),(0,255,0),(0,255,255)]
					if not (y_pred_list_global is None):
						if self.args.n_mixtures == 0:
							prediction_sample = model_vel_pred[step][0]
							# plot predicted trajectory global frame
							time = np.zeros([self.args.prediction_horizon])
							for i in range(self.args.prediction_horizon):
								time[i] = i*self.args.dt
								idx = i * self.args.output_pred_state_dim
								idy = i * self.args.output_pred_state_dim + 1
								if self.args.normalize_data:
									mu_x = prediction_sample[0, idx] / self.args.sx_vel + self.args.min_vel_x
									mu_y = prediction_sample[0, idy] / self.args.sy_vel + self.args.min_vel_y
								else:
									mu_x = prediction_sample[0, idx]
									mu_y = prediction_sample[0, idy]
								vel_pred[i, :] = [mu_x, mu_y]

							if True: #self.args.predict_positions
								traj_pred = vel_pred
							else:
								traj_pred = sup.path_from_vel(initial_pos=np.array(
									[x0,
									 y0]),
									pred_vel=pred_vel_global_frame, dt=self.args.dt)
							# sub-sample to make smoother trajectories
							#_, traj_pred_smooth, _ = self.smoothenTrajectory(time,traj_pred,vel_pred)

							obsv_XY = sup.to_image_frame(Hinv, traj_pred)
							sup.line_cv(overlay, obsv_XY, (0, 0, 255),2)
						else:
							for sample_id in range(n_samples):
								prediction_sample = model_vel_pred[step][sample_id]
								for mix_idx in range(self.args.n_mixtures): #self.args.n_mixtures
									# plot predicted trajectory global frame
									time = np.zeros([self.args.prediction_horizon])
									for pred_step in range(self.args.prediction_horizon):
										time[pred_step] = pred_step * self.args.dt
										idx = pred_step * self.args.output_pred_state_dim * self.args.n_mixtures + mix_idx
										if self.args.normalize_data:
											mu_x = prediction_sample[0, idx] / self.args.sx_vel + self.args.min_vel_x
											mu_y = prediction_sample[0, idx + self.args.n_mixtures] / self.args.sy_vel + self.args.min_vel_y
										else:
											mu_x = prediction_sample[0, idx]
											mu_y = prediction_sample[0, idx + self.args.n_mixtures]
										sigmax[pred_step, :] = prediction_sample[0][idx + 2 * self.args.n_mixtures]
										sigmay[pred_step, :] = prediction_sample[0][idx + 3 * self.args.n_mixtures]
										if self.args.output_pred_state_dim ==5:
											pis[pred_step, mix_idx] = prediction_sample[0][idx + 4 * self.args.n_mixtures]
										if math.isnan(mu_x) | math.isnan(mu_y):
											continue
										vel_pred[pred_step, :] = [mu_x, mu_y]

									if rotate:
										pred_vel_global_frame = sup.rotate_predicted_vel_to_global_frame(vel_pred, input[step, 2:])
									else:
										pred_vel_global_frame = vel_pred
									traj_pred = sup.path_from_vel(initial_pos=np.array(
										[x0,
										 y0]),
										pred_vel=pred_vel_global_frame, dt=self.args.dt)
									# sub-sample to make smoother trajectories
									#self.smoothenTrajectory(time, traj_pred, vel_pred)
									obsv_XY = sup.to_image_frame(Hinv, traj_pred)

									sup.line_cv(overlay, obsv_XY, colors[mix_idx], 2)

									# prior of 0 on the uncertainty of the pedestrian velocity
									sigma_x = np.square(min(sigmax[0],0.5)) * self.args.dt * self.args.dt +0.2
									sigma_y = np.square(min(sigmay[0],0.5)) * self.args.dt * self.args.dt +0.2
									if math.isnan(sigma_x) | math.isnan(sigma_y):
										continue
									axis = (int(np.sqrt(sigma_x) /resolution),int(np.sqrt(sigma_y) /resolution))
									center = (obsv_XY[0, 1], obsv_XY[0, 0])
									cv2.ellipse(overlay, center,       (5, 5),    0, 0, 360, (255,153,51),-1)

									for pred_step in range(1, self.args.prediction_horizon):
										sigma_x += np.square(min(sigmax[pred_step],0.5)) * self.args.dt * self.args.dt
										sigma_y += np.square(min(sigmay[pred_step],0.5)) * self.args.dt * self.args.dt
										if math.isnan(sigma_x) | math.isnan(sigma_y):
											continue
										axis = (int(np.sqrt(sigma_x) / resolution),
										        int(np.sqrt(sigma_y) / resolution))
										if (sigma_x<0) or (sigma_y<0):
											print("Negative ellipse")
										center = (obsv_XY[pred_step, 1], obsv_XY[pred_step, 0])
										cv2.ellipse(overlay, center, (5, 5), 0, 0, 360, (255,153,51),-1)

					# Plot other agents
					plot_other_agents = True
					if plot_other_agents:
						other_agents = np.expand_dims(other_agents_list[animation_idx][0,step],axis=0)
						axis = (20 * int(np.sqrt(0.3) / resolution),
						        20 * int(np.sqrt(0.3) / resolution))

						for agent_id in range(other_agents.shape[0]):
							obsv_XY = sup.to_image_frame(Hinv, np.expand_dims(other_agents[agent_id],axis=0))
							center = (obsv_XY[0, 1], obsv_XY[0, 0])
							cv2.ellipse(overlay, center, (5, 5), 0, 0, 360, (153, 153, 51), -1)

					# Adding legend
					font = cv2.FONT_HERSHEY_SIMPLEX
					cv2.line(overlay, (width - 70, 30), (width - 50, 30), (0, 0, 0), 4)
					cv2.putText(overlay, "Real Trajectory", (width - 40, 30), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
					cv2.line(overlay, (width - 70, 50), (width - 50, 50), (0, 0, 255), 4)
					if self.args.output_pred_state_dim == 5:
						cv2.putText(overlay, "Predicted Trajectory Prob " + str(np.sum(pis[:,0])/self.args.prediction_horizon*100), (width - 40, 50), font, 0.5, (0, 0, 255), 2,
						            cv2.LINE_AA)
						cv2.line(overlay, (width - 70, 70), (width - 50, 70), (0, 255, 0), 4)
						cv2.putText(overlay, "Predicted Trajectory Prob " + str(np.sum(pis[:,1])/self.args.prediction_horizon*100), (width - 40, 70), font, 0.5, (0, 255, 0), 2,
						            cv2.LINE_AA)
						cv2.line(overlay, (width - 70, 90), (width - 50, 90), (0, 255, 255), 4)
						cv2.putText(overlay, "Predicted Trajectory Prob " + str(np.sum(pis[:,2])/self.args.prediction_horizon*100), (width - 40, 90), font, 0.5, (0, 255, 255), 2,
						            cv2.LINE_AA)
						cv2.line(overlay, (width - 70, 110), (width - 50, 110), (255, 153, 51), 4)
					else:
						cv2.putText(overlay, "Predicted Trajectory Mode 0", (width - 40, 50), font, 0.5, (0, 0, 255), 2,
						            cv2.LINE_AA)
						cv2.line(overlay, (width - 70, 70), (width - 50, 70), (0, 255, 0), 4)
						cv2.putText(overlay, "Goal Position", (width - 40, 70), font, 0.5, (0, 255, 0), 2,
						            cv2.LINE_AA)
						cv2.line(overlay, (width - 70, 90), (width - 50, 90), (255, 0, 0), 4)
						cv2.putText(overlay, "Initial", (width - 40, 90), font, 0.5, (255, 0, 0), 2,
						            cv2.LINE_AA)

					cap.write(overlay)

					cv2.imwrite(self.args.model_path + '/results/' + self.args.scenario+"/figs/result_"+str(animation_idx)+"_"+str(step)+".jpg", overlay);

			# When everything done, release the video capture and video write objects
			cap.release()

			# Closes all the frames
			cv2.destroyAllWindows()

		else:
			print("Video not found")

	def animate_GA3C(self,input_list,goal_list,grid_list,ped_grid_list,y_pred_list_global,y_ground_truth_list,other_agents_list,traj_list,test_args):
		#self.args.dt = 0.1
		pl.show(block=False)
		if not os.path.exists(self.args.model_path + '/results/'+ self.args.scenario):
			os.makedirs(self.args.model_path + '/results/'+ self.args.scenario)
		if test_args.freeze_other_agents:
			video_file = self.args.model_path + '/results/' + self.args.scenario+"/frozen.mp4"
		else:
			video_file = self.args.model_path + '/results/' + self.args.scenario + "/final.mp4"
		with self.writer.saving(self.fig_ga3c, video_file, 100):
			for animation_idx in range(len(input_list)):
				input = input_list[animation_idx]
				goal = goal_list[animation_idx]
				traj = traj_list[animation_idx]
				if not (y_pred_list_global is None):
					model_vel_pred = y_pred_list_global[animation_idx]
				else:
					model_vel_pred = None
				other_agents_pos = other_agents_list[animation_idx]

				# Real Trajectory
				traj_real = np.zeros((len(traj), self.args.output_dim))
				for i in range(len(traj)):
					x = traj[i]["pedestrian_state"]["position"][0]
					y = traj[i]["pedestrian_state"]["position"][1]
					traj_real[i, :] = [x, y]

				for step in range(input.shape[0]): # trajectory length considering data from getTrajectoryAsBatch
					# Eucldean / position space
					self.ga3c_plot.clear()
					#self.ga3c_plot.plot(input[:, 0], input[:, 1], color='c', alpha=0.4, lw=1)
					#self.ga3c_plot.plot(input[:step, 0], input[:step, 1], color='b', lw=2)
					# Initial positions
					x0 = traj[0]["pedestrian_state"]["position"][0]
					y0 = traj[0]["pedestrian_state"]["position"][1]
					self.ga3c_plot.plot(x0, y0, color='r', marker='o')

					xf = traj[-1]["pedestrian_state"]["position"][0]
					yf = traj[-1]["pedestrian_state"]["position"][1]
					self.ga3c_plot.plot(xf, yf, color='g', marker='x')

					xf = goal[0,0]
					yf = goal[0,1]
					self.ga3c_plot.plot(xf, yf, color='g', marker='o')

					# Other agents
					for id in range(other_agents_pos.shape[0]): # number of agents
						self.ga3c_plot.plot(other_agents_pos[id][step, 0], other_agents_pos[id][step, 1], marker='o', color='b')
					vel_pred = np.zeros((self.args.prediction_horizon,  self.args.output_dim))
					sigmax = np.zeros((self.args.prediction_horizon, 1))
					sigmay = np.zeros((self.args.prediction_horizon, 1))
					#Predicted trajectory
					if not (y_pred_list_global is None):
						if self.args.n_mixtures == 0:
							prediction_sample = model_vel_pred[step][0]
							# plot predicted trajectory global frame
							time = np.zeros([self.args.prediction_horizon])
							for i in range(self.args.prediction_horizon):
								time[i] = i*self.args.dt
								idx = i * self.args.output_pred_state_dim
								idy = i * self.args.output_pred_state_dim + 1
								if self.args.normalize_data:
									mu_x = prediction_sample[0, idx] / self.args.sx_vel + self.args.min_vel_x
									mu_y = prediction_sample[0, idy] / self.args.sy_vel + self.args.min_vel_y
								else:
									mu_x = prediction_sample[0, idx]
									mu_y = prediction_sample[0, idy]
								vel_pred[i, :] = [mu_x, mu_y]
							x0 = traj[step]["pedestrian_state"]["position"][0]
							y0 = traj[step]["pedestrian_state"]["position"][1]
							if self.args.predict_positions:
								traj_pred = vel_pred
							else:
								traj_pred = sup.path_from_vel(initial_pos=np.array([x0,y0]),
									pred_vel=vel_pred, dt=self.args.dt)
							self.ga3c_plot.plot(traj_pred[:, 0], traj_pred[:, 1], color=self.colors[0], label='Predicted trajectory')
						else:
							for sample_id in range(test_args.n_samples):
								prediction_sample = model_vel_pred[step][sample_id]
								x0 = traj[step]["pedestrian_state"]["position"][0]
								y0 = traj[step]["pedestrian_state"]["position"][1]
								for mix_idx in range(self.args.n_mixtures):
									# plot predicted trajectory global frame
									for pred_step in range(self.args.prediction_horizon):
										idx = pred_step * self.args.output_pred_state_dim * self.args.n_mixtures + mix_idx
										mu_x = prediction_sample[0, idx]
										mu_y = prediction_sample[0, idx + self.args.n_mixtures]
										sigmax[pred_step, :] = prediction_sample[0][idx + 2 * self.args.n_mixtures]
										sigmay[pred_step, :] = prediction_sample[0][idx + 3 * self.args.n_mixtures]
										vel_pred[pred_step, :] = [mu_x, mu_y]

									pred_vel_global_frame = vel_pred
									traj_pred = sup.path_from_vel(initial_pos=np.array([x0,y0]),pred_vel=pred_vel_global_frame, dt=self.args.dt)

									self.ga3c_plot.plot(traj_pred[:, 0], traj_pred[:, 1], color=self.colors[mix_idx], label='Predicted trajectory')

									# prior of 0 on the uncertainty of the pedestrian velocity
									sigma_x = np.square(sigmax[0]) * self.args.dt * self.args.dt
									sigma_y = np.square(sigmay[0]) * self.args.dt * self.args.dt

									e1 = Ellipse(xy=(traj_pred[0, 0], traj_pred[0, 1]), width=np.sqrt(sigma_x) / 2,
									             height=np.sqrt(sigma_y) / 2, angle=0 / np.pi * 180)
									e1.set_alpha(0.5)
									self.ga3c_plot.add_patch(e1)
									#self.ga3c_plot.plot(traj_pred[:, 0], traj_pred[:, 1], color=self.colors[mix_idx], label='Predicted trajectory')
									for pred_step in range(1, self.args.prediction_horizon):
										sigma_x += np.square(sigmax[pred_step]) * self.args.dt * self.args.dt
										sigma_y += np.square(sigmay[pred_step]) * self.args.dt * self.args.dt
										e1 = Ellipse(xy=(traj_pred[pred_step, 0], traj_pred[pred_step, 1]), width=np.sqrt(sigma_x) / 2,
										             height=np.sqrt(sigma_y) / 2,
										             angle=0 / np.pi * 180)
										self.ga3c_plot.add_patch(e1)

						# Plot other agents
						plot_other_agents = True
						if plot_other_agents:
							other_agents = np.expand_dims(other_agents_list[animation_idx][0, step], axis=0)

							for agent_id in range(other_agents.shape[0]):
								self.ga3c_plot.plot(other_agents[agent_id,0], other_agents[agent_id,1], color='b', marker='o')

					# Real trajectory
					self.ga3c_plot.plot(traj_real[:, 0], traj_real[:, 1], color="k",marker='x', label='Real trajectory')

					model_vel_real = np.zeros((self.args.prediction_horizon, self.args.output_dim))
					for i in range(self.args.prediction_horizon):
						mu_x = traj[step+i]["pedestrian_state"]["velocity"][0]
						mu_y = traj[step+i]["pedestrian_state"]["velocity"][1]
						model_vel_real[i, :] = [mu_x, mu_y]
					if False:
						real_vel = sup.rotate_predicted_vel_to_global_frame(model_vel_real, batch_x[step, 2:])
					else:
						real_vel = model_vel_real
					traj_real_from_v = sup.path_from_vel(initial_pos=np.array([x0, y0]),
					                              pred_vel=real_vel, dt=self.args.dt)
					self.ga3c_plot.plot(traj_real_from_v[:, 0], traj_real_from_v[:, 1], color='m', label='Agent real trajectory')


					self.ga3c_plot.set_xlim([-15, 15])
					self.ga3c_plot.set_ylim([-15, 15])
					self.ga3c_plot.axis("on")
					self.ga3c_plot.set_xlabel('x [m]')
					self.ga3c_plot.set_ylabel('y [m]')
					self.ga3c_plot.set_aspect('equal')

					self.fig_ga3c.canvas.draw()
					if not os.path.exists(self.args.model_path + '/results/' + self.args.scenario+"/figs/"):
						os.makedirs(self.args.model_path + '/results/' + self.args.scenario+"/figs/")
					if not os.path.exists(self.args.model_path + '/results/' + self.args.scenario + "/figs_frozen/"):
						os.makedirs(self.args.model_path + '/results/' + self.args.scenario + "/figs_frozen/")
					if test_args.save_figs:
						if test_args.freeze_other_agents:
							self.fig_ga3c.savefig(self.args.model_path + '/results/' + self.args.scenario+"/figs_frozen/result_"+str(animation_idx)+"_"+str(step)+".jpg")
						else:
							self.fig_ga3c.savefig(self.args.model_path + '/results/' + self.args.scenario + "/figs/result_" + str(
								animation_idx) + "_" + str(step) + ".jpg")
					self.writer.grab_frame()