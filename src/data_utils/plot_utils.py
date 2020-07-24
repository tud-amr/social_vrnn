import numpy as np
import os
import random
import sys
import math
import cv2
import pickle as pkl
import time
if sys.version_info[0] < 3:
  sys.path.append('../src/data_utils')
  import Support as sup
else:
	import src.data_utils.Support as sup
import matplotlib.pyplot as pl
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Ellipse

def plot_scenario(enc_seq, dec_seq,traj, predictions, args):
	"""
		inputs:
			enc_seq: Encoder Inouts Sequences
			dec_seq: Decoder input sequences
			traj: Trajectory Groud Truth of Robot and Pedestrians
			predictions: Pedestrians Predicted Trajectory
			global_grid: Static Environment
		"""
	metadata = dict(title='Movie Test', artist='Matplotlib')
	writer = FFMpegWriter(fps=10, metadata=metadata)

	fig = pl.figure("Trajectory Predictions")
	ax_in = pl.subplot()

	ax_in.clear()

	#x_lim = [-global_grid.map_size[0]/2 ,global_grid.map_size[0]/2]
	#y_lim = [-global_grid.map_size[1]/2,global_grid.map_size[1]/2]

	x_lim = [-5 ,5]
	y_lim = [-5,5]

	ax_in.clear()
	pedestrian_trajectory = np.zeros((len(traj),2))
	for t in range(len(traj)):
		pedestrian_trajectory[t,:] = traj[t]["pedestrian_state"]["position"]
	with writer.saving(fig, args.data_path + "/movie.mp4",100):
		for t in range(len(enc_seq)):
				ax_in.clear()
				# plot scenario grid
				#sup.plot_grid(ax_in, np.array([0, 0]), global_grid.gridmap, global_grid.resolution, global_grid.map_size)
				ax_in.plot(pedestrian_trajectory[t:, 0], pedestrian_trajectory[t:, 1], color='red', label='Agent Real Traj')
				ax_in.plot(enc_seq[t][:,0], enc_seq[t][:,1], color='g', marker='o', label='Robot Prev Real Traj')
				ax_in.plot(enc_seq[t][:,4], enc_seq[t][:,5], color='red', marker='o', label='Agent Prev Traj')
				ax_in.plot(dec_seq[t][:, 0], dec_seq[t][:, 1], color='blue',marker='o', label='Agent Real')
				ax_in.plot(predictions[t][:, 0], predictions[t][:, 1], color='blue', marker='x', label='Agent Pred Traj')


				ax_in.set_xlim(x_lim)
				ax_in.set_ylim(y_lim)

				ax_in.legend()
				fig.canvas.draw()
				pl.show(block=False)
				#time.sleep(t)
				writer.grab_frame()

def plot_scenario_vel_tbp(enc_seq, dec_seq,traj, predictions, args):
	"""
		inputs:
			enc_seq: Encoder Inouts Sequences
			dec_seq: Decoder input sequences
			traj: Trajectory Groud Truth of Robot and Pedestrians
			predictions: Pedestrians Predicted Trajectory
			global_grid: Static Environment
		"""
	metadata = dict(title='Movie Test', artist='Matplotlib')
	writer = FFMpegWriter(fps=10, metadata=metadata)

	fig = pl.figure("Trajectory Predictions")
	ax_in = pl.subplot()

	ax_in.clear()

	#x_lim = [-global_grid.map_size[0]/2 ,global_grid.map_size[0]/2]
	#y_lim = [-global_grid.map_size[1]/2,global_grid.map_size[1]/2]

	x_lim = [-5 ,5]
	y_lim = [-5,5]

	ax_in.clear()
	pedestrian_trajectory = np.zeros((len(traj),2))
	for t in range(len(traj)):
		pedestrian_trajectory[t,:] = traj[t]["pedestrian_state"]["position"]
	with writer.saving(fig, args.data_path + "/movie.mp4",100):
		for t in range(len(enc_seq)):
				ax_in.clear()
				# plot scenario grid
				#sup.plot_grid(ax_in, np.array([0, 0]), global_grid.gridmap, global_grid.resolution, global_grid.map_size)
				ax_in.plot(pedestrian_trajectory[t:, 0], pedestrian_trajectory[t:, 1], color='red', label='Agent Real Traj')
				ax_in.plot(enc_seq[t][0,0::args.input_dim], enc_seq[t][0,1::args.input_dim], color='g', marker='o', label='Robot Prev Real Traj')
				ax_in.plot(enc_seq[t][0,4::args.input_dim], enc_seq[t][0,5::args.input_dim], color='red', marker='o', label='Agent Prev Traj')
				ax_in.plot(pedestrian_trajectory[t+args.prev_horizon:t+args.prev_horizon+args.prediction_horizon, 0], pedestrian_trajectory[t+args.prev_horizon:t+args.prev_horizon+args.prediction_horizon, 1], color='blue',marker='o', label='Agent Real')
				x0 = pedestrian_trajectory[t+args.prev_horizon,0]
				y0 = pedestrian_trajectory[t+args.prev_horizon,1]
				pred_vel = np.zeros((args.prediction_horizon,args.output_dim))
				for i in range(args.prediction_horizon):
					idx = i * args.output_dim
					idy = idx + 1
					vx = predictions[t][0,idx]
					vy = predictions[t][0,idy]
					pred_vel[i, :] = [vx, vy]

				traj_pred = sup.path_from_vel(initial_pos=np.array([x0 , y0]),pred_vel=pred_vel, dt=args.dt)
				ax_in.plot(traj_pred[:, 0], traj_pred[:, 1], color='blue', marker='x', label='Agent Pred Traj')
				ax_in.set_xlim(x_lim)
				ax_in.set_ylim(y_lim)

				ax_in.legend()
				fig.canvas.draw()
				pl.show(block=False)
				#time.sleep(t)
				writer.grab_frame()

#def plot_scenario_vel(enc_seq, dec_seq,traj, predictions, args):
	"""
		inputs:
			enc_seq: Encoder Inouts Sequences
			dec_seq: Decoder input sequences
			traj: Trajectory Groud Truth of Robot and Pedestrians
			predictions: Pedestrians Predicted Trajectory
			global_grid: Static Environment
		"""
	"""	metadata = dict(title='Movie Test', artist='Matplotlib')
	writer = FFMpegWriter(fps=1, metadata=metadata)

	fig = pl.figure("Trajectory Predictions")
	ax_in = pl.subplot()

	ax_in.clear()

	#x_lim = [-global_grid.map_size[0]/2 ,global_grid.map_size[0]/2]
	#y_lim = [-global_grid.map_size[1]/2,global_grid.map_size[1]/2]

	x_lim = [-5 ,5]
	y_lim = [-5,5]

	ax_in.clear()
	pedestrian_trajectory = np.zeros((len(traj),2))
	for t in range(len(traj)):
		pedestrian_trajectory[t,:] = traj[t]["pedestrian_state"]["position"]
	with writer.saving(fig, args.data_path + "/movie.mp4",100):
		for t in range(len(enc_seq)):
				ax_in.clear()
				# plot scenario grid
				#sup.plot_grid(ax_in, np.array([0, 0]), global_grid.gridmap, global_grid.resolution, global_grid.map_size)
				ax_in.plot(pedestrian_trajectory[t:, 0], pedestrian_trajectory[t:, 1], color='red', label='Agent Real Traj')
				ax_in.plot(enc_seq[t][:,0], enc_seq[t][:,1], color='g', marker='o', label='Robot Prev Real Traj')
				ax_in.plot(enc_seq[t][:,4], enc_seq[t][:,5], color='red', marker='o', label='Agent Prev Traj')
				ax_in.plot(pedestrian_trajectory[t+args.prev_horizon:t+args.prev_horizon+args.prediction_horizon, 0], pedestrian_trajectory[t+args.prev_horizon:t+args.prev_horizon+args.prediction_horizon, 1], color='blue',marker='o', label='Agent Real')
				x0 = pedestrian_trajectory[t+args.prev_horizon,0]
				y0 = pedestrian_trajectory[t+args.prev_horizon,1]
				pred_vel = predictions[t]
				traj_pred = sup.path_from_vel(initial_pos=np.array([x0 , y0]),pred_vel=pred_vel, dt=0.1)
				ax_in.plot(traj_pred[:, 0], traj_pred[:, 1], color='k', marker='x', label='Agent Pred Traj')
				ax_in.set_xlim(x_lim)
				ax_in.set_ylim(y_lim)

				ax_in.legend()
				fig.canvas.draw()
				pl.show(block=False)
				#time.sleep(t)
				writer.grab_frame()
"""

def plot_scenario_vel_OpenCV_simdata(trajectories, batch_target, all_predictions, args, exp_num=0,display = True):
	"""
		inputs:
			enc_seq: Encoder Inouts Sequences
			dec_seq: Decoder input sequences
			traj: Trajectory Groud Truth of Robot and Pedestrians
			predictions: Pedestrians Predicted Trajectory
			global_grid: Static Environment
	"""
	homography_file = os.path.join(args.data_path + args.scenario, 'H.txt')
	if os.path.exists(homography_file):
		Hinv = np.linalg.inv(np.loadtxt(homography_file))
	else:
		print('[INF] No homography file')
	scenario = args.scenario
	video_file = args.model_path +"/"  + args.model_name+ "_" + str(exp_num) + '.avi'
	img_file = args.data_path + args.scenario +'/map.png'

	if os.path.exists(img_file):
		print('[INF] Using image file ' + img_file)
		img = ~cv2.imread(img_file)
		img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
		height, width, channels = img.shape

		scale_percent = 400  # percent of original size
		width = int(img.shape[1] * scale_percent / 100)
		height = int(img.shape[0] * scale_percent / 100)
		dim = (width, height)
		# resize image
		resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

		# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
		out = cv2.VideoWriter(video_file,
		                      cv2.VideoWriter_fourcc('M', 'P', '4', '2'), 10, (width, height))

		# Adding legend
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.line(resized_img, (width - 150, 30), (width - 130, 30), (0, 255, 0), 4)
		cv2.putText(resized_img, "Query Agent", (width - 120, 30), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
		cv2.line(resized_img, (width - 150, 50), (width - 130, 50), (255, 0, 0), 4)
		cv2.putText(resized_img, "Other Agents", (width - 120, 50), font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
		cv2.line(resized_img, (width - 150, 70), (width - 130, 70), (255, 255, 0), 4)
		cv2.putText(resized_img, "Query Agent traj", (width - 120, 70), font, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
		cv2.line(resized_img, (width - 150, 90), (width - 130, 90), (128, 128, 128), 4)
		cv2.putText(resized_img, "Pred traj 1", (width - 120, 90), font, 0.5, (128, 128, 128), 2, cv2.LINE_AA)
		cv2.line(resized_img, (width - 150, 110), (width - 130, 110), (0, 128, 255), 4)
		cv2.putText(resized_img, "Pred traj 2", (width - 120, 110), font, 0.5, (0, 128, 255), 2, cv2.LINE_AA)
		cv2.line(resized_img, (width - 150, 130), (width - 130, 130), (255, 0, 127), 4)
		cv2.putText(resized_img, "Pred traj 3", (width - 120, 130), font, 0.5, (255, 0, 127), 2, cv2.LINE_AA)

		for exp_idx in range(len(all_predictions)):
			predictions = all_predictions[exp_idx]
			traj = trajectories[exp_idx]
			ped_vel = batch_target[exp_idx][0]

			for t in range(len(predictions)):
				pedestrian_trajectory = traj.pose_vec[:,:2]
				overlay_ellipses = resized_img.copy()

				# Plot Query Agent Positions
				query_agent_pos = traj.pose_vec[t,:2]
				query_agent = np.expand_dims(np.array((query_agent_pos[0], query_agent_pos[1])), axis=0).astype(float)
				obsv_XY = sup.to_image_frame(Hinv, query_agent)*int(scale_percent / 100)
				sigma_x = 0.2
				sigma_y = 0.2
				axis = (
					int(np.sqrt(sigma_x) / args.submap_resolution / scale_percent * 100), int(np.sqrt(sigma_y) / args.submap_resolution / scale_percent * 100))
				center = (obsv_XY[0, 1], obsv_XY[0, 0])
				cv2.ellipse(overlay_ellipses, center, axis, 0, 0, 360, (0, 255, 0), -1)

				# Plot other pedestrians positions
				agent_pos = traj.other_agents_positions[t]
				for agent_id in range(agent_pos.shape[0]):
					ped_pos = np.expand_dims(np.array((agent_pos[agent_id, 0], agent_pos[agent_id, 1])), axis=0).astype(float)
					obsv_XY = sup.to_image_frame(Hinv, ped_pos)*int(scale_percent / 100)
					axis = (
						int(np.sqrt(sigma_x) / args.submap_resolution / scale_percent * 100),
						int(np.sqrt(sigma_y) / args.submap_resolution / scale_percent * 100))
					center = (obsv_XY[0, 1], obsv_XY[0, 0])
					cv2.ellipse(overlay_ellipses, center, axis, 0, 0, 360, (255, 0, 0), 1)

				# Query-agent real trajectory
				obsv_XY = sup.to_image_frame(Hinv, pedestrian_trajectory[t:, :])*int(scale_percent / 100)
				sup.line_cv(overlay_ellipses, obsv_XY, (255, 0, 0), 1)  # bgr convention

				# Query-agent predicted real trajectory from positions
				obsv_XY = sup.to_image_frame(Hinv, pedestrian_trajectory[t:t+args.prediction_horizon, :]) * int(scale_percent / 100)
				sup.line_cv(overlay_ellipses, obsv_XY, (255, 255, 0), 1)  # bgr convention

				#Query-agent Predicted Real TRajectory
				x0 = traj.pose_vec[t+1+args.prev_horizon,0]
				y0 = traj.pose_vec[t+1+args.prev_horizon,1]
				pedestrian_velocity = np.zeros((args.prediction_horizon, args.output_dim))
				for t_idx in range(args.truncated_backprop_length):
					if args.normalize_data:
						pedestrian_velocity[t_idx, 0] = ped_vel[t, t_idx * 2] / args.sx_vel + args.min_vel_x
						pedestrian_velocity[t_idx, 1] = ped_vel[t, t_idx * 2+1] / args.sy_vel + args.min_vel_y
					else:
						pedestrian_velocity[t_idx, :] = ped_vel[t, t_idx * 2:t_idx * 2 + 2]
				pedestrian_traj = sup.path_from_vel(initial_pos=np.array([x0, y0]), pred_vel=pedestrian_velocity, dt=args.dt)
				obsv_XY = sup.to_image_frame(Hinv, pedestrian_traj)*int(scale_percent / 100)
				sup.line_cv(overlay_ellipses, obsv_XY, (255, 255, 0), 1)  # bgr convention

				# Query-agent real trajectory
				obsv_XY = sup.to_image_frame(Hinv, pedestrian_trajectory[t:t+args.prediction_horizon, :])*int(scale_percent / 100)
				sup.line_cv(overlay_ellipses, obsv_XY, (0, 0, 0), 1)  # bgr convention

				# Plot Predictions
				pred_vel = np.zeros((args.prediction_horizon, args.output_dim))
				sigmax = np.zeros((args.prediction_horizon, 1))
				sigmay = np.zeros((args.prediction_horizon, 1))
				colors=[(128,128,128),(0,128,255),(255,0,127)]
				for sample_id in range(len(predictions[t])):
					prediction_sample = predictions[t][sample_id]
					for mix_id in range(args.n_mixtures):
						for pred_step in range(args.prediction_horizon):
							idx = pred_step * args.output_pred_state_dim * args.n_mixtures + mix_id
							if args.normalize_data:
								pred_vel[pred_step, 0] = prediction_sample[0][idx] / args.sx_vel + args.min_vel_x
								pred_vel[pred_step, 1] = prediction_sample[0][idx+args.n_mixtures] / args.sy_vel + args.min_vel_y
							else:
								pred_vel[pred_step, :] = np.array((prediction_sample[0][idx],prediction_sample[0][idx+args.n_mixtures]
							                                   ))
							if args.output_pred_state_dim > args.output_dim:
								sigmax[pred_step, :] = prediction_sample[0][idx + 2*args.n_mixtures]
								sigmay[pred_step, :] = prediction_sample[0][idx + 3*args.n_mixtures]
						try:
							if args.predict_positions:
								traj_pred = pred_vel
							else:
								traj_pred = sup.path_from_vel(initial_pos=np.array([x0, y0]), pred_vel=pred_vel, dt=args.dt)
						except:
							traj_pred = sup.path_from_vel(initial_pos=np.array([x0, y0]), pred_vel=pred_vel, dt=args.dt)
						obsv_XY = sup.to_image_frame(Hinv, traj_pred)*int(scale_percent / 100)
						sup.line_cv(overlay_ellipses, obsv_XY, colors[mix_id], 1)  # bgr convention

						if args.output_pred_state_dim > args.output_dim:
							sigma_x = np.square(sigmax[0]) * (args.dt ** 2)
							sigma_y = np.square(sigmay[0]) * (args.dt ** 2)

							obsv_XY = sup.to_image_frame(Hinv, traj_pred)*int(scale_percent / 100)
							axis = (
								int(np.sqrt(sigma_x) / args.submap_resolution * scale_percent / 100), int(np.sqrt(sigma_y) / args.submap_resolution * scale_percent / 100))
							center = (obsv_XY[0, 1], obsv_XY[0, 0])
							try:
								cv2.ellipse(overlay_ellipses, center, axis, 0, 0, 360, colors[mix_id], 1)
							except:
								print("failed to plot ellipse")

							for pred_step in range(1,args.prediction_horizon):
								sigma_x += np.square(sigmax[pred_step]) * (args.dt ** 2)
								sigma_y += np.square(sigmay[pred_step]) * (args.dt ** 2)
								axis = (
									int(np.sqrt(sigma_x) / args.submap_resolution * scale_percent / 100), int(np.sqrt(sigma_y) / args.submap_resolution * scale_percent / 100))
								center = (obsv_XY[pred_step, 1], obsv_XY[pred_step, 0])
								try:
									cv2.ellipse(overlay_ellipses, center, axis, 0, 0, 360, colors[mix_id], 1)
								except:
									print("failed to plot ellipse")

				#image_new = cv2.addWeighted(resized_img, 0.2, overlay_ellipses, 0.8, 0)
				if display:
					cv2.imshow("image", overlay_ellipses)
					cv2.waitKey(100)
				out.write(overlay_ellipses)

		# When everything done, release the video capture and video write objects
		out.release()

	print("Done recording...")

def plot_local_scenario_vel_OpenCV(trajectories, batch_target, all_predictions, args, exp_num=0,display = True):
	"""
		inputs:
			enc_seq: Encoder Inouts Sequences
			dec_seq: Decoder input sequences
			traj: Trajectory Groud Truth of Robot and Pedestrians
			predictions: Pedestrians Predicted Trajectory
			global_grid: Static Environment
		"""
	homography_file = os.path.join(args.data_path + args.scenario, 'H.txt')
	if os.path.exists(homography_file):
		Hinv = np.linalg.inv(np.loadtxt(homography_file))
	else:
		print('[INF] No homography file')
	scenario = args.scenario
	video_file = args.model_path +"/"  + args.model_name+ "_" + str(exp_num) + '_local.avi'
	img_file = args.data_path + args.scenario +'/map.png'

	if os.path.exists(img_file):
		print('[INF] Using image file ' + img_file)
		img = ~cv2.imread(img_file)

		height, width, channels = img.shape

		scale_percent = 200  # percent of original size
		width = int(img.shape[1] * scale_percent / 100)
		height = int(img.shape[0] * scale_percent / 100)
		dim = (width, height)
		# resize image
		resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

		# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
		out = cv2.VideoWriter(video_file,
		                      cv2.VideoWriter_fourcc('M', 'P', '4', '2'), 10, (600, 600))

		# Adding legend
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.line(resized_img, (width - 150, 30), (width - 130, 30), (0, 255, 0), 4)
		cv2.putText(resized_img, "Query Agent", (width - 120, 30), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
		cv2.line(resized_img, (width - 150, 50), (width - 130, 50), (255, 0, 0), 4)
		cv2.putText(resized_img, "Other Agents", (width - 120, 50), font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
		cv2.line(resized_img, (width - 150, 70), (width - 130, 70), (255, 255, 0), 4)
		cv2.putText(resized_img, "Query Agent traj", (width - 120, 70), font, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
		cv2.line(resized_img, (width - 150, 90), (width - 130, 90), (128, 128, 128), 4)
		cv2.putText(resized_img, "Pred traj 1", (width - 120, 90), font, 0.5, (128, 128, 128), 2, cv2.LINE_AA)
		cv2.line(resized_img, (width - 150, 110), (width - 130, 110), (0, 128, 255), 4)
		cv2.putText(resized_img, "Pred traj 2", (width - 120, 110), font, 0.5, (0, 128, 255), 2, cv2.LINE_AA)
		cv2.line(resized_img, (width - 150, 130), (width - 130, 130), (255, 0, 127), 4)
		cv2.putText(resized_img, "Pred traj 3", (width - 120, 130), font, 0.5, (255, 0, 127), 2, cv2.LINE_AA)

		for exp_idx in range(len(all_predictions)):
			predictions = all_predictions[exp_idx]
			traj = trajectories[exp_idx]
			ped_vel = batch_target[exp_idx][0]

			for t in range(len(predictions)):
				pedestrian_trajectory = traj.pose_vec[:,:2]
				overlay_ellipses = resized_img.copy()

				# Plot Query Agent Positions
				query_agent_pos = traj.pose_vec[t,:2]
				query_agent = np.expand_dims(np.array((query_agent_pos[0], query_agent_pos[1])), axis=0).astype(float)
				obsv_XY = sup.to_image_frame(Hinv, query_agent)*int(scale_percent / 100)
				sigma_x = 0.4
				sigma_y = 0.4
				axis = (
					int(np.sqrt(sigma_x) / args.submap_resolution / scale_percent * 100), int(np.sqrt(sigma_y) / args.submap_resolution / scale_percent * 100))
				agent_center = (obsv_XY[0, 1], obsv_XY[0, 0])
				cv2.ellipse(overlay_ellipses, agent_center, axis, 0, 0, 360, (0, 255, 0), -1)

				# Plot ped real position
				agent_pos = traj.other_agents_positions[t]
				for agent_id in range(agent_pos.shape[0]):
					ped_pos = np.expand_dims(np.array((agent_pos[agent_id, 0], agent_pos[agent_id, 1])), axis=0).astype(float)
					obsv_XY = sup.to_image_frame(Hinv, ped_pos)*int(scale_percent / 100)
					axis = (
						int(np.sqrt(sigma_x) / args.submap_resolution / scale_percent * 100),
						int(np.sqrt(sigma_y) / args.submap_resolution / scale_percent * 100))
					center = (obsv_XY[0, 1], obsv_XY[0, 0])
					cv2.ellipse(overlay_ellipses, center, axis, 0, 0, 360, (255, 0, 0), -1)

				# Query-agent real trajectory
				obsv_XY = sup.to_image_frame(Hinv, pedestrian_trajectory[t:, :])*int(scale_percent / 100)
				sup.line_cv(overlay_ellipses, obsv_XY, (255, 0, 0), 1)  # bgr convention

				#Query-agent Predicted Real TRajectory
				x0 = traj.pose_vec[t+1+args.prev_horizon,0]
				y0 = traj.pose_vec[t+1+args.prev_horizon,1]
				pedestrian_velocity = np.zeros((args.prediction_horizon, args.output_dim))
				for t_idx in range(args.truncated_backprop_length):
					if args.normalize_data:
						pedestrian_velocity[t_idx, 0] = ped_vel[t, t_idx * 2] / args.sx_vel + args.min_vel_x
						pedestrian_velocity[t_idx, 1] = ped_vel[t, t_idx * 2+1] / args.sy_vel + args.min_vel_y
					else:
						pedestrian_velocity[t_idx, :] = ped_vel[t, t_idx * 2:t_idx * 2 + 2]
				pedestrian_trajectory = sup.path_from_vel(initial_pos=np.array([x0, y0]), pred_vel=pedestrian_velocity, dt=args.dt)
				obsv_XY = sup.to_image_frame(Hinv, pedestrian_trajectory)*int(scale_percent / 100)
				sup.line_cv(overlay_ellipses, obsv_XY, (255, 255, 0), 1)  # bgr convention


				# Plot Predictions
				pred_vel = np.zeros((args.prediction_horizon, args.output_dim))
				sigmax = np.zeros((args.prediction_horizon, 1))
				sigmay = np.zeros((args.prediction_horizon, 1))
				colors=[(128,128,128),(0,128,255),(255,0,127)]
				for sample_id in range(len(predictions[t])):
					prediction_sample = predictions[t][sample_id]
					for mix_id in range(args.n_mixtures):
						for pred_step in range(args.prediction_horizon):
							idx = pred_step * args.output_pred_state_dim * args.n_mixtures + mix_id
							if args.normalize_data:
								pred_vel[pred_step, 0] = prediction_sample[0][idx] / args.sx_vel + args.min_vel_x
								pred_vel[pred_step, 1] = prediction_sample[0][idx+args.n_mixtures] / args.sy_vel + args.min_vel_y
							else:
								pred_vel[pred_step, :] = np.array((prediction_sample[0][idx],prediction_sample[0][idx+args.n_mixtures]))
							if args.output_pred_state_dim > args.output_dim:
								sigmax[pred_step, :] = prediction_sample[0][idx + 2*args.n_mixtures]
								sigmay[pred_step, :] = prediction_sample[0][idx + 3*args.n_mixtures]
						try:
							if args.predict_positions:
								traj_pred = pred_vel
							else:
								traj_pred = sup.path_from_vel(initial_pos=np.array([x0, y0]), pred_vel=pred_vel, dt=args.dt)
						except:
							traj_pred = sup.path_from_vel(initial_pos=np.array([x0, y0]), pred_vel=pred_vel, dt=args.dt)
						obsv_XY = sup.to_image_frame(Hinv, traj_pred)*int(scale_percent / 100)
						sup.line_cv(overlay_ellipses, obsv_XY, colors[mix_id], 1)  # bgr convention

						if args.output_pred_state_dim > args.output_dim:
							sigma_x = np.square(sigmax[0]) * (args.dt ** 2)
							sigma_y = np.square(sigmay[0]) * (args.dt ** 2)

							obsv_XY = sup.to_image_frame(Hinv, traj_pred)*int(scale_percent / 100)
							axis = (
								int(np.sqrt(sigma_x) / args.submap_resolution * scale_percent / 100), int(np.sqrt(sigma_y) / args.submap_resolution * scale_percent / 100))
							center = (obsv_XY[0, 1], obsv_XY[0, 0])
							cv2.ellipse(overlay_ellipses, center, axis, 0, 0, 360, colors[mix_id], 1)

							for pred_step in range(1,args.prediction_horizon):
								sigma_x += np.square(sigmax[pred_step]) * (args.dt ** 2)
								sigma_y += np.square(sigmay[pred_step]) * (args.dt ** 2)
								axis = (
									int(np.sqrt(sigma_x) / args.submap_resolution * scale_percent / 100), int(np.sqrt(sigma_y) / args.submap_resolution * scale_percent / 100))
								center = (obsv_XY[pred_step, 1], obsv_XY[pred_step, 0])
								try:
									cv2.ellipse(overlay_ellipses, center, axis, 0, 0, 360, colors[mix_id], 1)
								except:
									print("failed to plot ellipse")

				crop_img = cv2.getRectSubPix(overlay_ellipses,(600,600),agent_center)
				if display:
					cv2.imshow("image", crop_img)
					cv2.waitKey(100)
				out.write(crop_img)

		# When everything done, release the video capture and video write objects
		out.release()

	print("Done recording...")

def plot_scenario_vel_OpenCV(trajectories, batch_target, all_predictions, args, exp_num=0,display = False):
	"""
		inputs:
			enc_seq: Encoder Inouts Sequences
			dec_seq: Decoder input sequences
			traj: Trajectory Groud Truth of Robot and Pedestrians
			predictions: Pedestrians Predicted Trajectory
			global_grid: Static Environment
		"""
	homography_file = os.path.join(args.data_path+ args.scenario, 'H.txt')
	if os.path.exists(homography_file):
		Hinv = np.linalg.inv(np.loadtxt(homography_file))
	else:
		print('[INF] No homography file')
	scenario = args.scenario
	video_file = args.model_path +"/"  + args.model_name+ "_" + str(exp_num) + '.avi'
	map_resolution = 1 / Hinv[0, 0]
	print("Recording to: " + video_file)

	img_file = args.data_path+ args.scenario + '/map.png'

	if os.path.exists(img_file):
		print('[INF] Using image file ' + img_file)
		img = cv2.imread(img_file)

		#height, width, channels = img.shape
		width = 600
		height = 600
		dim = (width, height)

		# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
		out = cv2.VideoWriter(video_file,
		                      cv2.VideoWriter_fourcc('M', 'P', '4', '2'), 10, (width, height))

		for exp_idx in range(len(all_predictions)):
			predictions = all_predictions[exp_idx]
			traj = trajectories[exp_idx]
			ped_vel = batch_target[exp_idx][0]

			pedestrian_trajectory = np.zeros((len(traj), 2))
			for t in range(len(traj)):
				pedestrian_trajectory[t, :] = traj[t]["pedestrian_state"]["position"]

			for t in range(len(predictions)):
				overlay_ellipses = img.copy()

				# Plot Robot Positions
				robot_pos = traj[t]["robot_state"]
				robot = np.expand_dims(np.array((robot_pos[0], robot_pos[1])), axis=0).astype(float)
				obsv_XY = sup.to_image_frame(Hinv, robot)
				sigma_x = 0.4
				sigma_y = 0.4
				axis_agent = (
					int(np.sqrt(sigma_x) / map_resolution), int(np.sqrt(sigma_y) / map_resolution))
				center = (obsv_XY[0, 0], obsv_XY[0, 1])
				cv2.ellipse(overlay_ellipses, center, axis_agent, 0, 0, 360, (0, 0, 0), -1)

				# Plot ped real position
				agent_pos = traj[t]["pedestrian_state"]["position"]
				ped_pos = np.expand_dims(np.array((agent_pos[0], agent_pos[1])), axis=0).astype(float)
				obsv_XY = sup.to_image_frame(Hinv, ped_pos)

				center = (obsv_XY[0, 0], obsv_XY[0, 1])
				cv2.ellipse(overlay_ellipses, center, axis_agent, 0, 0, 360, (255, 0, 0), -1)

				# Pedestrian real trajectory
				obsv_XY = sup.to_image_frame(Hinv, pedestrian_trajectory[t:, :])
				sup.line_cv(overlay_ellipses, obsv_XY, (255, 0, 0), 3)  # bgr convention

				#Pedestrian Predicted Real TRajectory
				obsv_XY = sup.to_image_frame(Hinv, pedestrian_trajectory[t + args.prev_horizon:t + args.prev_horizon + args.prediction_horizon, :])
				sup.line_cv(overlay_ellipses, obsv_XY, (255, 255, 0), 3)  # bgr convention

				#Pedestrian Predicted Real TRajectory
				x0 = pedestrian_trajectory[t, 0]
				y0 = pedestrian_trajectory[t, 1]
				pedestrian_velocity = np.zeros((args.prediction_horizon, args.output_dim))
				for t_idx in range(args.truncated_backprop_length):
					pedestrian_velocity[t_idx, :] = ped_vel[t, t_idx * 2:t_idx * 2 + 2]
				pedestrian_traj = sup.path_from_vel(initial_pos=np.array([x0, y0]), pred_vel=pedestrian_velocity, dt=args.dt)
				obsv_XY = sup.to_image_frame(Hinv, pedestrian_traj)
				sup.line_cv(overlay_ellipses, obsv_XY, (0, 0, 255), 3)  # bgr convention

				# Plot Predictions
				x0 = pedestrian_trajectory[t, 0]
				y0 = pedestrian_trajectory[t, 1]
				pred_vel = np.zeros((args.prediction_horizon, args.output_dim))
				sigmax = np.zeros((args.prediction_horizon, 1))
				sigmay = np.zeros((args.prediction_horizon, 1))
				colors=[(128,128,128),(0,128,255),(255,0,127)]
				for sample_id in range(len(predictions[t])):
					prediction_sample = predictions[t][sample_id]
					for mix_id in range(args.n_mixtures):
						for pred_step in range(args.prediction_horizon):
							idx = pred_step * args.output_pred_state_dim * args.n_mixtures + mix_id
							pred_vel[pred_step, :] = np.array((prediction_sample[0][idx],prediction_sample[0][idx+args.n_mixtures]
							                                   ))
							if args.output_pred_state_dim > args.output_dim:
								sigmax[pred_step, :] = prediction_sample[0][idx + 2*args.n_mixtures]
								sigmay[pred_step, :] = prediction_sample[0][idx + 3*args.n_mixtures]
						traj_pred = sup.path_from_vel(initial_pos=np.array([x0, y0]), pred_vel=pred_vel, dt=args.dt)
						obsv_XY = sup.to_image_frame(Hinv, traj_pred)
						sup.line_cv(overlay_ellipses, obsv_XY, colors[mix_id], 3)  # bgr convention

						if args.output_pred_state_dim > args.output_dim:
							sigma_x = np.square(np.exp(sigmax[0])) * (args.dt ** 2)
							sigma_y = np.square(np.exp(sigmay[0])) * (args.dt ** 2)

							obsv_XY = sup.to_image_frame(Hinv, traj_pred)
							axis = (
								int(np.sqrt(sigma_x) / map_resolution), int(np.sqrt(sigma_y) / map_resolution))
							#axis = (int(axis[0] + axis_agent[0]),int(axis[1] + axis_agent[1]))
							center = (obsv_XY[0, 0], obsv_XY[0, 1])
							cv2.ellipse(overlay_ellipses, center, axis, 0, 0, 360, colors[mix_id], -1)

							for pred_step in range(1,args.prediction_horizon):
								sigma_x += np.square(np.exp(sigmax[pred_step])) * (args.dt ** 2)
								sigma_y += np.square(np.exp(sigmay[pred_step])) * (args.dt ** 2)
								axis = (
									int(np.sqrt(sigma_x) / map_resolution), int(np.sqrt(sigma_y) / map_resolution))
								#axis = (int(axis[0] + axis_agent[0]), int(axis[1] + axis_agent[1]))
								center = (obsv_XY[pred_step, 0], obsv_XY[pred_step, 1])
								cv2.ellipse(overlay_ellipses, center, axis, 0, 0, 360, colors[mix_id], -1)

				image_new = cv2.addWeighted(img, 0.2, overlay_ellipses, 0.8, 0)

				# resize image
				resized_img = cv2.resize(image_new, dim, interpolation=cv2.INTER_AREA)
				if display:
					cv2.imshow("image", resized_img)
					cv2.waitKey(100)
				out.write(resized_img)

	# When everything done, release the video capture and video write objects
	out.release()

	print("Done recording...")

def plot_scenario_vel(trajectories, all_predictions, args, grid = None):
	"""
		inputs:
			enc_seq: Encoder Inouts Sequences
			dec_seq: Decoder input sequences
			traj: Trajectory Groud Truth of Robot and Pedestrians
			predictions: Pedestrians Predicted Trajectory
			global_grid: Static Environment
		"""
	metadata = dict(title='Movie Test', artist='Matplotlib')
	writer = FFMpegWriter(fps=2, metadata=metadata)

	fig = pl.figure("Trajectory Predictions")
	ax_in = pl.subplot()

	ax_in.clear()

	x_lim = [-5 ,5]
	y_lim = [-5,5]

	ax_in.clear()

	with writer.saving(fig, args.data_path + "/movie.mp4",100):

		for exp_idx in range(len(all_predictions)):
			predictions = all_predictions[exp_idx]
			traj = trajectories[exp_idx]

			pedestrian_trajectory = np.zeros((len(traj), 2))
			for t in range(len(traj)):
				pedestrian_trajectory[t, :] = traj[t]["pedestrian_state"]["position"]

			for t in range(len(predictions)):
				ax_in.clear()
				# plot scenario grid
				#sup.plot_grid(ax_in, np.array([0, 0]), global_grid.gridmap, global_grid.resolution, global_grid.map_size)
				ax_in.plot(pedestrian_trajectory[t:, 0], pedestrian_trajectory[t:, 1], color='red', label='Agent Real Traj')
				robot_pos = traj[t]["robot_state"]
				ax_in.plot(robot_pos[0], robot_pos[1],
				           color='g', marker='o', label='Robot')
				agent_pos = traj[t]["pedestrian_state"]["position"]
				ax_in.plot(agent_pos[0], agent_pos[1],
				           color='red', marker='o', label='Agent')

				if grid == None:
					gridmap = np.zeros((int((x_lim[1] - x_lim[0]) / map_resolution),
					                   int((y_lim[1] - y_lim[0]) / map_resolution)))

				sup.add_other_agents_to_grid(gridmap,robot_pos,args)

				sup.plot_grid(ax_in, np.array([0.0, 0.0]), gridmap, map_resolution,
				              np.array((10,10)))

				ax_in.plot(pedestrian_trajectory[t+args.prev_horizon:t+args.prev_horizon+args.prediction_horizon, 0], pedestrian_trajectory[t+args.prev_horizon:t+args.prev_horizon+args.prediction_horizon, 1], color='blue',marker='o', label='Agent Real')

				x0 = pedestrian_trajectory[t+args.prev_horizon,0]
				y0 = pedestrian_trajectory[t+args.prev_horizon,1]
				pred_vel = np.zeros((args.prediction_horizon,args.output_dim))
				for i in range(args.prediction_horizon):
					pred_vel[i,:] = predictions[t][i*args.output_pred_state_dim:i*args.output_pred_state_dim+2]
				traj_pred = sup.path_from_vel(initial_pos=np.array([x0 , y0]),pred_vel=pred_vel, dt=args.dt)
				ax_in.plot(traj_pred[:, 0], traj_pred[:, 1], color='k', marker='x', label='Agent Pred Traj')

				sigma_x = np.square(predictions[t][2]) * (args.dt **2)
				sigma_y = np.square(predictions[t][3]) * (args.dt **2)

				e1 = Ellipse(xy=(traj_pred[0, 0], traj_pred[0, 1]), width=np.sqrt(sigma_x) / 2,
				             height=np.sqrt(sigma_y) / 2, angle=0 / np.pi * 180)
				e1.set_alpha(0.5)
				ax_in.add_patch(e1)

				for pred_step in range(args.prediction_horizon):
					for mix_idx in range(args.n_mixtures):
						sigma_x += np.square(predictions[t][args.n_mixtures * pred_step * args.output_pred_state_dim +
						                                          mix_idx * args.output_pred_state_dim +2]) * (args.dt **2)
						sigma_y += np.square(predictions[t][args.n_mixtures * pred_step * args.output_pred_state_dim +
						                                          mix_idx * args.output_pred_state_dim +3])* (args.dt **2)

						e1 = Ellipse(xy=(traj_pred[pred_step, 0], traj_pred[pred_step, 1]), width=np.sqrt(sigma_x) / 2,
						             height=np.sqrt(sigma_y) / 2,
						             angle=0 / np.pi * 180)
						ax_in.add_patch(e1)

				ax_in.set_xlim(x_lim)
				ax_in.set_ylim(y_lim)

				ax_in.legend()
				fig.canvas.draw()
				pl.show(block=False)
				time.sleep(0.1)
				writer.grab_frame()

def plot_batch(batch_x, batch_grid, batch_ped_grid, batch_y, other_agents_pos,args):
	"""
		inputs:
			enc_seq: Encoder Inouts Sequences
			dec_seq: Decoder input sequences
			traj: Trajectory Groud Truth of Robot and Pedestrians
			predictions: Pedestrians Predicted Trajectory
			global_grid: Static Environment
		"""

	fig = pl.figure("Batch Data")
	ax_in = pl.subplot()

	ax_in.clear()

	x_lim = [-args.grid_width*args.submap_resolution/2 ,args.grid_width*args.submap_resolution/2]
	y_lim = [-args.grid_height*args.submap_resolution/2,args.grid_height*args.submap_resolution/2]
	x_lim = [-5,5]
	y_lim = [-5, 5]
	ax_in.clear()

	for batch_id in range(batch_x.shape[0]):
		for t_step in range(batch_x.shape[1]):
			ax_in.clear()
			# plot scenario grid
			sup.plot_grid(ax_in, np.array([0, 0]), batch_grid[batch_id,t_step,:,:],
			              args.submap_resolution, np.array((args.grid_width,args.grid_height))*args.submap_resolution)

			robot_pos = other_agents_pos[batch_id,t_step,:]
			agent_pos = batch_x[batch_id, t_step, :2]
			ax_in.plot(robot_pos[0]-agent_pos[0], robot_pos[1]-agent_pos[1],
				           color='g', marker='o', label='Robot')

			ax_in.plot(0, 0,
				           color='red', marker='o', label='Agent')

			ax_in.set_xlim(x_lim)
			ax_in.set_ylim(y_lim)

			ax_in.legend()
			fig.canvas.draw()
			pl.show(block=False)

def plot_batch_OpenCV(step,batch_x, batch_grid, batch_ped_grid, batch_y, other_agents_pos,_model_prediction, args):
	"""
		inputs:
			enc_seq: Encoder Inouts Sequences
			dec_seq: Decoder input sequences
			traj: Trajectory Groud Truth of Robot and Pedestrians
			predictions: Pedestrians Predicted Trajectory
			global_grid: Static Environment
		"""
	homography_file = os.path.join(args.data_path+ args.scenario, 'H.txt')
	if os.path.exists(homography_file):
		Hinv = np.linalg.inv(np.loadtxt(homography_file))
	else:
		print('[INF] No homography file')

	video_file = args.model_path +"/"  + args.model_name+ "_" + str(step) + '_training.avi'
	img_file = args.data_path+ args.scenario + '/map.png'
	font = cv2.FONT_HERSHEY_SIMPLEX
	if os.path.exists(img_file):
		print('[INF] Using image file ' + img_file)
		img = cv2.imread(img_file)

		height, width, channels = img.shape

		# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
		recorder = cv2.VideoWriter(video_file,
		                      cv2.VideoWriter_fourcc('M', 'P', '4', '2'), 10, (width, height))

		# Adding legend
		cv2.putText(img, str(step), (30, 30), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
		cv2.line(img, (850, 30), (900, 30), (0, 0, 0), 4)
		cv2.putText(img, "Robot Pos", (910, 30), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
		cv2.line(img, (850, 50), (900, 50), (255, 0, 0), 4)
		cv2.putText(img, "Pedestrian", (910, 50), font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
		cv2.line(img, (850, 70), (900, 70), (255, 255, 0), 4)
		cv2.putText(img, "Pedestrian traj", (910, 70), font, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
		cv2.line(img, (850, 90), (900, 90), (128,128,128), 4)
		cv2.putText(img, "Pred traj 1", (910, 90), font, 0.5, (128,128,128), 2, cv2.LINE_AA)
		cv2.line(img, (850, 110), (900, 110), (0,128,255), 4)
		cv2.putText(img, "Pred traj 2", (910, 110), font, 0.5, (0,128,255), 2, cv2.LINE_AA)
		cv2.line(img, (850, 130), (900, 130), (255,0,127), 4)
		cv2.putText(img, "Pred traj 3", (910, 130), font, 0.5, (255,0,127), 2, cv2.LINE_AA)

		for batch_idx in range(batch_x.shape[0]):
			if "keras" in args.model_name:
				predictions = _model_prediction[batch_idx]
			else:
				predictions = np.transpose(_model_prediction,(1,0,2))[batch_idx]
			ped_vel = batch_y[batch_idx]

			for t in range(predictions.shape[0]):
				overlay_ellipses = img.copy()

				# Plot Robot Positions
				robot_pos = batch_x[batch_idx,t,:2]
				robot = np.expand_dims(np.array((robot_pos[0], robot_pos[1])), axis=0).astype(float)
				obsv_XY = sup.to_image_frame(Hinv, robot)
				sigma_x = 0.4
				sigma_y = 0.4
				axis = (
					int(np.sqrt(sigma_x) / args.submap_resolution), int(np.sqrt(sigma_y) / args.submap_resolution))
				center = (obsv_XY[0, 1], obsv_XY[0, 0])
				cv2.ellipse(overlay_ellipses, center, axis, 0, 0, 360, (0, 0, 0), -1)

				# Plot ped real position
				agent_pos = other_agents_pos[batch_idx][t]
				for agent_id in range(agent_pos.shape[0]):
					ped_pos = np.expand_dims(np.array((agent_pos[agent_id,0], agent_pos[agent_id,1])), axis=0).astype(float)
					obsv_XY = sup.to_image_frame(Hinv, ped_pos)
					axis = (
						int(np.sqrt(sigma_x) / args.submap_resolution), int(np.sqrt(sigma_y) / args.submap_resolution))
					center = (obsv_XY[0, 1], obsv_XY[0, 0])
					cv2.ellipse(overlay_ellipses, center, axis, 0, 0, 360, (255, 0, 0), -1)

				#Pedestrian Predicted Real TRajectory
				x0 = batch_x[batch_idx, t, 0]
				y0 = batch_x[batch_idx, t, 1]
				pedestrian_velocity = np.zeros((args.prediction_horizon, args.output_dim))
				for t_idx in range(args.truncated_backprop_length):
					pedestrian_velocity[t_idx, :] = ped_vel[t, t_idx * 2:t_idx * 2 + 2]
				pedestrian_trajectory = sup.path_from_vel(initial_pos=np.array([x0, y0]), pred_vel=pedestrian_velocity, dt=args.dt)
				obsv_XY = sup.to_image_frame(Hinv, pedestrian_trajectory)
				sup.line_cv(overlay_ellipses, obsv_XY, (255, 255, 0), 3)  # bgr convention

				# Plot Predictions
				pred_vel = np.zeros((args.prediction_horizon, args.output_dim))
				sigmax = np.zeros((args.prediction_horizon, 1))
				sigmay = np.zeros((args.prediction_horizon, 1))
				# orange grey and purple
				colors=[(128,128,128),(0,128,255),(255,0,127)]
				for mix_id in range(args.n_mixtures):
					for pred_step in range(args.prediction_horizon):
						idx = pred_step * args.output_pred_state_dim * args.n_mixtures + mix_id
						pred_vel[pred_step, :] = np.array((predictions[t][idx],predictions[t][idx+args.n_mixtures]
						                                   ))
						if args.output_pred_state_dim > args.output_dim:
							sigmax[pred_step, :] = predictions[t][idx + 2*args.n_mixtures]
							sigmay[pred_step, :] = predictions[t][idx + 3*args.n_mixtures]
					traj_pred = sup.path_from_vel(initial_pos=np.array([x0, y0]), pred_vel=pred_vel, dt=args.dt)
					obsv_XY = sup.to_image_frame(Hinv, traj_pred)
					sup.line_cv(overlay_ellipses, obsv_XY, colors[mix_id], 3)  # bgr convention

					if args.output_pred_state_dim > args.output_dim:
						#Propagate uncertainty and plot
						sigma_x = np.square(sigmax[0]) * (args.dt ** 2)
						sigma_y = np.square(sigmay[0]) * (args.dt ** 2)

						obsv_XY = sup.to_image_frame(Hinv, traj_pred)
						axis = (
							int(np.sqrt(sigma_x) / args.submap_resolution), int(np.sqrt(sigma_y) / args.submap_resolution))
						center = (obsv_XY[0, 1], obsv_XY[0, 0])
						cv2.ellipse(overlay_ellipses, center, axis, 0, 0, 360, colors[mix_id], -1)

						for pred_step in range(1,args.prediction_horizon):
							sigma_x += np.square(sigmax[pred_step]) * (args.dt ** 2)
							sigma_y += np.square(sigmay[pred_step]) * (args.dt ** 2)
							axis = (
								int(np.sqrt(sigma_x) / args.submap_resolution), int(np.sqrt(sigma_y) / args.submap_resolution))
							center = (obsv_XY[pred_step, 1], obsv_XY[pred_step, 0])
							cv2.ellipse(overlay_ellipses, center, axis, 0, 0, 360, colors[mix_id], -1)

				image_new = cv2.addWeighted(img, 0.2, overlay_ellipses, 0.8, 0)
				width = 1200
				height = 1200
				dim = (width, height)
				# resize image
				resized_img = cv2.resize(image_new, dim, interpolation=cv2.INTER_AREA)
				if False:
					cv2.imshow("image",resized_img)
					cv2.waitKey(100)
				recorder.write(resized_img)
		recorder.release()

def plot_local_scenario_vel_OpenCV2(trajectories, batch_target, all_predictions, args, exp_num=0, display = True):
	"""
		inputs:
			enc_seq: Encoder Inouts Sequences
			dec_seq: Decoder input sequences
			traj: Trajectory Groud Truth of Robot and Pedestrians
			predictions: Pedestrians Predicted Trajectory
			global_grid: Static Environment
		"""
	homography_file = os.path.join(args.data_path + args.scenario, 'H.txt')
	if os.path.exists(homography_file):
		Hinv = np.linalg.inv(np.loadtxt(homography_file))
	else:
		print('[INF] No homography file')
	scenario = args.scenario
	video_file = args.model_path +"/"  + args.model_name+ "_" + str(exp_num) + '_local.avi'
	img_file = args.data_path + args.scenario +'/map.png'

	if os.path.exists(img_file):
		print('[INF] Using image file ' + img_file)
		img = ~cv2.imread(img_file)

		scale_percent = 100  # percent of original size
		width = 300
		height = 300
		dim = (width, height)
		# resize image
		resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

		# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
		out = cv2.VideoWriter(video_file,
		                      cv2.VideoWriter_fourcc('M', 'P', '4', '2'), 10, (width, height))

		# Adding legend
		font = cv2.FONT_HERSHEY_SIMPLEX

		for exp_idx in range(len(all_predictions)):
			predictions = all_predictions[exp_idx]
			traj = trajectories[exp_idx]
			ped_vel = batch_target[exp_idx][0]

			for t in range(len(predictions)):
				pedestrian_trajectory = traj.pose_vec[:,:2]

				# Plot Query Agent Positions
				query_agent_pos = traj.pose_vec[t,:2]
				query_agent = np.expand_dims(np.array((query_agent_pos[0], query_agent_pos[1])), axis=0).astype(float)
				obsv_XY = sup.to_image_frame(Hinv, query_agent)*int(scale_percent / 100)
				sigma_x = 0.4
				sigma_y = 0.4
				axis = (
					int(np.sqrt(sigma_x) / args.submap_resolution), int(np.sqrt(sigma_y) / args.submap_resolution))
				query_agent_center = np.array([obsv_XY[0, 0], obsv_XY[0, 1]])
				axis_pos = query_agent_center - np.array([args.submap_width/args.submap_resolution/2,args.submap_height/args.submap_resolution/2])
				overlay_ellipses = resized_img.copy()
				# Crop image for obtain surrounding static environment
				crop_img = cv2.getRectSubPix(overlay_ellipses,(int(args.submap_height / args.submap_resolution),int(args.submap_width / args.submap_resolution)),(query_agent_center[0],query_agent_center[1]))

				# resize image
				resized_crop_img = cv2.resize(crop_img, dim, interpolation=cv2.INTER_AREA)

				cv2.line(resized_crop_img, (width - 150, 30), (width - 130, 30), (0, 255, 0), 4)
				cv2.putText(resized_crop_img, "Query Agent", (width - 120, 30), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
				cv2.line(resized_crop_img, (width - 150, 50), (width - 130, 50), (255, 0, 0), 4)
				cv2.putText(resized_crop_img, "Other Agents", (width - 120, 50), font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
				cv2.line(resized_crop_img, (width - 150, 70), (width - 130, 70), (255, 255, 0), 4)
				cv2.putText(resized_crop_img, "Query Agent traj", (width - 120, 70), font, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
				cv2.line(resized_crop_img, (width - 150, 90), (width - 130, 90), (128, 128, 128), 4)
				cv2.putText(resized_crop_img, "Pred traj 1", (width - 120, 90), font, 0.5, (128, 128, 128), 2, cv2.LINE_AA)
				cv2.line(resized_crop_img, (width - 150, 110), (width - 130, 110), (0, 128, 255), 4)
				cv2.putText(resized_crop_img, "Pred traj 2", (width - 120, 110), font, 0.5, (0, 128, 255), 2, cv2.LINE_AA)
				cv2.line(resized_crop_img, (width - 150, 130), (width - 130, 130), (255, 0, 127), 4)
				cv2.putText(resized_crop_img, "Pred traj 3", (width - 120, 130), font, 0.5, (255, 0, 127), 2, cv2.LINE_AA)

				# Plot Query Agent Positions
				center = (dim[0]//2, dim[1]//2)
				cv2.ellipse(resized_crop_img, center, axis, 0, 0, 360, (0, 255, 0), -1)

				# Plot other pedestrians
				agent_pos = traj.other_agents_positions[t]
				for agent_id in range(agent_pos.shape[0]):
					ped_pos = np.expand_dims(np.array((agent_pos[agent_id, 0], agent_pos[agent_id, 1])), axis=0).astype(float)
					obsv_XY = sup.to_image_frame(Hinv, ped_pos)*int(scale_percent / 100)
					axis = (
						int(np.sqrt(sigma_x) / args.submap_resolution / scale_percent * 100),
						int(np.sqrt(sigma_y) / args.submap_resolution / scale_percent * 100))
					center = (np.array([obsv_XY[0, 0], obsv_XY[0, 1]]) - axis_pos)
					if (center[0]>0) & (center[0]<width) & (center[1]>0) & (center[1]<height):
						agent_center = (int(center[0]),int(center[1]))
						cv2.ellipse(resized_crop_img, agent_center, axis, 0, 0, 360, (0, 0, 0), -1)

				# Query-agent real trajectory
				obsv_XY = (sup.to_image_frame(Hinv, pedestrian_trajectory[t:, :])- axis_pos)
				obsv_XY_filtered = []
				for i in range(obsv_XY.shape[0]):
					if (obsv_XY[i,0]>0) & (obsv_XY[i,0]<dim[0]) & (obsv_XY[i,1]>0) & (obsv_XY[i,1]<dim[1]):
						obsv_XY_filtered.append(obsv_XY[i])
				sup.line_cv(resized_crop_img, np.asarray(obsv_XY_filtered), (255, 0, 0), 1)  # bgr convention

				#Query-agent Predicted Real TRajectory
				pedestrian_velocity = np.zeros((args.prediction_horizon, args.output_dim))
				for t_idx in range(args.prediction_horizon):
					pedestrian_velocity[t_idx, :] = ped_vel[t, t_idx * 2:t_idx * 2 + 2]
				pedestrian_trajectory = sup.path_from_vel(initial_pos=np.array([0, 0]), pred_vel=pedestrian_velocity, dt=args.dt)/args.submap_resolution
				obsv_XY = (sup.to_image_frame(Hinv, pedestrian_trajectory)- axis_pos)*int(scale_percent / 100)
				obsv_XY_filtered = []
				for i in range(obsv_XY.shape[0]):
					if (obsv_XY[i, 0] > 0) & (obsv_XY[i, 0] < dim[0]) & (obsv_XY[i, 1] > 0) & (obsv_XY[i, 1] < dim[1]):
						obsv_XY_filtered.append(obsv_XY[i])
				sup.line_cv(resized_crop_img, np.asarray(obsv_XY_filtered), (255, 255, 0), 1)  # bgr convention

				# Plot Predictions
				pred_vel = np.zeros((args.prediction_horizon, args.output_dim))
				sigmax = np.zeros((args.prediction_horizon, 1))
				sigmay = np.zeros((args.prediction_horizon, 1))
				colors=[(128,128,128),(0,128,255),(255,0,127)]
				""""""
				for mix_id in range(args.n_mixtures):
					for pred_step in range(args.prediction_horizon):
						idx = pred_step * args.output_pred_state_dim * args.n_mixtures + mix_id
						pred_vel[pred_step, :] = np.array((predictions[t][idx],predictions[t][idx+args.n_mixtures]
						                                   ))
						if args.output_pred_state_dim > args.output_dim:
							sigmax[pred_step, :] = predictions[t][idx + 2*args.n_mixtures]
							sigmay[pred_step, :] = predictions[t][idx + 3*args.n_mixtures]
					traj_pred = sup.path_from_vel(initial_pos=np.array([0, 0]), pred_vel=pred_vel, dt=args.dt)/args.submap_resolution
					obsv_XY = (sup.to_image_frame(Hinv, traj_pred)- axis_pos)*int(scale_percent / 100)
					obsv_XY_filtered = []
					for i in range(obsv_XY.shape[0]):
						if (obsv_XY[i, 0] > 0) & (obsv_XY[i, 0] < dim[0]) & (obsv_XY[i, 1] > 0) & (obsv_XY[i, 1] < dim[1]):
							obsv_XY_filtered.append(obsv_XY[i])
					sup.line_cv(resized_crop_img, np.asarray(obsv_XY_filtered), colors[mix_id], 1)  # bgr convention

					if args.output_pred_state_dim > args.output_dim:
						sigma_x = np.square(sigmax[0]) * (args.dt ** 2)/ args.submap_resolution
						sigma_y = np.square(sigmay[0]) * (args.dt ** 2)/ args.submap_resolution

						obsv_XY = sup.to_image_frame(Hinv, traj_pred)*int(scale_percent / 100)
						axis = (
							int(np.sqrt(sigma_x) / args.submap_resolution), int(np.sqrt(sigma_y) / args.submap_resolution))
						if (obsv_XY[0, 0] > 0) & (obsv_XY[0, 0] < dim[0]) & (obsv_XY[0, 1] > 0) & (obsv_XY[0, 1] < dim[1]):
							center = (obsv_XY[0, 1], obsv_XY[0, 0])
							cv2.ellipse(resized_crop_img, center, axis, 0, 0, 360, colors[mix_id], -1)

						for pred_step in range(1,args.prediction_horizon):
							sigma_x += np.square(sigmax[pred_step]) * (args.dt ** 2)/ args.submap_resolution
							sigma_y += np.square(sigmay[pred_step]) * (args.dt ** 2)/ args.submap_resolution
							axis = (
								int(np.sqrt(sigma_x) / args.submap_resolution), int(np.sqrt(sigma_y) / args.submap_resolution))
							if (obsv_XY[pred_step, 0] > 0) & (obsv_XY[pred_step, 0] < dim[0]) & (obsv_XY[pred_step, 1] > 0) & (obsv_XY[pred_step, 1] < dim[1]):
								center = (obsv_XY[pred_step, 1], obsv_XY[pred_step, 0])
								cv2.ellipse(resized_crop_img, center, axis, 0, 0, 360, colors[mix_id], -1)

				if display:
					cv2.imshow("image", resized_crop_img)
					cv2.waitKey(10)
				out.write(resized_crop_img)

		# When everything done, release the video capture and video write objects
		out.release()

	print("Done recording...")

def plot_dataset(trajs, args):
	"""
		inputs:
			enc_seq: Encoder Inouts Sequences
			dec_seq: Decoder input sequences
			traj: Trajectory Groud Truth of Robot and Pedestrians
			predictions: Pedestrians Predicted Trajectory
			global_grid: Static Environment
	"""
	homography_file = os.path.join(args.data_path + args.scenario, 'H.txt')
	print("Looking for homography at: " + homography_file)
	if os.path.exists(homography_file):
		Hinv = np.linalg.inv(np.loadtxt(homography_file))
	else:
		print('[INF] No homography file')

	img_file = args.data_path + args.scenario +'/map.png'

	if os.path.exists(img_file):
		print('[INF] Using image file ' + img_file)
		img = cv2.imread(img_file)
		map_resolution = 1/Hinv[0,1]
		height, width, channels = img.shape

		# Adding legend

		font = cv2.FONT_HERSHEY_SIMPLEX

		for exp_idx in range(len(trajs)):
			overlay_img = img.copy()
			save_img_to_file = args.data_path + "/" + args.scenario + "/" + str(exp_idx) + '.png'

			cv2.line(overlay_img, (width - 150, 30), (width - 130, 30), (0, 0, 0), 4)
			cv2.putText(overlay_img, "Agent", (width - 120, 30), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
			cv2.line(overlay_img, (width - 150, 50), (width - 130, 50), (255, 0, 0), 4)
			cv2.putText(overlay_img, "Robot", (width - 120, 50), font, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

			sigma_x = 1.0
			sigma_y = 1.0
			axis = (
				int(np.sqrt(sigma_x) / map_resolution), int(np.sqrt(sigma_y) / map_resolution))

			for t in range(len(trajs[exp_idx])):
				dict = trajs[exp_idx][t]
				robot_pos = np.expand_dims(dict["robot_state"][:2],axis=0).astype(float)
				ped_pos = np.expand_dims(dict["pedestrian_state"]["position"], axis=0).astype(float)
				obsv_XY = sup.to_image_frame(Hinv, robot_pos)
				center = (obsv_XY[0, 0], obsv_XY[0,1])
				cv2.ellipse(overlay_img, center, axis, 0, 0, 360, (0, 0, 0), -1)

				obsv_XY = sup.to_image_frame(Hinv, ped_pos)
				center = (obsv_XY[0, 0], obsv_XY[0, 1])
				cv2.ellipse(overlay_img, center, axis, 0, 0, 360, (255, 0, 0), -1)

			if True:
				cv2.imshow("image",overlay_img)
				cv2.waitKey(100)
			cv2.imwrite(save_img_to_file,overlay_img)

def matplot_dataset(trajs, args):
	"""
		inputs:
			enc_seq: Encoder Inouts Sequences
			dec_seq: Decoder input sequences
			traj: Trajectory Groud Truth of Robot and Pedestrians
			predictions: Pedestrians Predicted Trajectory
			global_grid: Static Environment
	"""
	fig = pl.figure('Training Performance ' + args.model_name)
	ax = pl.subplot()
	for exp_idx in range(len(trajs)):
		ax.clear()
		save_img_to_file = args.data_path + "/" + args.scenario + "/" + str(exp_idx) + '.png'

		sigma_x = 1.0
		sigma_y = 1.0
		ax.set_aspect('equal')
		ax.set_xlabel('x [m]')
		ax.set_ylabel('y [m]')
		robot_traj = np.zeros((len(trajs[exp_idx]),2))
		ped_traj = np.zeros((len(trajs[exp_idx]), 2))
		for t in range(len(trajs[exp_idx])):
			dict = trajs[exp_idx][t]
			robot_pos = dict["robot_state"][:2]
			ped_pos = dict["pedestrian_state"]["position"]
			robot_traj[t, :] = robot_pos
			ped_traj[t,:] = ped_pos

			e1 = Ellipse(xy=(robot_pos[0], robot_pos[1]), width=sigma_x / 2, height=sigma_y / 2,
				             angle=0 / np.pi * 180)
			e1.set_alpha(0.5)

			ax.add_patch(e1)

			e1 = Ellipse(xy=(ped_pos[0], ped_pos[1]), width=sigma_x / 2, height=sigma_y / 2,
				             angle=0 / np.pi * 180,facecolor='orange')
			e1.set_alpha(0.5)

			ax.add_patch(e1)
			pl.show(block=False)
		min_x = min(min(robot_traj[:,0]),min(ped_traj[:,0]))-1.0
		max_x = max(max(robot_traj[:, 0]), max(ped_traj[:, 0])) + 1.0
		min_y = min(min(robot_traj[:, 1]), min(ped_traj[:, 1])) - 1.0
		max_y = max(max(robot_traj[:, 1]), max(ped_traj[:, 1])) + 1.0
		ax.set_xlim([min_x, max_x])
		ax.set_ylim([min_y, max_y])
		fig.savefig(save_img_to_file)