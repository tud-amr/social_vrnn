import os
import numpy as np
import cv2
import sys
if sys.version_info[0] < 3:
	import Support as sup
else:
	import src.data_utils.Support as sup
import math

class Plotter:

	@staticmethod
	def generate_zoom_plot(cmdargs, input_list, grid_list, y_pred_list_global, y_ground_truth_list, other_agents_list, traj_list, test_args, social_trajectories=None):

		scale_factor = 8
		context_scalar = 1.5

		# Relevant file paths
		scenario = cmdargs.scenario.split('/')[-1]
		video_file = cmdargs.model_path + '/../videos/'+ scenario + str(cmdargs.exp_num) + "_zoom.avi"
		map_file = cmdargs.data_path + cmdargs.scenario + '/map.png'
		homography_file = os.path.join(cmdargs.data_path + cmdargs.scenario, 'H.txt')
		if not os.path.exists(map_file): print("No map file found"); exit()
		if os.path.exists(homography_file): Hinv = np.linalg.inv(np.loadtxt(homography_file))
		else: print('[INF] No homography file'); exit()
		resolution = cmdargs.submap_resolution

		submap_size = int(cmdargs.submap_width * Hinv[0][0] * scale_factor)
		# print(submap_size)
		# exit()

		# np.set_printoptions(suppress=True)
		# print(Hinv)
		# exit()

		# print("Submap resolution: " + str(resolution))

		im = np.uint8(cv2.imread(map_file) * -1 + 255)
		im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE) # for roboat output
		frame_height, frame_width, layers = im.shape
		resized_img = cv2.resize(im, (frame_width * scale_factor, frame_height * scale_factor), interpolation=cv2.INTER_AREA)

		cap = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc('M', 'P', '4', '2'), 5,
										(submap_size, submap_size))

		vel_real = np.zeros((cmdargs.prediction_horizon,  cmdargs.output_dim))
		vel_pred = np.zeros((cmdargs.prediction_horizon,  cmdargs.output_dim))
		sigmax = np.zeros((cmdargs.prediction_horizon, 1))
		sigmay = np.zeros((cmdargs.prediction_horizon, 1))
		pis = np.zeros((cmdargs.prediction_horizon, 3))

		# For each separate trajectory
		for animation_idx in range(len(input_list)):
			traj = traj_list[animation_idx]

			if social_trajectories:
				social_traj = social_trajectories[animation_idx]
			gt_vel = y_ground_truth_list[animation_idx]
			# print(gt_vel)
			# print(gt_vel.shape)
			# exit()
			input = input_list[animation_idx]
			if not (y_pred_list_global is None):
				model_vel_pred = y_pred_list_global[animation_idx]
			else:
				model_vel_pred = None

			# For each different step within trajectory
			# plot real trajectory global frame
			# print(input.shape[0])
			for step in range(input.shape[0]):

				overlay = resized_img.copy()

				# Initial positions
				if cmdargs.normalize_data:
					x0 = input[step, 0] / cmdargs.sx_pos + cmdargs.min_pos_x
					y0 = input[step, 1] / cmdargs.sy_pos + cmdargs.min_pos_y
				else:
					x0 = input[step, 0]
					y0 = input[step, 1]

				# Real Trajectory
				# print(cmdargs.prediction_horizon)
				# print(cmdargs.output_dim)
				# print(f"step {step}:")
				for i in range(cmdargs.prediction_horizon):
					idx = i * cmdargs.output_dim
					idy = i * cmdargs.output_dim + 1
					mu_x = gt_vel[step, idx]
					mu_y = gt_vel[step, idy]
					# print(f"{mu_x}, {mu_y}")
					vel_real[i, :] = [mu_x, mu_y]
					# print(vel_real.dtype)

				real_vel_global_frame = vel_real
				traj_real = sup.path_from_vel(initial_pos=np.array([x0,
																						y0]),
														pred_vel=real_vel_global_frame, dt=cmdargs.dt)
				# print(traj_real)
				# obsv_XY = sup.to_image_frame(Hinv, traj_real)*scale_factor
				# print(obsv_XY)
				agent_XY = (sup.to_image_frame_float(Hinv, traj_real)*scale_factor).astype(int)
				obsv_XY = agent_XY
				# print(agent_XY)
				# print("Agent path:")
				# print(sup.to_image_frame(Hinv, traj_real))
				sup.line_cv(overlay, obsv_XY, (255, 0, 0), 2) # bgr convention

				# Plot social trajectory
				if social_trajectories:
					for sample_id in range(3):
						social_traj_pred = sup.path_from_vel(initial_pos=np.array([input[step, 0],
																							input[step, 1]]),
															pred_vel=social_traj[step][sample_id], dt=1)
						obsv_XY = (sup.to_image_frame_float(Hinv, social_traj_pred)*scale_factor).astype(int)
						sup.line_cv(overlay, obsv_XY, (255, 0, 0), 3)  # bgr convention

				# Plot real predicted traj from positions
				traj_real = traj.pose_vec[step+cmdargs.prev_horizon:step+cmdargs.prev_horizon+cmdargs.prediction_horizon,:2]
				# obsv_XY = sup.to_image_frame(Hinv, traj_real)*scale_factor
				obsv_XY = (sup.to_image_frame_float(Hinv, traj_real) * scale_factor).astype(int)
				if False: sup.line_cv(overlay, obsv_XY, (128, 128, 0), 3)  # bgr convention

				traj_real = traj.pose_vec[step:,:2]
				# obsv_XY2 = sup.to_image_frame(Hinv, traj_real)*scale_factor
				obsv_XY2 = (sup.to_image_frame_float(Hinv, traj_real) * scale_factor).astype(int)
				obsv_XY3 = obsv_XY2[cmdargs.prev_horizon:cmdargs.prev_horizon+cmdargs.prediction_horizon]
				delta = obsv_XY - obsv_XY3

				if np.max(np.abs(delta))>0: print("problem")
				if False: sup.line_cv(overlay, obsv_XY2, (0, 0, 0), 3)  # bgr convention

				# Predicted trajectory
				colors = [(0,0,255),(0,255,0),(0,255,255)]
				if not (y_pred_list_global is None):
					if cmdargs.n_mixtures == 0:
						prediction_sample = model_vel_pred[step][0]
						# plot predicted trajectory global frame
						time = np.zeros([cmdargs.prediction_horizon])
						for i in range(cmdargs.prediction_horizon):
							time[i] = i*cmdargs.dt
							idx = i * cmdargs.output_pred_state_dim
							idy = i * cmdargs.output_pred_state_dim + 1
							if cmdargs.normalize_data:
								mu_x = prediction_sample[0, idx] / cmdargs.sx_vel + cmdargs.min_vel_x
								mu_y = prediction_sample[0, idy] / cmdargs.sy_vel + cmdargs.min_vel_y
							else:
								mu_x = prediction_sample[0, idx]
								mu_y = prediction_sample[0, idy]
							vel_pred[i, :] = [mu_x, mu_y]

						pred_vel_global_frame = vel_pred

						if cmdargs.predict_positions:
							traj_pred = vel_pred
						else:
							traj_pred = sup.path_from_vel(initial_pos=np.array(
								[x0,
									y0]),
								pred_vel=pred_vel_global_frame, dt=cmdargs.dt)
						# sub-sample to make smoother trajectories
						#_, traj_pred_smooth, _ = self.smoothenTrajectory(time,traj_pred,vel_pred)

						# obsv_XY = sup.to_image_frame(Hinv, traj_pred)*scale_factor
						obsv_XY = (sup.to_image_frame_float(Hinv, traj_pred) * scale_factor).astype(int)
						sup.line_cv(overlay, obsv_XY, (0, 0, 255),2)
					else:
						for sample_id in range(test_args.n_samples):
							prediction_sample = model_vel_pred[step][sample_id]
							for mix_idx in range(cmdargs.n_mixtures): #cmdargs.n_mixtures
								# plot predicted trajectory global frame
								time = np.zeros([cmdargs.prediction_horizon])
								for pred_step in range(cmdargs.prediction_horizon):
									time[pred_step] = pred_step * cmdargs.dt
									idx = pred_step * cmdargs.output_pred_state_dim * cmdargs.n_mixtures + mix_idx
									if cmdargs.normalize_data:
										mu_x = prediction_sample[0, idx] / cmdargs.sx_vel + cmdargs.min_vel_x
										mu_y = prediction_sample[0, idx + cmdargs.n_mixtures] / cmdargs.sy_vel + cmdargs.min_vel_y
									else:
										mu_x = prediction_sample[0, idx]
										mu_y = prediction_sample[0, idx + cmdargs.n_mixtures]
									if cmdargs.output_pred_state_dim > 2:
										sigmax[pred_step, :] = prediction_sample[0][idx + 2 * cmdargs.n_mixtures]
										sigmay[pred_step, :] = prediction_sample[0][idx + 3 * cmdargs.n_mixtures]
									if cmdargs.output_pred_state_dim ==5:
										pis[pred_step, mix_idx] = prediction_sample[0][idx + 4 * cmdargs.n_mixtures]
									if math.isnan(mu_x) | math.isnan(mu_y):
										continue
									vel_pred[pred_step, :] = [mu_x, mu_y]

								pred_vel_global_frame = vel_pred
								traj_pred = sup.path_from_vel(initial_pos=np.array(
									[x0,
										y0]),
									pred_vel=pred_vel_global_frame, dt=cmdargs.dt)
								# sub-sample to make smoother trajectories
								#self.smoothenTrajectory(time, traj_pred, vel_pred)
								# obsv_XY = sup.to_image_frame(Hinv, traj_pred)*scale_factor
								obsv_XY = (sup.to_image_frame_float(Hinv, traj_pred) * scale_factor).astype(int)

								sup.line_cv(overlay, obsv_XY, colors[mix_idx], 2)

								try:
									if cmdargs.output_pred_state_dim > 2:
										# prior of 0 on the uncertainty of the pedestrian velocity
										sigma_x = np.square(sigmax[0]) * cmdargs.dt * cmdargs.dt +6.0
										sigma_y = np.square(sigmay[0]) * cmdargs.dt * cmdargs.dt +3.0
										if math.isnan(sigma_x) | math.isnan(sigma_y):
											continue
										axis = (int(np.sqrt(sigma_x) /resolution*scale_factor),int(np.sqrt(sigma_y) /resolution*scale_factor))
										y = obsv_XY[0, 1]*scale_factor
										x = obsv_XY[0, 0]*scale_factor
										center = (y , x)
										cv2.ellipse(overlay, center,       axis,    0, 0, 360, (255,153,51),-1)

										for pred_step in range(1, cmdargs.prediction_horizon):
											sigma_x += np.square(sigmax[pred_step]) * cmdargs.dt * cmdargs.dt
											sigma_y += np.square(sigmay[pred_step]) * cmdargs.dt * cmdargs.dt
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
						# print('is used')
						# obsv_XY = (sup.to_image_frame_float(Hinv, np.expand_dims(other_agents[agent_id,:2], axis=0)) * scale_factor).astype(int)
						center = (int(obsv_XY[0, 1]*scale_factor), int(obsv_XY[0, 0]*scale_factor))
						cv2.ellipse(overlay, center, (5, 5), 0, 0, 360, (153, 153, 51), -1)

				# Adding legend
				font = cv2.FONT_HERSHEY_SIMPLEX
				cv2.line(overlay, (int(frame_height*scale_factor*0.9-50), 30), (int(frame_height*scale_factor*0.9-40), 30), (255, 0, 0), 4)
				cv2.putText(overlay, "Real Trajectory", (int(frame_height*scale_factor*0.9-30), 30), font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
				cv2.line(overlay, (int(frame_height*scale_factor*0.9-50), 50), (int(frame_height*scale_factor*0.9-40), 50), (0, 0, 255), 4)
				cv2.putText(overlay, "Predicted Trajectory", (int(frame_height*scale_factor * 0.9 - 40), 50), font, 0.5, (0, 0, 255), 2,
								cv2.LINE_AA)

				obsvs_XY = agent_XY # obsv_XY*scale_factor
				# print(f"{input[step, 0]}, {input[step, 1]}")
				# print(overlay.shape)
				# print("original:\t" + str(obsv_XY[0]))
				# print("scaled:\t\t" + str(obsvs_XY[0]))
				# print(str(int(obsvs_XY[0][0] - 0.5*submap_size)) + ":" + str(int(obsvs_XY[0][0] + 0.5*submap_size)) + ", " + str(int(obsvs_XY[0][1] - 0.5*submap_size)) + ":" + str(int(obsvs_XY[0][1] + 0.5*submap_size)))
				# print(overlay[int(obsvs_XY[0][0] - 0.5*submap_size):int(obsvs_XY[0][0] + 0.5*submap_size), int(obsvs_XY[0][1] - 0.5*submap_size):int(obsvs_XY[0][1] + 0.5*submap_size)].shape)
				# print(cmdargs.submap_width * cmdargs.submap_resolution)
				center = obsvs_XY[0]

				
				cap.write(overlay[int(obsvs_XY[0][0] - 0.5*submap_size):int(obsvs_XY[0][0] + 0.5*submap_size), int(obsvs_XY[0][1] - 0.5*submap_size):int(obsvs_XY[0][1] + 0.5*submap_size)])

				#cv2.imwrite(cmdargs.model_path + '/results/' + cmdargs.scenario+"/figs/result_"+str(animation_idx)+"_"+str(step)+".jpg", overlay);

		# When everything done, release the video capture and video write objects
		cap.release()

		# Closes all the frames
		cv2.destroyAllWindows()
		