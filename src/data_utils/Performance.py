import os
import numpy as np
import math
import pylab as pl
import sys
if sys.version_info[0] < 3:
	import Support as sup
else:
	import src.data_utils.Support as sup


def compute_trajectory_prediction_mse(args,ground_truth, predictions):
	"""
		inputs:
			args: model parameters
			ground_truth: list of groudn truth velocities in absolute frame
			predictions: list of predicted velocities 		"""
	avg_mse = 0
	cnt = 0
	mse_list = []
	for pred, gt in zip(predictions, ground_truth):
		avg_mse = 0
		cnt = 0
		# compute average per trajectory
		for t in range(len(pred)):
			# real trajectory global frame
			real_vel_global_frame = gt.vel_vec[t+args.prev_horizon+1:t+args.prediction_horizon+args.prev_horizon+1,:2]
			real_traj_global_frame = sup.path_from_vel(initial_pos=np.array([0, 0]), pred_vel=real_vel_global_frame,
			                                           dt=args.dt)
			vel_pred = np.zeros((args.prediction_horizon, args.output_dim))
			error = 0
			pred_t = pred[t][0]
			min_error = np.zeros((pred_t.shape[0]))
			for sample_id in range(pred_t.shape[0]):
				error = np.zeros((args.n_mixtures))
				for mix_idx in range(args.n_mixtures):
						# plot predicted trajectory global frame
						for i in range(args.prediction_horizon):
							idx = i * args.output_pred_state_dim * args.n_mixtures + mix_idx
							idy = i * args.output_pred_state_dim * args.n_mixtures + mix_idx + args.n_mixtures
							if args.normalize_data:
								mu_x = pred_t[sample_id,idx] / args.sx_vel + args.min_vel_x
								mu_y = pred_t[sample_id,idy] / args.sy_vel + args.min_vel_y
							else:
								mu_x = pred_t[sample_id,idx]
								mu_y = pred_t[sample_id,idy]
							vel_pred[i,:] = [mu_x, mu_y]

						traj_pred = sup.path_from_vel(initial_pos=np.array([0,0]),pred_vel=vel_pred, dt=args.dt)
						for pred_step in range(args.prediction_horizon):
							error[mix_idx] += np.linalg.norm(real_traj_global_frame[pred_step, :] - traj_pred[pred_step, :])/args.prediction_horizon
				min_error[sample_id] = min(error)
			avg_mse = (avg_mse * cnt + min(min_error)) / (cnt + 1)
			cnt += 1
		mse_list.append(avg_mse)
	return avg_mse, mse_list

def compute_trajectory_fde(args,ground_truth, predictions):
	"""
		inputs:
			args: model parameters
			ground_truth: list of groudn truth velocities in absolute frame
			predictions: list of predicted velocities 		"""
	avg_fde = 0
	cnt = 0
	avg_fde_list = []
	real_traj_global_frame = np.zeros((args.prediction_horizon, args.output_dim))
	for pred, gt in zip(predictions, ground_truth):
		avg_fde = 0
		cnt = 0
		# compute average per trajectory
		for t in range(len(pred)):
			# real trajectory global frame
			real_vel_global_frame = gt.vel_vec[t+args.prev_horizon+1:t+args.prediction_horizon+args.prev_horizon+1,:2]
			real_traj_global_frame = sup.path_from_vel(initial_pos=np.array([0, 0]), pred_vel=real_vel_global_frame,
			                                           dt=args.dt)
			vel_pred = np.zeros((args.prediction_horizon, args.output_dim))
			error = 0
			pred_t = pred[t][0]
			if args.n_mixtures<=1:
				# plot predicted trajectory global frame
				for sample_id in range(1):
					for i in range(args.prediction_horizon):
						idx = i * args.output_pred_state_dim
						idy = i * args.output_pred_state_dim + 1
						mu_x = pred_t[sample_id, idx]
						mu_y = pred_t[sample_id, idy]
						vel_pred[i, :] = [mu_x, mu_y]

				pred_vel_global_frame = vel_pred
				traj_pred = sup.path_from_vel(initial_pos=np.array([0, 0]),pred_vel=np.squeeze(pred_vel_global_frame), dt=args.dt)
				error = np.linalg.norm(real_traj_global_frame[-1, :] - traj_pred[-1, :])
				avg_fde = (avg_fde*cnt+error)/(cnt+1)
				cnt += 1
			else:
				min_error = np.zeros((pred_t.shape[0]))
				for sample_id in range(pred_t.shape[0]):
					error = np.zeros((args.n_mixtures))
					for mix_idx in range(args.n_mixtures):
						# plot predicted trajectory global frame
						for i in range(args.prediction_horizon):
							idx = i * args.output_pred_state_dim * args.n_mixtures + mix_idx
							idy = i * args.output_pred_state_dim * args.n_mixtures + mix_idx + args.n_mixtures
							if args.normalize_data:
								mu_x = pred_t[sample_id,idx] / args.sx_vel + args.min_vel_x
								mu_y = pred_t[sample_id,idy] / args.sy_vel + args.min_vel_y
							else:
								mu_x = pred_t[sample_id,idx]
								mu_y = pred_t[sample_id,idy]
							vel_pred[i,:] = [mu_x, mu_y]

						traj_pred = sup.path_from_vel(initial_pos=np.array([0,0]),pred_vel=vel_pred, dt=args.dt)
						error[mix_idx] = np.linalg.norm(real_traj_global_frame[-1, :] - traj_pred[-1, :])
					min_error[sample_id] = min(error)
				avg_fde = (min(min_error)+ avg_fde*cnt)/(cnt+1)
				cnt +=1
			avg_fde_list.append(avg_fde)
	return avg_fde, avg_fde_list

def compute_2_wasserstein(args, predictions):
	"""
		inputs:
			args: model parameters
			ground_truth: list of groudn truth velocities in absolute frame
			predictions: list of predicted velocities 		"""
	avg_wasser = 0
	cnt = 0
	wasser_list = []
	for prediction in predictions: # number of trajectories
		avg_wasser = 0
		cnt = 0
		# compute average per trajectory
		for pred_step in prediction: # trajectory length
			error = 0
			for i in range(args.prediction_horizon):  # prediction length
				if args.n_mixtures == 2:
					idx = i * args.output_pred_state_dim * args.n_mixtures
					idy = i * args.output_pred_state_dim * args.n_mixtures + args.n_mixtures
					mu_x = pred_step[0][ 0, idx]
					mu_y = pred_step[0][ 0, idy]
					mu_x1 = pred_step[0][ 0, idx+1]
					mu_y1 = pred_step[0][ 0, idy+1]
					sigma_x = pred_step[0][0, idy + args.n_mixtures]
					sigma_y = pred_step[0][0, idy + 2*args.n_mixtures]
					sigma_x1 = pred_step[0][0, idy + args.n_mixtures+1]
					sigma_y1 = pred_step[0][0, idy + 2*args.n_mixtures+1]
					wasser1 = np.linalg.norm(np.array([mu_x,mu_y])-np.array([mu_x1,mu_y1])) + sigma_x+sigma_x1+sigma_y+sigma_y1-2*np.sqrt(sigma_x*sigma_x1)-2*np.sqrt(sigma_y*sigma_y1)
					error += wasser1/args.n_mixtures
					avg_wasser = (avg_wasser * cnt + error) / (cnt + 1)
					cnt += 1
				if args.n_mixtures == 3:
					wasser1 = 0
					for mix_idx in range(args.n_mixtures):
						idx = i * args.output_pred_state_dim * args.n_mixtures + mix_idx
						idy = i * args.output_pred_state_dim * args.n_mixtures + args.n_mixtures + mix_idx
						mu_x = pred_step[0][ 0, idx]
						mu_y = pred_step[0][ 0, idy]
						sigma_x = pred_step[0][ 0, idy + args.n_mixtures]
						sigma_y = pred_step[0][ 0, idy + 2*args.n_mixtures]
						if mix_idx+1 >= args.n_mixtures:
							mu_x1 = pred_step[0][ 0, idx - mix_idx]
							mu_y1 = pred_step[0][ 0, idy - mix_idx]
							sigma_x1 = pred_step[0][ 0, idy + args.n_mixtures - mix_idx]
							sigma_y1 = pred_step[0][ 0, idy + 2 * args.n_mixtures - mix_idx]
						else:
							mu_x1 = pred_step[0][ 0, idx+1]
							mu_y1 = pred_step[0][ 0, idy+1]
							sigma_x1 = pred_step[0][ 0, idy + args.n_mixtures+1]
							sigma_y1 = pred_step[0][ 0, idy + 2*args.n_mixtures+1]
						wasser1 += np.linalg.norm(np.array([mu_x,mu_y])-np.array([mu_x1,mu_y1]))+sigma_x+sigma_x1+sigma_y+sigma_y1-2*np.sqrt(sigma_x*sigma_x1)-2*np.sqrt(sigma_y*sigma_y1)
					error += wasser1/args.n_mixtures
					avg_wasser = (avg_wasser * cnt + error) / (cnt + 1)
					cnt += 1
		wasser_list.append(avg_wasser)
	return avg_wasser, wasser_list

def compute_nll(args, ground_truth,predictions):
	"""
		inputs:
			args: model parameters
			ground_truth: list of groudn truth velocities in absolute frame
			predictions: list of predicted velocities 		"""
	cnt = 0
	avg_list = []
	for pred, gt in zip(predictions, ground_truth): # number of trajectories
		avg_nll = 0
		cnt = 0
		# compute average per trajectory
		for t in range(pred.shape[0]): # trajectory length
			nll = 0
			for i in range(args.prediction_seq_length):  # prediction length
				term1 = 0
				for mix_id in range(args.n_mixtures):
					id = i * args.output_state_dim
					idx = i * args.output_pred_state_dim * args.n_mixtures
					idy = i * args.output_pred_state_dim * args.n_mixtures + args.n_mixtures
					mu_x = pred[t, idx+ mix_id]
					mu_y = pred[t, idy+ mix_id]
					sigma_x = max(pred[t, idy + args.n_mixtures+ mix_id],0)
					sigma_y = max(pred[t, idy + 2 * args.n_mixtures+ mix_id],0)
					pi = pred[t, idy + 3 * args.n_mixtures+ mix_id]
					normx = gt[t, id] - mu_x
					normy = gt[t, id + 1] - mu_y
					# Calculate sx*sy
					sxsy = sigma_x*sigma_y
					# Calculate the exponential factor
					z = np.square(normx/sigma_x) + np.square(normy/sigma_y)
					# Numerator
					result = np.exp(-z/2)
					# Normalization constant
					denom = 2 * np.pi * sxsy
					# Final PDF calculation
					result = result / denom
					term1 += pi*result
				nll += -np.log(np.maximum(term1, 1e-5))
			nll = nll / args.prediction_seq_length
			avg_nll = (avg_nll * cnt + nll) / (cnt + 1)
			cnt += 1
		avg_list.append(avg_nll)
	return avg_nll, avg_list

def compute_nll2(args, ground_truth,predictions):
	"""
		inputs:
			args: model parameters
			ground_truth: list of groudn truth velocities in absolute frame
			predictions: list of predicted velocities 		"""
	cnt = 0
	avg_list = []
	for pred, gt in zip(predictions, ground_truth): # number of trajectories
		avg_nll = 0
		cnt = 0
		# compute average per trajectory
		for t in range(pred.shape[0]): # trajectory length
			nll = 0
			pi = pred[t, args.prediction_seq_length*args.n_mixtures*args.output_pred_state_dim:]
			for i in range(args.prediction_seq_length):  # prediction length
				term1 = 0
				for mix_id in range(args.n_mixtures):
					id = i * args.output_state_dim
					idx = i * args.output_pred_state_dim * args.n_mixtures
					idy = i * args.output_pred_state_dim * args.n_mixtures + args.n_mixtures
					mu_x = pred[t, idx+ mix_id]
					mu_y = pred[t, idy+ mix_id]
					sigma_x = pred[t, idy + args.n_mixtures+ mix_id]
					sigma_y = pred[t, idy + 2 * args.n_mixtures+ mix_id]
					w = pred[t, idy + 3 * args.n_mixtures+ mix_id]
					normx = gt[t, id] - mu_x
					normy = gt[t, id + 1] - mu_y
					# Calculate sx*sy
					sxsy = sigma_x*sigma_y
					# Calculate the exponential factor
					z = (np.square(normx/sigma_x) + np.square(normy/sigma_y))/2.0
					# Numerator
					result = math.exp(-z)
					# Normalization constant
					denom = 2 * np.pi * sxsy
					# Final PDF calculation
					result2 = result / denom #pi[mix_id]*
					term1 += result2 / args.n_mixtures
				nll += -np.log(np.maximum(term1, 1e-5))
			nll = nll / args.prediction_seq_length
			avg_nll = (avg_nll * cnt + nll) / (cnt + 1)
			cnt += 1
		avg_list.append(avg_nll)
	return avg_nll, avg_list