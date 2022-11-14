import sys
import os
sys.path.append('../')
import numpy as np
import argparse
import pylab as pl
import json
import importlib
if sys.version_info[0] < 3:
	print("Using Python " + str(sys.version_info[0]))
	sys.path.append('../src/data_utils')
	sys.path.append('../src/models')
	import DataHandlerLSTM as dhlstm
	from plot_utils import *
	import Support as sup
	from utils import *
else:
	print("Using Python " + str(sys.version_info[0]))
	from src.data_utils import DataHandlerLSTM as dhlstm
	from src.data_utils.plot_utils import *
	from src.data_utils import Support as sup
	from src.data_utils.utils import *
from datetime import datetime
import pickle as pkl
import time
from copy import deepcopy
from multiprocessing.pool import ThreadPool
import colorama
from colorama import Fore, Style

#os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

pretrained_convnet_path = "../trained_models/autoencoder_with_ped"

data_path = '../data/2_agents_swap/trajs/'
scenario = 'GA3C-CADRL-10-py27'
data_path = '../data/cyberzoo_experiments/'
scenario = 'all_trajectories'
exp_num = 6
data_path = '../data/'
scenario = '20_ped_with_obstacles/short_few_obstacles'

# Hyperparameters
n_epochs = 2

batch_size = 128
regularization_weight = 0.0001

# Time parameters
truncated_backprop_length = 10
prediction_horizon = 10
prev_horizon = 0

rnn_state_size = 32
rnn_state_size_lstm_grid = 256
rnn_state_size_lstm_ped = 128
rnn_state_ped_size = 16
rnn_state_size_lstm_concat = 512
prior_size = 512
latent_space_size = 256
x_dim = 512
fc_hidden_unit_size = 256
learning_rate_init = 0.001
beta_rate_init = 0.01
keep_prob = 1.0
dropout = False
reg = 1e-4
n_mixtures = 1  # USE ZERO FOR MSE MODEL
grads_clip = 1.0
n_other_agents = 18
tensorboard_logging = True

# Model parameters
input_dim = 4  # [vx, vy]
input_state_dim = 2  # [vx, vy]
output_dim = 2  # data state dimension
output_pred_state_dim = 4  # ux uy simgax sigmay
pedestrian_vector_dim = 36
pedestrian_vector_state_dim = 2
cmd_vector_dim = 2
pedestrian_radius = 0.3
max_range_ped_grid = 5

print_freq = 200
save_freq = 500
total_training_steps = 20000
dt = 0.1

warmstart_model = False
pretrained_convnet = False
pretained_encoder = False
multipath = False
real_world_data = False
end_to_end = True
agents_on_grid = False
rotated_grid = False
centered_grid = True
noise = False
normalize_data = False
real_world_data = False
regulate_log_loss = False
# Map parameters
submap_resolution = 0.1
submap_width = 6
submap_height = 6
diversity_update = False
predict_positions = False
warm_start_convnet = True
debug_plotting = False

# Dataset division
train_set = 0.8

def parse_args():
	parser = argparse.ArgumentParser(description='LSTM model training')

	parser.add_argument('--model_name',
	                    help='Path to directory that comprises the model (default="model_name").',
	                    type=str, default= "RNN")
	parser.add_argument('--model_path',
	                    help='Path to directory to save the model (default=""../trained_models/"+model_name").',
	                    type=str, default='../trained_models/')
	parser.add_argument('--pretrained_convnet_path',
	                    help='Path to directory that comprises the pre-trained convnet model (default=" ").',
	                    type=str, default=pretrained_convnet_path)
	parser.add_argument('--log_dir',
	                    help='Path to the log directory of the model (default=""../trained_models/"+model_name").',
	                    type=str, default="\log")
	parser.add_argument('--scenario', help='Scenario of the dataset (default="").',
	                    type=str, default=scenario)
	parser.add_argument('--real_world_data', help='Real world dataset (default=True).', type=sup.str2bool,
	                    default=real_world_data)
	parser.add_argument('--data_path', help='Path to directory that saves pickle data (default=" ").', type=str,
	                    default=data_path)
	parser.add_argument('--dataset', help='Dataset pkl file', type=str,
	                    default= scenario + '.pkl')
	parser.add_argument('--data_handler', help='Datahandler class needed to load the data', type=str,
	                    default='LSTM')
	parser.add_argument('--warmstart_model', help='Restore from pretained model (default=False).', type=bool,
	                    default=warmstart_model)
	parser.add_argument('--warm_start_convnet', help='Restore from pretained convnet model (default=False).', type=bool,
	                    default=warm_start_convnet)
	parser.add_argument('--dt', help='Data samplig time (default=0.3).', type=float,
	                    default=dt)
	parser.add_argument('--n_epochs', help='Number of epochs (default=10000).', type=int, default=n_epochs)
	parser.add_argument('--total_training_steps', help='Number of training steps (default=20000).', type=int, default=total_training_steps)
	parser.add_argument('--batch_size', help='Batch size for training (default=32).', type=int, default=batch_size)
	parser.add_argument('--regularization_weight', help='Weight scaling of regularizer (default=0.01).', type=float,
	                    default=regularization_weight)
	parser.add_argument('--keep_prob', help='Dropout (default=0.8).', type=float,
	                    default=keep_prob)
	parser.add_argument('--learning_rate_init', help='Initial learning rate (default=0.005).', type=float,
	                    default=learning_rate_init)
	parser.add_argument('--beta_rate_init', help='Initial beta rate (default=0.005).', type=float,
	                    default=beta_rate_init)
	parser.add_argument('--dropout', help='Enable Dropout', type=sup.str2bool,
	                    default=dropout)
	parser.add_argument('--grads_clip', help='Gridient clipping (default=10.0).', type=float,
	                    default=grads_clip)
	parser.add_argument('--truncated_backprop_length', help='Backpropagation length during training (default=5).',
	                    type=int, default=truncated_backprop_length)
	parser.add_argument('--prediction_horizon', help='Length of predicted sequences (default=10).', type=int,
	                    default=prediction_horizon)
	parser.add_argument('--prev_horizon', help='Previous seq length.', type=int,
	                    default=prev_horizon)
	parser.add_argument('--rnn_state_size', help='Number of RNN / LSTM units (default=16).', type=int,
	                    default=rnn_state_size)
	parser.add_argument('--rnn_state_size_lstm_ped',
	                    help='Number of RNN / LSTM units of the grid lstm layer (default=32).',
	                    type=int, default=rnn_state_size_lstm_ped)
	parser.add_argument('--rnn_state_ped_size',
	                    help='Number of RNN / LSTM units of the grid lstm layer (default=32).',
	                    type=int, default=rnn_state_ped_size)
	parser.add_argument('--rnn_state_size_lstm_grid',
	                    help='Number of RNN / LSTM units of the grid lstm layer (default=32).',
	                    type=int, default=rnn_state_size_lstm_grid)
	parser.add_argument('--rnn_state_size_lstm_concat',
	                    help='Number of RNN / LSTM units of the concatenation lstm layer (default=32).',
	                    type=int, default=rnn_state_size_lstm_concat)
	parser.add_argument('--prior_size', help='prior_size',
	                    type=int, default=prior_size)
	parser.add_argument('--latent_space_size', help='latent_space_size',
	                    type=int, default=latent_space_size)
	parser.add_argument('--x_dim', help='x_dim',
	                    type=int, default=x_dim)
	parser.add_argument('--fc_hidden_unit_size',
	                    help='Number of fully connected layer units after LSTM layer (default=64).',
	                    type=int, default=fc_hidden_unit_size)
	parser.add_argument('--input_state_dim', help='Input state dimension (default=).', type=int,
	                    default=input_state_dim)
	parser.add_argument('--input_dim', help='Input state dimension (default=).', type=float,
	                    default=input_dim)
	parser.add_argument('--output_dim', help='Output state dimension (default=).', type=float,
	                    default=output_dim)
	parser.add_argument('--goal_size', help='Goal dimension (default=).', type=int,
	                    default=2)
	parser.add_argument('--output_pred_state_dim', help='Output prediction state dimension (default=).', type=int,
	                    default=output_pred_state_dim)
	parser.add_argument('--cmd_vector_dim', help='Command control dimension.', type=int,
	                    default=cmd_vector_dim)
	parser.add_argument('--n_mixtures', help='Number of modes (default=).', type=int,
	                    default=n_mixtures)
	parser.add_argument('--pedestrian_vector_dim', help='Number of angular grid sectors (default=72).', type=int,
	                    default=pedestrian_vector_dim)
	parser.add_argument('--pedestrian_vector_state_dim', help='Number of angular grid sectors (default=2).', type=int,
	                    default=pedestrian_vector_state_dim)
	parser.add_argument('--max_range_ped_grid', help='Maximum pedestrian distance (default=2).', type=float,
	                    default=max_range_ped_grid)
	parser.add_argument('--pedestrian_radius', help='Pedestrian radius (default=0.3).', type=float,
	                    default=pedestrian_radius)
	parser.add_argument('--n_other_agents', help='Number of other agents incorporated in the network.', type=int,
	                    default=n_other_agents)
	parser.add_argument('--debug_plotting', help='Plotting for debugging (default=False).', type=sup.str2bool, default=debug_plotting)
	parser.add_argument('--print_freq', help='Print frequency of training info (default=100).', type=int,
	                    default=print_freq)
	parser.add_argument('--save_freq', help='Save frequency of the temporary model during training. (default=20k).',
	                    type=int, default=save_freq)
	parser.add_argument('--exp_num', help='Experiment number', type=int, default=exp_num)
	parser.add_argument('--noise', help='Likelihood? (default=True).', type=sup.str2bool,
	                    default=False)
	parser.add_argument('--agents_on_grid', help='Likelihood? (default=True).', type=sup.str2bool,
	                    default=agents_on_grid)
	parser.add_argument('--normalize_data', help='Normalize? (default=False).', type=sup.str2bool,
	                    default=normalize_data)
	parser.add_argument('--rotated_grid', help='Rotate grid? (default=False).', type=sup.str2bool,
	                    default=rotated_grid)
	parser.add_argument('--centered_grid', help='Center grid? (default=False).', type=sup.str2bool,
	                    default=centered_grid)
	parser.add_argument('--sigma_bias', help='Percentage of the dataset used for trainning', type=float,
	                    default=0)
	parser.add_argument('--submap_width', help='width of occupancy grid', type=int, default=submap_width)
	parser.add_argument('--submap_height', help='height of occupancy grid', type=int, default=submap_height)
	parser.add_argument('--submap_resolution', help='Map resolution.', type=float,
	                    default=submap_resolution)
	parser.add_argument('--min_buffer_size', help='Minimum buffer size (default=1000).', type=int, default=1000)
	parser.add_argument('--max_buffer_size', help='Maximum buffer size (default=100k).', type=int, default=100000)
	parser.add_argument('--max_trajectories', help='maximum number of trajectories to be recorded', type=int, default=30)
	parser.add_argument('--end_to_end', help='End to end trainning.', type=sup.str2bool,
	                    default=False)
	parser.add_argument('--predict_positions', help='predict_positions.', type=sup.str2bool,
	                    default=predict_positions)
	parser.add_argument('--gpu', help='Enable GPU training.', type=sup.str2bool,
	                    default=False)
	parser.add_argument('--sequence_info', help='Use relative info for other agents.', type=sup.str2bool,
	                    default=False)
	parser.add_argument('--others_info', help='Use relative info for other agents.', type=str,
	                    default="none")
	parser.add_argument('--regulate_log_loss', help='Enable GPU training.', type=sup.str2bool,
	                    default=regulate_log_loss)
	parser.add_argument('--diversity_update', help='diversity_update', type=sup.str2bool,
	                    default=diversity_update)
	parser.add_argument('--topics_config', help='yaml file containg subscription topics (default=" ").', type=str,
	                    default='../config/topics.yaml')
	parser.add_argument('--min_pos_x', help='min_pos_x', type=float, default=-1)
	parser.add_argument('--min_pos_y', help='min_pos_y', type=float, default=-1)
	parser.add_argument('--max_pos_x', help='max_pos_x', type=float, default=1)
	parser.add_argument('--max_pos_y', help='max_pos_y', type=float, default=1)
	parser.add_argument('--min_vel_x', help='min_vel_x', type=float, default=-1)
	parser.add_argument('--min_vel_y', help='min_vel_y', type=float, default=-1)
	parser.add_argument('--max_vel_x', help='max_vel_x', type=float, default=1)
	parser.add_argument('--max_vel_y', help='max_vel_y', type=float, default=1)
	parser.add_argument('--sx_vel', help='sx_vel', type=float, default=1)
	parser.add_argument('--sy_vel', help='sy_vel', type=float, default=1)
	parser.add_argument('--sx_pos', help='sx_pos', type=float, default=1)
	parser.add_argument('--sy_pos', help='sy_pos', type=float, default=1)
	parser.add_argument('--train_set', help='Percentage of the dataset used for training', type=float, default=train_set)
	args = parser.parse_args()

	return args


args = parse_args()

# Enable / Disable GPU
if args.gpu:
	import tensorflow as tf
else:
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
	import tensorflow as tf

# Create Log and Model Directory to save training model
args.model_path = '../trained_models/' + args.model_name + "/" + str(args.exp_num)
args.log_dir = args.model_path + '/log'
if not os.path.exists(args.log_dir):
	os.makedirs(args.log_dir)
args.dataset = '/' + args.scenario + '.pkl'
model_parameters = {"args": args}

# Check whether model folder exists, otherwise make directory
if not os.path.exists(args.model_path):
	os.makedirs(args.model_path)

# Save Model Parameters
param_file = open(args.model_path + '/model_parameters.pkl', 'wb')
pkl.dump(model_parameters, param_file, protocol=2)  # encoding='latin1'
param_file.close()
with open(args.model_path + '/model_parameters.json', 'w') as f:
	json.dump(args.__dict__,f)

# Create Datahandler class
data_prep = dhlstm.DataHandlerLSTM(args)
# Only used to create a map from png
# Make sure this parameters are correct otherwise it will fail training and ploting the results
map_args = {"file_name": 'map.png',
	            "resolution": 0.1,
	            "map_size": np.array([30., 6.]), }
# Load dataset
data_prep.processData(**map_args)

# Import Deep Learning model
module = importlib.import_module("src.models."+args.model_name)
globals().update(module.__dict__)

# Create Model Graph
model = NetworkModel(args)

config = tf.ConfigProto()

# Start Training Session
with tf.Session(config=config) as sess:

	# Load a pre-trained model
	if args.warmstart_model:
		model.warmstart_model(args, sess)
	else:
		# Initialize all TF variables
		sess.run(tf.global_variables_initializer())

	# Load Convnet Model
	try:
		if args.warm_start_convnet:
			model.warmstart_convnet(args, sess)
	except:
		print("Failed to initialized Convnet or Convnet does not exist")

	# if the training was interrupted load last training step index
	try:
		initial_step = int(open(args.model_path + "/tf_log", 'r').read().split('\n')[-2]) + 1
	except:
		initial_step = 1

	epoch = 0
	training_loss = []
	diversity_loss = []
	training_loss.append(0)

	# Set up multithreadign for data handler
	pool = ThreadPool(1)
	res = None
	_model_prediction = []
	start_time = time.time()
	best_loss = float('inf')
	avg_training_loss = np.ones(100)

	for step in range(initial_step,args.total_training_steps):
		start_time_loop = time.time()

		# Get Next Batch of Data
		if res == None:
			batch_x, batch_vel, batch_pos,batch_goal,batch_grid, batch_ped_grid, batch_y,batch_pos_target, other_agents_pos, new_epoch = data_prep.getBatch()
		else:
			batch = res.get(timeout=5)

		# Create dictionary to feed into the model
		dict = {"batch_x": batch_x,
		        "batch_vel": batch_vel,
		        "batch_pos": batch_pos,
		        "batch_goal": batch_goal,
			      "batch_grid": batch_grid,
			      "batch_ped_grid": batch_ped_grid,
			      "step": step,
			      "batch_y": batch_y,
			      "batch_pos_target": batch_pos_target,
		        "batch_div": batch_y,
		        "other_agents_pos": other_agents_pos
			      }

		feed_dict_train = model.feed_dic(**dict)

		#res = pool.apply_async(data_prep.getBatch)

		epoch += new_epoch

		# Initialize the new sequences with a hidden state of zeros, the continuing sequences get assigned the previous hidden state
		model.reset_cells(data_prep.sequence_reset)

		start_time_training = time.time()

		model_output = model.train_step(sess, feed_dict_train,step)

		avg_training_time = time.time() - start_time_training
		avg_loop_time = time.time() - start_time_loop

		training_loss.append(model_output["batch_loss"])

		if step == 1 :
			avg_training_loss *= model_output["batch_loss"]
		else:
			avg_training_loss = np.roll(avg_training_loss,shift=1)
			avg_training_loss[0] = model_output["batch_loss"]

		# Print training info
		if step % print_freq == 0:

			# Get batch to compute validation loss
			validation_dict = data_prep.getTestBatch()

			model.reset_test_cells(data_prep.val_sequence_reset)

			feed_dict_validation = model.feed_val_dic(**validation_dict)

			validation_loss, validation_summary, validation_predictions = model.validation_step(sess, feed_dict_train)

			ellapsed_time = time.time() - start_time

			print(Fore.BLUE + "\n\nEpoch {:d}, Steps: {:d}, Train loss: {:01.2f}, Validation loss: {:01.2f}, Epoch time: {:01.2f} sec"
			      .format(epoch + 1, step, np.mean(avg_training_loss), validation_loss, ellapsed_time)+Style.RESET_ALL)

			if tensorboard_logging:
				model.summary_writer.add_summary(model_output["summary"], step)
				model.summary_writer.flush()
				model.validation_summary_writer.add_summary(validation_summary, step)
				model.validation_summary_writer.flush()

			# Plot Global and Local Scenarios to validate datasets
			if args.debug_plotting:
				for seq_index in range(args.batch_size):
					for t in range(args.truncated_backprop_length):
						data_prep.plot_global_scenario(batch_grid, batch_x, batch_y, batch_goal, other_agents_pos,
						                     model_output["model_predictions"], t, seq_index)
						data_prep.plot_local_scenario(batch_grid, batch_x, batch_y, batch_goal, other_agents_pos,
						                    model_output["model_predictions"], t, seq_index)

			with open(args.model_path + "/tf_log", 'a') as f:
				f.write(str(step) + '\n')
			curr_loss = (validation_loss+np.mean(avg_training_loss))/2.0
			if curr_loss < best_loss:
				save_path = args.model_path + '/model_ckpt'
				model.full_saver.save(sess, save_path, global_step=step)
				best_loss = curr_loss
				print(Fore.LIGHTCYAN_EX+'Step {}: Saving model under {}'.format(step, save_path))

		step = step + 1

	write_summary(training_loss[-1], args)
	full_path = args.model_path + '/final-model.ckpt'
	#model.full_saver.save(sess, full_path)
	print('Saved final model under "{}"'.format(full_path))


	if tensorboard_logging:
		model.summary_writer.close()

sess.close()
