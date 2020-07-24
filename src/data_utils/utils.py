import os
import csv
import sys
import pickle as pkl
import argparse

sys.path.append('../src/data_utils')
if sys.version_info[0] < 3:
	import Support as sup
else:
	import src.data_utils.Support as sup

sys.path.append('../src/data_utils')
class bcolors:
	HEADER = '\033[95m'
	OKBLUE = '\033[94m'
	OKGREEN = '\033[92m'
	WARNING = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'


def write_summary(training_loss, args):
	if not os.path.isfile("trained_models_summary.csv"):
		with open("trained_models_summary.csv", 'w') as csvfile:
			# Write header
			writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
			writer.writerow(["Model name", "Dataset", "Ped Info Size",
			                 "Batch size", "Training loss", "MSE", "FDE",
			                 "N epochs", "Learning rate", "Grad clip", "N mixtures","Test Dataset"])
			writer.writerow(
				[args.model_name, args.scenario, args.pedestrian_vector_dim, args.batch_size,
				 training_loss, 0, 0, args.n_epochs,
				 args.learning_rate_init, args.grads_clip, args.n_mixtures])
	else:
		with open("trained_models_summary.csv", 'r') as readFile:
			reader = csv.reader(readFile)
			lines = list(reader)
		with open("trained_models_summary.csv", 'w') as csvfile:
			writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
			if len(lines) > args.exp_num + 1:
				for i in range(len(lines)):
					if i == args.exp_num + 1:
						writer.writerow(
							[args.model_name + "_" + str(args.exp_num), args.scenario, args.pedestrian_vector_dim, args.batch_size,
							 training_loss, 0, 0, args.n_epochs,
							 args.learning_rate_init, args.grads_clip, args.n_mixtures])
					else:
						writer.writerow(lines[i])
			else:
				writer.writerows(lines)
				writer.writerow(
					[args.model_name, args.scenario, args.pedestrian_vector_dim, args.batch_size,
					 training_loss, 0, 0, args.n_epochs,
					 args.learning_rate_init, args.grads_clip, args.n_mixtures])


def write_results_summary(mse, fde, avg_div, args, test_args):
	if not os.path.isfile(test_args.model_name + "_summary.csv"):
		with open(test_args.model_name + "_summary.csv", 'w') as csvfile:
			# Write header
			writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
			writer.writerow(["Model name", "Dataset", "Ped Info Size",
			                 "Batch size", "MSE", "FDE","Diversity",
			                 "dt", "Prediction Horizon", "Previous Steps", "tbpt","Test Dataset","Others Info"])
			writer.writerow(
				[args.model_name+ "_" + str(args.exp_num), args.scenario, args.pedestrian_vector_dim, args.batch_size,
				 mse, fde,avg_div, args.dt,
				 args.prediction_horizon, args.prev_horizon, args.truncated_backprop_length,test_args.scenario,args.others_info])

	else:
		with open(test_args.model_name + "_summary.csv", 'r') as readFile:
			reader = csv.reader(readFile)
			lines = list(reader)
		with open(test_args.model_name + "_summary.csv", 'w') as csvfile:
			writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
			if len(lines) > 3000:#args.exp_num + 1:
				for i in range(len(lines)):
					if i == args.exp_num + 1:
						writer.writerow(
							[args.model_name + "_" + str(args.exp_num), args.scenario, args.pedestrian_vector_dim, args.batch_size,
							 mse, fde, avg_div,args.dt,
							 args.prediction_horizon, args.prev_horizon, args.truncated_backprop_length, test_args.scenario,args.others_info])
					else:
						writer.writerow(lines[i])
			else:
				writer.writerows(lines)
				writer.writerow(
					[args.model_name + "_" + str(args.exp_num), args.scenario, args.pedestrian_vector_dim, args.batch_size,
					 mse, fde,avg_div, args.dt,
					 args.prediction_horizon, args.prev_horizon, args.truncated_backprop_length, test_args.scenario,args.others_info])

def write_keras_results_summary(loss, args):
	if not os.path.isfile(args.model_name + "_summary.csv"):
		with open(args.model_name + "_summary.csv", 'w') as csvfile:
			# Write header
			writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
			writer.writerow(["Model name", "Dataset", "Ped Info Size", "Batch size", "Number of steps", "Loss"])
			writer.writerow( [args.model_name+ "_" + str(args.exp_num), args.scenario, args.pedestrian_vector_dim, args.batch_size, args.n_steps, loss] )

	else:
		with open(args.model_name + "_summary.csv", 'r') as readFile:
			reader = csv.reader(readFile)
			lines = list(reader)
		with open(args.model_name + "_summary.csv", 'w') as csvfile:
			writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
			if len(lines) > 3000:#args.exp_num + 1:
				for i in range(len(lines)):
					if i == args.exp_num + 1:
						writer.writerow( [args.model_name+ "_" + str(args.exp_num), args.scenario, args.pedestrian_vector_dim, args.batch_size, args.n_steps, loss] )
					else:
						writer.writerow(lines[i])
			else:
				writer.writerows(lines)
				writer.writerow( [args.model_name+ "_" + str(args.exp_num), args.scenario, args.pedestrian_vector_dim, args.batch_size, args.n_steps, loss] )


def convert_dataset_python_version(pwd,pwd_out):
	print("Loading data from: '{}'".format(pwd))
	file = open(pwd, 'rb')
	tmp_self = pkl.load(file, encoding='latin1')
	file.close()
	file = open(pwd_out, 'wb')
	pkl.dump(tmp_self, file, protocol=2)


def model_selector(args):
	if "modelKerasRNN_arbitraryAgents" in args.model_name:
		if sys.version_info[0] < 3:
			from modelKerasRNN_arbitraryAgents import NetworkModel
		else:
			from src.models.modelKerasRNN_arbitraryAgents import NetworkModel
	elif "IntNet" in args.model_name:
		if sys.version_info[0] < 3:
			from IntNet import NetworkModel
		else:
			from src.models.IntNet import NetworkModel
	elif "Grid" in args.model_name:
		if sys.version_info[0] < 3:
			from VGDNN_Grid import NetworkModel
		else:
			from src.models.VGDNN_Grid import NetworkModel
	elif "no_grid" in args.model_name:
		if sys.version_info[0] < 3:
			from RNN_no_grid_goal import NetworkModel
		else:
			from src.models.RNN_no_grid_goal import NetworkModel
	elif "goal" in args.model_name:
		if sys.version_info[0] < 3:
			from RNN_goal import NetworkModel
		else:
			from src.models.RNN_goal import NetworkModel
	elif "simple" in args.model_name:
		if sys.version_info[0] < 3:
			from RNN_simple import NetworkModel
		else:
			from src.models.RNN_simple import NetworkModel
	elif "RNN" in args.model_name:
		if sys.version_info[0] < 3:
			from RNN import NetworkModel
		else:
			from src.models.RNN import NetworkModel
	elif "KL" in args.model_name:
		if sys.version_info[0] < 3:
			from VGDNN_KL import NetworkModel
		else:
			from src.models.VGDNN_KL import NetworkModel
	elif "diversity" in args.model_name:
		if sys.version_info[0] < 3:
			from VGDNN_attention_diversity import NetworkModel
		else:
			from src.models.VGDNN_attention_diversity import NetworkModel
	elif "ped" in args.model_name:
		if sys.version_info[0] < 3:
			from VGDNN_ped_attention import NetworkModel
		else:
			from src.models.VGDNN_ped_attention import NetworkModel
	elif "attention" in args.model_name:
		if sys.version_info[0] < 3:
			from VGDNN_attention import NetworkModel
		else:
			from src.models.VGDNN_attention import NetworkModel
	elif "multihead" in args.model_name:
		if sys.version_info[0] < 3:
			from VGDNN_multihead_attention import NetworkModel
		else:
			from src.models.VGDNN_multihead_attention import NetworkModel
	elif "simple" in args.model_name:
		if sys.version_info[0] < 3:
			from VGDNN_simple import NetworkModel
		else:
			from src.models.VGDNN_simple import NetworkModel
	elif "pos" in args.model_name:
		if sys.version_info[0] < 3:
			from VGDNN_pos import NetworkModel
		else:
			from src.models.VGDNN_pos import NetworkModel
	else:
		if sys.version_info[0] < 3:
			from VGDNN import NetworkModel
		else:
			from src.models.VGDNN import NetworkModel

	return NetworkModel(args)

def parse_args(defaults, parse_type):
	parser = argparse.ArgumentParser(description='LSTM model training')

	parser.add_argument('--model_name', help='Path to directory that comprises the model (default="model_name").', type=str, default=defaults['model_name'])
	parser.add_argument('--exp_num', help='Experiment number', type=int, default=defaults['exp_num'])
	parser.add_argument('--scenario', help='Scenario of the dataset (default="").', type=str, default=defaults['scenario'])
	parser.add_argument('--data_path', help='Path to directory that saves pickle data (default=" ").', type=str, default=defaults['data_path'])

	if parse_type == "Train" or parse_type == "train":
		parser.add_argument('--input1_type', help='Data type for the first input', type=str, default=defaults['input1_type'])
		parser.add_argument('--input2_type', help='Data type for the second input', type=str, default=defaults['input2_type'])
		parser.add_argument('--model_path', help='Path to directory to save the model (default=""../trained_models/"+model_name").', type=str, default=defaults['model_path'])
		parser.add_argument('--pretrained_convnet_path', help='Path to directory that comprises the pre-trained convnet model (default=" ").', type=str, default=defaults['pretrained_convnet_path'])
		parser.add_argument('--log_dir', help='Path to the log directory of the model (default=""../trained_models/"+model_name").', type=str, default=defaults['log_dir'])
		parser.add_argument('--scenario_val', help='Dataset pkl file', type=str, default= defaults['scenario_val'])
		parser.add_argument('--real_world_data', help='Real world dataset (default=True).', type=sup.str2bool, default=defaults['real_world_data'])
		parser.add_argument('--dataset', help='Dataset pkl file', type=str, default= defaults['scenario'] + '.pkl')
		parser.add_argument('--dataset_val', help='Dataset pkl file', type=str, default= defaults['scenario_val'] + '.pkl')
		parser.add_argument('--data_handler', help='Datahandler class needed to load the data', type=str, default='LSTM')
		parser.add_argument('--warmstart_model', help='Restore from pretained model (default=False).', type=bool, default=defaults['warmstart_model'])
		parser.add_argument('--warm_start_convnet', help='Restore from pretained convnet model (default=False).', type=bool, default=defaults['warm_start_convnet'])
		parser.add_argument('--dt', help='Data samplig time (default=0.3).', type=float, default=defaults['dt'])
		parser.add_argument('--n_steps', help='Number of epochs (default=10000).', type=int, default=defaults['n_steps'])
		parser.add_argument('--batch_size', help='Batch size for training (default=32).', type=int, default=defaults['batch_size'])
		parser.add_argument('--regularization_weight', help='Weight scaling of regularizer (default=0.01).', type=float, default=defaults['regularization_weight'])
		parser.add_argument('--keep_prob', help='Dropout (default=0.8).', type=float, default=defaults['keep_prob'])
		parser.add_argument('--learning_rate_init', help='Initial learning rate (default=0.005).', type=float, default=defaults['learning_rate_init'])
		parser.add_argument('--beta_rate_init', help='Initial beta rate (default=0.005).', type=float, default=defaults['beta_rate_init'])
		parser.add_argument('--dropout', help='Enable Dropout', type=sup.str2bool, default=defaults['dropout'])
		parser.add_argument('--grads_clip', help='Gridient clipping (default=10.0).', type=float, default=defaults['grads_clip'])
		parser.add_argument('--truncated_backprop_length', help='Backpropagation length during training (default=5).', type=int, default=defaults['truncated_backprop_length'])
		parser.add_argument('--prediction_horizon', help='Length of predicted sequences (default=10).', type=int, default=defaults['prediction_horizon'])
		parser.add_argument('--prev_horizon', help='Previous seq length.', type=int, default=defaults['prev_horizon'])
		parser.add_argument('--rnn_state_size', help='Number of RNN / LSTM units (default=16).', type=int, default=defaults['rnn_state_size'])
		parser.add_argument('--rnn_state_size_lstm_ped', help='Number of RNN / LSTM units of the pedestrian lstm layer (default=32).', type=int, default=defaults['rnn_state_size_lstm_ped'])
		parser.add_argument('--rnn_state_size_bilstm_ped', help='Number of RNN / LSTM units of the pedestrian bidirectional lstm layer (default=32).',type=int, default=defaults['rnn_state_size_bilstm_ped'])
		parser.add_argument('--rnn_state_size_lstm_grid', help='Number of RNN / LSTM units of the grid lstm layer (default=32).', type=int, default=defaults['rnn_state_size_lstm_grid'])
		parser.add_argument('--rnn_state_size_lstm_concat', help='Number of RNN / LSTM units of the concatenation lstm layer (default=32).', type=int, default=defaults['rnn_state_size_lstm_concat'])
		parser.add_argument('--prior_size', help='prior_size', type=int, default=defaults['prior_size'])
		parser.add_argument('--latent_space_size', help='latent_space_size', type=int, default=defaults['latent_space_size'])
		parser.add_argument('--x_dim', help='x_dim', type=int, default=defaults['x_dim'])
		parser.add_argument('--fc_hidden_unit_size', help='Number of fully connected layer units after LSTM layer (default=64).', type=int, default=defaults['fc_hidden_unit_size'])
		parser.add_argument('--input_state_dim', help='Input state dimension (default=).', type=int, default=defaults['input_state_dim'])
		parser.add_argument('--input_dim', help='Input state dimension (default=).', type=float, default=defaults['input_dim'])
		parser.add_argument('--output_dim', help='Output state dimension (default=).', type=float, default=defaults['output_dim'])
		parser.add_argument('--output_pred_state_dim', help='Output prediction state dimension (default=).', type=int, default=defaults['output_pred_state_dim'])
		parser.add_argument('--cmd_vector_dim', help='Command control dimension.', type=int, default=defaults['cmd_vector_dim'])
		parser.add_argument('--n_mixtures', help='Number of modes (default=).', type=int, default=defaults['n_mixtures'])
		parser.add_argument('--pedestrian_vector_dim', help='Number of angular grid sectors (default=72).', type=int, default=defaults['pedestrian_vector_dim'])
		parser.add_argument('--pedestrian_vector_state_dim', help='Number of angular grid sectors (default=2).', type=int, default=defaults['pedestrian_vector_state_dim'])
		parser.add_argument('--max_range_ped_grid', help='Maximum pedestrian distance (default=2).', type=float, default=defaults['max_range_ped_grid'])
		parser.add_argument('--pedestrian_radius', help='Pedestrian radius (default=0.3).', type=float, default=defaults['pedestrian_radius'])
		parser.add_argument('--debug_plotting', help='Plotting for debugging (default=False).', type=int, default=0)
		parser.add_argument('--print_freq', help='Print frequency of training info (default=100).', type=int, default=defaults['print_freq'])
		parser.add_argument('--save_freq', help='Save frequency of the temporary model during training. (default=20k).', type=int, default=defaults['save_freq'])
		parser.add_argument('--patience', help='Patience to stop the training loop early', type=int, default=defaults['patience'])
		parser.add_argument('--noise', help='Likelihood? (default=True).', type=sup.str2bool, default=False)
		parser.add_argument('--agents_on_grid', help='Likelihood? (default=True).', type=sup.str2bool, default=defaults['agents_on_grid'])
		parser.add_argument('--normalize_data', help='Normalize? (default=False).', type=sup.str2bool, default=defaults['normalize_data'])
		parser.add_argument('--rotated_grid', help='Rotate grid? (default=False).', type=sup.str2bool, default=defaults['rotated_grid'])
		parser.add_argument('--centered_grid', help='Center grid? (default=False).', type=sup.str2bool, default=defaults['centered_grid'])
		parser.add_argument('--sigma_bias', help='Percentage of the dataset used for trainning', type=float, default=0)
		parser.add_argument('--submap_width', help='width of occupancy grid', type=int, default=defaults['submap_width'])
		parser.add_argument('--submap_height', help='height of occupancy grid', type=int, default=defaults['submap_height'])
		parser.add_argument('--submap_resolution', help='Map resolution.', type=float, default=defaults['submap_resolution'])
		parser.add_argument('--min_buffer_size', help='Minimum buffer size (default=1000).', type=int, default=1000)
		parser.add_argument('--max_buffer_size', help='Maximum buffer size (default=100k).', type=int, default=100000)
		parser.add_argument('--max_trajectories', help='maximum number of trajectories to be recorded', type=int, default=30)
		parser.add_argument('--end_to_end', help='End to end trainning.', type=sup.str2bool, default=False)
		parser.add_argument('--predict_positions', help='predict_positions.', type=sup.str2bool, default=defaults['predict_positions'])
		parser.add_argument('--gpu', help='Enable GPU training.', type=sup.str2bool, default=True)
		parser.add_argument('--relative_info', help='Use relative info for other agents.', type=sup.str2bool, default=True)
		parser.add_argument('--regulate_log_loss', help='Enable GPU training.', type=sup.str2bool, default=defaults['regulate_log_loss'])
		parser.add_argument('--diversity_update', help='diversity_update', type=sup.str2bool, default=defaults['diversity_update'])
		parser.add_argument('--topics_config', help='yaml file containg subscription topics (default=" ").', type=str, default='../config/topics.yaml')
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
	else:
		parser.add_argument('--num_test_sequences', help='Number of test sequences', type=int, default=10)
		parser.add_argument('--n_samples', help='Number of samples', type=int, default=1)
		parser.add_argument('--record', help='Is grid rotated? (default=True).', type=sup.str2bool, default=True)
		parser.add_argument('--save_figs', help='Save figures?', type=sup.str2bool, default=False)
		parser.add_argument('--noise_cell_state', help='Adding noise to cell state of the agent', type=float, default=0)
		parser.add_argument('--noise_cell_grid', help='Adding noise to cell state of the grid', type=float, default=5)
		parser.add_argument('--real_world_data', help='real_world_data', type=sup.str2bool, default=False)
		parser.add_argument('--update_state', help='update_state', type=sup.str2bool, default=False)
		parser.add_argument('--gpu', help='Enable GPU training.', type=sup.str2bool, default=False)
		parser.add_argument('--freeze_other_agents', help='Freeze other agents.', type=sup.str2bool, default=False)
		parser.add_argument('--unit_testing', help='Run  Unit Tests.', type=sup.str2bool,default=False)

	args = parser.parse_args()

	return args

if __name__ == '__main__':
	convert_dataset_python_version("/home/bbrito/Code/I-LSTM/data/2_agents_swap/trajs/GA3C-CADRL-10.pkl",
	                               "/home/bbrito/Code/I-LSTM/data/2_agents_swap/trajs/GA3C-CADRL-10-py27.pkl")
