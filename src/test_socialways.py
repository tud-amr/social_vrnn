import sys
import os

sys.path.append('../')
import numpy as np
import argparse
import pylab as pl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pickle as pkl
import importlib
# import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import random
import torch
import torch.optim as opt
import json
from copy import deepcopy
import colorama
from colorama import Fore, Style

if sys.version_info[0] < 3:
	sys.path.append('../src/data_utils')
	sys.path.append('../src/models')
	import DataHandlerLSTM as dhlstm
	from plot_utils import *
	import Support as sup
	from Performance import *
	from utils import *
	import Recorder as rec
	from socialways import *
else:
	from src.data_utils import DataHandlerLSTM as dhlstm
	from src.data_utils.plot_utils import *
	from src.data_utils import Support as sup
	from src.data_utils.Performance import *
	from src.data_utils.utils import *
	from src.data_utils.Recorder import Recorder as rec
	from src.external.socialways import *


# Model directories
def parse_args():
	parser = argparse.ArgumentParser(description='LSTM model training')

	parser.add_argument('--model_name',
	                    help='Path to directory that comprises the model (default="model_name").',
	                    type=str, default="VGDNN_simple")
	parser.add_argument('--num_test_sequences', help='Number of test sequences', type=int, default=10)
	parser.add_argument('--exp_num', help='Experiment number', type=int, default=9)
	parser.add_argument('--n_samples', help='Number of samples', type=int, default=1)
	parser.add_argument('--scenario', help='Scenario of the dataset (default="").',
	                    type=str, default="datasets/ewap_dataset/seq_eth")
	parser.add_argument('--record', help='Is grid rotated? (default=True).', type=sup.str2bool,
	                    default=True)
	parser.add_argument('--save_figs', help='Save figures?', type=sup.str2bool,
	                    default=True)
	parser.add_argument('--noise_cell_state', help='Adding noise to cell state of the agent', type=float,
	                    default=0.0)
	parser.add_argument('--noise_cell_grid', help='Adding noise to cell state of the grid', type=float,
	                    default=0.0)
	parser.add_argument('--noise_cell_ped', help='Adding noise to cell others pedestrians info', type=float,
	                    default=0.0)
	parser.add_argument('--noise_cell_concat', help='Adding noise to latent state', type=float,
	                    default=0.0)
	parser.add_argument('--real_world_data', help='real_world_data', type=sup.str2bool,
	                    default=False)
	parser.add_argument('--update_state', help='update_state', type=sup.str2bool,
	                    default=False)
	parser.add_argument('--gpu', help='Enable GPU training.', type=sup.str2bool,
	                    default=False)
	parser.add_argument('--freeze_other_agents', help='FReeze other agents.', type=sup.str2bool,
	                    default=False)
	parser.add_argument('--unit_testing', help='Run  Unit Tests.', type=sup.str2bool,
	                    default=False)
	args = parser.parse_args()

	return args


test_args = parse_args()

if test_args.gpu:
	import tensorflow as tf
else:
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
	import tensorflow as tf

cwd = os.getcwd()

model_path = os.path.normpath(cwd + '/../') + '/trained_models/' + test_args.model_name + "/" + str(test_args.exp_num)

print("Loading data from: '{}'".format(model_path))
file = open(model_path + '/model_parameters.pkl', 'rb')
if sys.version_info[0] < 3:
	model_parameters = pkl.load(file)  # ,encoding='latin1')
else:
	model_parameters = pkl.load(file, encoding='latin1')
file.close()
args = model_parameters["args"]

with open(args.model_path + '/model_parameters.json', 'w') as f:
	json.dump(args.__dict__,f)

# change some args because we are doing inference
truncated_backprop_length = args.truncated_backprop_length
args.truncated_backprop_length = 1
args.batch_size = 1
args.keep_prob = 1.0

training_scenario = args.scenario
args.scenario = test_args.scenario
args.real_world_data = test_args.real_world_data
args.dataset = '/' + args.scenario + '.pkl'
data_prep = dhlstm.DataHandlerLSTM(args)

# Only used to create a map from png
# Load Map Parameters
map_params = os.path.join(args.data_path+args.scenario, 'map.json')
with open(map_params) as json_file:
	data = json.load(json_file)
map_args = {"file_name": data["file_name"],
	          "resolution": data["resolution"],
	          "map_size": np.array(data["map_size"]),
            "map_center": np.array(data["map_center"])}

data_prep.processData(**map_args)
if args.normalize_data:
	data_prep.compute_min_max_values()

# Import model
module = importlib.import_module("src.models."+args.model_name)
globals().update(module.__dict__)

model = NetworkModel(args)

"""Social Ways Model"""
model_name = 'socialWays'
model_file = '../trained_models/' + model_name + '-' + test_args.scenario + '.pt'

if "eth" in test_args.scenario:
	model_file = '../trained_models/' + model_name + '-seq_eth.pt'
if "hotel" in test_args.scenario:
	model_file = '../trained_models/' + model_name + '-seq_hotel.pt'
if "zara_01" in test_args.scenario:
	model_file = '../trained_models/' + model_name + '-zara_01.pt'
if "st" in test_args.scenario:
	model_file = '../trained_models/' + model_name + '-st.pt'
if "zara_02" in test_args.scenario:
	model_file = '../trained_models/' + model_name + '-zara_02.pt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder = EncoderLstm(hidden_size, n_lstm_layers).to(device)
feature_embedder = EmbedSocialFeatures(num_social_features, social_feature_size).to(device)
attention = AttentionPooling(hidden_size, social_feature_size).to(device)
#use_social = False
# Decoder
decoder = DecoderFC(hidden_size + social_feature_size + noise_len).to(device)
# decoder = DecoderLstm(social_feature_size + VEL_VEC_LEN + noise_len, traj_code_len).to(device)

# The Generator parameters and their optimizer
predictor_params = chain(attention.parameters(), feature_embedder.parameters(),
                         encoder.parameters(), decoder.parameters())
predictor_optimizer = opt.Adam(predictor_params, lr=lr_g, betas=(0.9, 0.999))

# The Discriminator parameters and their optimizer
D = Discriminator(n_next, hidden_size, n_latent_codes).to(device)
D_optimizer = opt.Adam(D.parameters(), lr=lr_d, betas=(0.9, 0.999))

print('hidden dim = %d | lr(G) =  %.5f | lr(D) =  %.5f' % (hidden_size, lr_g, lr_d))

if os.path.isfile(model_file):
    print('Loading Social-Ways model from ' + model_file)
    checkpoint = torch.load(model_file,map_location=torch.device('cpu'))
    start_epoch = checkpoint['epoch'] + 1

    attention.load_state_dict(checkpoint['attentioner_dict'])
    feature_embedder.load_state_dict(checkpoint['feature_embedder_dict'])
    encoder.load_state_dict(checkpoint['encoder_dict'])
    decoder.load_state_dict(checkpoint['decoder_dict'])
    predictor_optimizer.load_state_dict(checkpoint['pred_optimizer'])

    D.load_state_dict(checkpoint['D_dict'])
    D_optimizer.load_state_dict(checkpoint['D_optimizer'])

# Lists for logging of the input / output data of the model
input_list = []
grid_list = []
goal_list = []
ped_grid_list = []
y_ground_truth_list = []
y_pred_list = []  # uses ground truth as input at every step
other_agents_list = []
all_predictions = []
all_social_prediction = []
all_traj_likelihood = []
trajectories = []
batch_y = []
batch_loss = []

config = tf.ConfigProto(
	device_count={'GPU': 0}
)

def social_ways_predict(obsv_p, noise, n_next, sub_batches=[]):
    # Batch size
    bs = obsv_p.shape[0]
    # Adds the velocity component to the observations.
    # This makes of obsv_4d a batch_sizexTx4 tensor
    obsv_4d = get_traj_4d(obsv_p, [])
    # Initial values for the hidden and cell states (zero)
    lstm_h_c = (torch.zeros(n_lstm_layers, bs, encoder.hidden_size).to(device),
                torch.zeros(n_lstm_layers, bs, encoder.hidden_size).to(device))
    encoder.init_lstm(lstm_h_c[0], lstm_h_c[1])
    # Apply the encoder to the observed sequence
    # obsv_4d: batch_sizexTx4 tensor
    encoder(obsv_4d)
    if len(sub_batches) == 0:
        sub_batches = [[0, obsv_p.size(0)]]

    if use_social:
        features = SocialFeatures(obsv_4d, sub_batches)
        emb_features = feature_embedder(features, sub_batches)
        weighted_features = attention(emb_features, encoder.lstm_h[0].squeeze(), sub_batches)
    else:
        weighted_features = torch.zeros_like(encoder.lstm_h[0].squeeze())

    pred_4ds = []
    last_obsv = obsv_4d[:, -1]
    # For all the steps to predict, applies a step of the decoder
    for ii in range(n_next):
        # Takes the current output of the encoder to feed the decoder
        # Gets the ouputs as a displacement/velocity
        new_v = decoder(encoder.lstm_h[0].view(bs, -1), weighted_features.view(bs, -1), noise).view(bs, 2)
        # Deduces the predicted position
        new_p = new_v + last_obsv[:, :2]
        # The last prediction done will be new_p,new_v
        last_obsv = torch.cat([new_p, new_v], dim=1)
        # Keeps all the predictions
        pred_4ds.append(last_obsv)
        # Applies LSTM encoding to the last prediction
        # pred_4ds[-1]: batch_sizex4 tensor
        encoder(pred_4ds[-1])

    return torch.stack(pred_4ds, 1)

if test_args.unit_testing:
	data_handler = dhlstm.DataHandlerLSTM(args)
	data_handler.unit_test_data_(map_args)

with tf.Session(config=config) as sess:
	model.warmstart_model(args, sess)
	try:
			model.warmstart_convnet(args, sess)
	except:
		print("")

	for exp_id in range(np.minimum(test_args.num_test_sequences,len(data_prep.trajectory_set)-1)):
		predictions = []
		social_predictions = []
		traj_likelihood = []
		# sample a trajectory id for testing
		traj_id = random.randint(0, len(data_prep.trajectory_set) - 1)
		batch_x, batch_vel, batch_pos,batch_goal, batch_grid, other_agents_info, batch_target,batch_end_pos, other_agents_pos, traj = data_prep.getTrajectoryAsBatch(
			traj_id,freeze = test_args.freeze_other_agents)  # trajectory_set random.randint(0, len(data_prep.dataset) - 1)

		trajectories.append(traj)
		x_input_series = np.zeros([0, (args.prev_horizon + 1) * args.input_dim])
		goal_input_series = np.zeros([0, 2])
		grid_input_series = np.zeros(
			[0, int(args.submap_width / args.submap_resolution), int(args.submap_height / args.submap_resolution)])
		if args.others_info == "relative":
			ped_grid_series = np.zeros([0, args.n_other_agents,args.pedestrian_vector_dim])
		elif "sequence" in args.others_info:
			ped_grid_series = np.zeros([0, args.n_other_agents, args.pedestrian_vector_dim*args.prediction_horizon])
		elif args.others_info == "prev_sequence":
			ped_grid_series = np.zeros([0, args.n_other_agents, args.pedestrian_vector_dim *(args.prev_horizon+1)])
		elif args.others_info == "sequence2":
			ped_grid_series = np.zeros([0, args.n_other_agents, args.prediction_horizon,args.pedestrian_vector_dim])
		else:
			ped_grid_series = np.zeros([0, args.pedestrian_vector_dim])
		y_ground_truth_series = np.zeros([0, args.prediction_horizon * 2])
		y_pred_series = np.zeros([0, args.n_mixtures * args.prediction_horizon * args.output_pred_state_dim])

		batch_y.append(batch_target)
		model.reset_test_cells(np.ones((args.batch_size)))
		cell_state_list= []
		cell_ped_list = []
		cell_concat_list = []
		if "grid" in args.model_name:
			batch_ped_grid_backup = np.zeros_like(batch_grid)
			data_prep.add_other_agents_to_grid(batch_ped_grid_backup, batch_x, [other_agents_pos])

		for step in range(batch_x.shape[1]):
			samples = []
			social_samples = []
			# Assemble feed dict for training
			dict = {"batch_x": batch_x,
							"batch_vel": batch_vel,
			        "batch_pos": batch_pos,
			        "batch_grid": batch_grid,
			        "batch_ped_grid": other_agents_info,
			        "step": step,
			        "batch_goal": batch_goal,
			        "state_noise": 0.0,
			        "grid_noise": 0.0,
			        "ped_noise": 0.0,
			        "concat_noise": 0.0,
			        "other_agents_pos": [other_agents_pos]
			}
			feed_dict_ = model.feed_test_dic(**dict)

			# Append to logging series
			x_input_series = np.append(x_input_series, batch_x[:, step, :], axis=0)
			grid_input_series = np.append(grid_input_series, batch_grid[:, step, :, :], axis=0)
			goal_input_series = np.append(goal_input_series, batch_goal[:, step, :], axis=0)

			y_ground_truth_series = np.append(y_ground_truth_series, batch_target[:, step, :], axis=0)

			y_model_pred, likelihood = model.predict(sess, feed_dict_, True)

			# Backup cell states for later analysis
			cell_state_list.append(model.test_cell_state_current[0,:])
			#cell_ped_list.append(model.test_cell_state_current_lstm_ped[0, :])
			cell_concat_list.append(model.test_cell_state_current_lstm_concat[0, :])

			# Rotate predictions to global frame
			if args.rotated_grid:
				heading = math.atan2(batch_vel[0, step, 1], batch_vel[0, step, 0])
				rot_mat = np.array([[np.cos(heading), -np.sin(heading)], [np.sin(heading), np.cos(heading)]])
				for pred_step in range(args.prediction_horizon):
					y_model_pred[0][0, 2 * pred_step:2 * pred_step + 2] = np.dot(rot_mat, y_model_pred[0][0,2 * pred_step:2 * pred_step + 2])
				samples.append(y_model_pred[0])
			else:
				samples.append(y_model_pred[:,0,:])

			# If sample more than one trajectory from the model
			for sample_id in range(test_args.n_samples - 1):
				dict = {"batch_x": batch_x,
				        "batch_vel": batch_vel,
				        "batch_pos": batch_pos,
				        "batch_grid": batch_grid,
				        "batch_ped_grid": other_agents_info,
				        "step": step,
				        "batch_goal": batch_goal,
				        "state_noise": test_args.noise_cell_state,
				        "grid_noise": test_args.noise_cell_grid,
				        "ped_noise": test_args.noise_cell_ped,
				        "concat_noise": test_args.noise_cell_concat,
				        "other_agents_pos": [other_agents_pos]
				        }
				feed_dict_ = model.feed_test_dic(**dict)
				y_model_pred, likelihood = model.predict(sess, feed_dict_, test_args.update_state)
				samples.append(y_model_pred[:,0,:])

			traj_likelihood.append(likelihood)
			predictions.append(samples)

			# Assemble Pytorch Social Ways model feed dict for training
			positions = np.zeros((1,args.prev_horizon+1,2))
			velocities = np.zeros((1,args.prev_horizon+1,2))
			for j in range(int(batch_x.shape[2]/4)):
				positions[0,args.prev_horizon-j,:] = [batch_x[:, step, j*4],batch_x[0, step, j*4+1]]

			positions_norm = data_prep.normalize_pos(positions,inPlace=False)
			obsv = torch.FloatTensor(positions_norm).to(device)
			for sample_id in range(30):
				noise = torch.FloatTensor(torch.rand(1, noise_len)).to(device)
				pred_hat_4d = social_ways_predict(obsv, noise, n_next)

				social_predicted_positions = pred_hat_4d.data.cpu().numpy()[0,:,:2]
				social_predicted_velocities = pred_hat_4d.data.cpu().numpy()[0, :, 2:]
				social_predicted_positions_normalized = data_prep.denormalize_pos(social_predicted_positions,inPlace=False)
				social_predicted_velocities_normalized = data_prep.denormalize_vel(social_predicted_velocities,shift=False,inPlace=False)

				social_samples.append(social_predicted_positions_normalized)

			social_predictions.append(social_samples)

		all_predictions.append(predictions)
		all_social_prediction.append(social_predictions)
		all_traj_likelihood.append(traj_likelihood)
		input_list.append(x_input_series)
		goal_list.append(goal_input_series)
		grid_list.append(grid_input_series)
		y_ground_truth_list.append(y_ground_truth_series)
		other_agents_list.append(other_agents_pos)
		# update progress bar

sess.close()

if test_args.record:
		recorder = rec(args, data_prep.agent_container.occupancy_grid)
		if ( "real_world" in test_args.scenario) and not test_args.unit_testing:
			print("Real data!!")
			recorder.plot_on_video(input_list, grid_list, all_predictions, y_ground_truth_list,
			                       other_agents_list,
			                       trajectories,all_traj_likelihood, test_args,all_social_prediction)
		else:
			#recorder.plot_on_image(input_list, grid_list, all_predictions, y_ground_truth_list, other_agents_list,
			#	                       trajectories,test_args)
			recorder.animate_local(input_list, grid_list, ped_grid_list, all_predictions, y_ground_truth_list, other_agents_list,
		                 trajectories,test_args)
			recorder.animate_global(input_list, grid_list, all_predictions, y_ground_truth_list,
			                       other_agents_list,
			                       trajectories, all_traj_likelihood,test_args)

		print("Recorder is done!")
else:
	print("Performance tests")
	pred_error, pred_error_summary_lstm = compute_trajectory_prediction_mse(args, trajectories, all_predictions)
	pred_fde, pred_error_summary_lstm_fde = compute_trajectory_fde(args, trajectories, all_predictions)
	diversity, diversity_summary = compute_2_wasserstein(args, all_predictions)
	args.scenario = training_scenario
	args.truncated_backprop_length = truncated_backprop_length
	write_results_summary(np.mean(pred_error_summary_lstm), np.mean(pred_error_summary_lstm_fde), np.mean(diversity_summary), args, test_args)
	print(
		Fore.LIGHTBLUE_EX + "\nMSE: {:01.2f}, FDE: {:01.2f}, DIVERSITY: {:01.2f}".format(np.mean(pred_error_summary_lstm), np.mean(pred_error_summary_lstm_fde),np.mean(diversity_summary))+Style.RESET_ALL)
