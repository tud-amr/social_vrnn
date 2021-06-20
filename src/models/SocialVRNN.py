from tensorflow_probability import distributions as tfd
import sys

if sys.version_info[0] < 3:
    sys.path.append('../src/external')
    from cells.vrnn_cell import VariationalRNNCell as vrnn_cell
    from models.tf_utils import *
else:
    from src.cells.vrnn_cell import VariationalRNNCell as vrnn_cell
    from src.models.tf_utils import *
import numpy as np
import os
from colorama import Fore, Style


class NetworkModel():

    def __init__(self, args, is_training=True, batch_size=None):
        self.log_dir = args.log_dir
        self.args = args
        # Input- / Output dimensions
        self.input_dim = args.input_dim  # [x, y, vx, vy]
        self.input_state_dim = args.input_state_dim  # [vx, vy]
        self.output_dim = args.output_dim
        self.output_pred_state_dim = self.args.output_pred_state_dim
        self.truncated_backprop_length = args.truncated_backprop_length
        self.prev_horizon = args.prev_horizon
        self.n_mixtures = args.n_mixtures
        self.prediction_horizon = args.prediction_horizon
        self.output_placeholder_dim = self.prediction_horizon * self.output_dim
        self.output_vec_dim = self.prediction_horizon * self.output_pred_state_dim * self.n_mixtures
        self.pedestrian_vector_dim = args.pedestrian_vector_dim
        self.learning_rate_init = args.learning_rate_init
        self.beta_rate_init = args.beta_rate_init
        # Network attributes
        self.rnn_state_size = args.rnn_state_size
        self.rnn_state_size_lstm_grid = args.rnn_state_size_lstm_grid
        self.rnn_state_size_lstm_ped = args.rnn_state_size_lstm_ped
        self.rnn_state_size_lstm_concat = args.rnn_state_size_lstm_concat
        self.fc_hidden_unit_size = args.fc_hidden_unit_size
        self.prior_hidden_size = self.rnn_state_size + self.rnn_state_size_lstm_grid + self.rnn_state_size_lstm_ped
        try:
            self.sigma_bias = args.sigma_bias
        except:
            self.sigma_bias = 0.0
        # Training parameters
        self.lambda_ = args.regularization_weight
        self.batch_size = None
        self.is_training = is_training
        self.grads_clip = args.grads_clip
        self.regularization_weight = args.regularization_weight
        self.grid_width = int(args.submap_width / args.submap_resolution)
        self.grid_height = int(args.submap_height / args.submap_resolution)
        # Specify placeholders
        self.input_state_placeholder = tf.placeholder(dtype=tf.float32,
                                                      shape=[self.batch_size, self.truncated_backprop_length,
                                                             self.input_state_dim * (self.prev_horizon + 1)],
                                                      name='input_state')
        self.input_grid_placeholder = tf.placeholder(dtype=tf.float32,
                                                     shape=[self.batch_size, self.truncated_backprop_length,
                                                            self.grid_width, self.grid_height], name='input_grid')
        self.input_ped_grid_placeholder = tf.placeholder(dtype=tf.float32,
                                                         shape=[self.batch_size, self.truncated_backprop_length,
                                                                self.pedestrian_vector_dim*self.args.n_other_agents], name='ped_grid')
        self.output_placeholder = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.truncated_backprop_length,
                                                                          self.output_placeholder_dim], name='output')
        self.diversity_placeholder = tf.placeholder(dtype=tf.float32,
                                                    shape=[self.batch_size, self.truncated_backprop_length,
                                                           self.output_placeholder_dim], name='output')
        self.dropout_placeholder = tf.placeholder(dtype=tf.float32, name='dropout')
        self.step = tf.placeholder(dtype=tf.float32,
                                   shape=[], name='global_step')
        self.seq_length = tf.placeholder(tf.int32, [None])
        # Initialize hidden states with zeros
        self.cell_state_current = np.zeros([args.batch_size, args.rnn_state_size])
        self.hidden_state_current = np.zeros([args.batch_size, args.rnn_state_size])
        self.cell_state_current_lstm_grid = np.zeros([args.batch_size, args.rnn_state_size_lstm_grid])
        self.hidden_state_current_lstm_grid = np.zeros([args.batch_size, args.rnn_state_size_lstm_grid])
        self.cell_state_current_lstm_ped = np.zeros([args.batch_size, args.rnn_state_size_lstm_ped])
        self.hidden_state_current_lstm_ped = np.zeros([args.batch_size, args.rnn_state_size_lstm_ped])
        self.cell_state_current_lstm_concat = np.zeros([args.batch_size, self.rnn_state_size_lstm_concat])
        self.hidden_state_current_lstm_concat = np.zeros([args.batch_size, self.rnn_state_size_lstm_concat])

        # Cells used for testing
        # Initialize hidden states with zeros
        self.test_cell_state_current = np.zeros([args.batch_size, args.rnn_state_size])
        self.test_hidden_state_current = np.zeros([args.batch_size, args.rnn_state_size])
        self.test_cell_state_current_lstm_grid = np.zeros([args.batch_size, args.rnn_state_size_lstm_grid])
        self.test_hidden_state_current_lstm_grid = np.zeros([args.batch_size, args.rnn_state_size_lstm_grid])
        self.test_cell_state_current_lstm_ped = np.zeros([args.batch_size, args.rnn_state_size_lstm_ped])
        self.test_hidden_state_current_lstm_ped = np.zeros([args.batch_size, args.rnn_state_size_lstm_ped])
        self.test_cell_state_current_lstm_concat = np.zeros([args.batch_size, self.rnn_state_size_lstm_concat])
        self.test_hidden_state_current_lstm_concat = np.zeros([args.batch_size, self.rnn_state_size_lstm_concat])

        # Pedestrian state LSTM
        self.cell_state = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.rnn_state_size],
                                         name='cell_state')  # internal state of the cell (before output gate)
        self.hidden_state = tf.placeholder(dtype=tf.float32, shape=[None, self.rnn_state_size],
                                           name='hidden_state')  # output of the cell (after output gate)
        self.init_state_tuple = tf.contrib.rnn.LSTMStateTuple(self.cell_state, self.hidden_state)

        # Static occupancy grid LSTM
        self.cell_state_lstm_grid = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.rnn_state_size_lstm_grid],
                                                   name='cell_state_lstm_grid')  # internal state of the cell (before output gate)
        self.hidden_state_lstm_grid = tf.placeholder(dtype=tf.float32, shape=[None, self.rnn_state_size_lstm_grid],
                                                     name='hidden_state_lstm_grid')  # output of the cell (after output gate)
        self.init_state_tuple_lstm_grid = tf.contrib.rnn.LSTMStateTuple(self.cell_state_lstm_grid,
                                                                        self.hidden_state_lstm_grid)

        # Pedestrian LSTM
        self.cell_state_lstm_ped = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.rnn_state_size_lstm_ped],
                                                  name='cell_state_lstm_ped')  # internal state of the cell (before output gate)
        self.hidden_state_lstm_ped = tf.placeholder(dtype=tf.float32, shape=[None, self.rnn_state_size_lstm_ped],
                                                    name='hidden_state_lstm_ped')  # output of the cell (after output gate)
        self.init_state_tuple_lstm_ped = tf.contrib.rnn.LSTMStateTuple(self.cell_state_lstm_ped, self.hidden_state_lstm_ped)

        # Concatenation LSTM (state, grid, pedestrians)
        self.cell_state_lstm_concat = tf.placeholder(dtype=tf.float32,
                                                     shape=[self.batch_size, self.rnn_state_size_lstm_concat],
                                                     name='cell_state_lstm_concat')  # internal state of the cell (before output gate)
        self.hidden_state_lstm_concat = tf.placeholder(dtype=tf.float32, shape=[None, self.rnn_state_size_lstm_concat],
                                                       name='hidden_state_lstm_concat')  # output of the cell (after output gate)
        self.init_state_tuple_lstm_concat = tf.contrib.rnn.LSTMStateTuple(self.cell_state_lstm_concat,
                                                                          self.hidden_state_lstm_concat)

        inputs_series = tf.unstack(self.input_state_placeholder, axis=1)
        outputs_series = tf.unstack(self.output_placeholder, axis=1)
        outputs_diversity_series = tf.unstack(self.diversity_placeholder, axis=1)
        # Print network info
        print("Input list length: {}".format(len(inputs_series)))
        print("Output list length: {}".format(len(outputs_series)))
        print("Single input shape: {}".format(inputs_series[0].get_shape()))
        print("Single output shape: {}".format(outputs_series[0].get_shape()))

        ###################### Assemble CNN #####################
        # Process occupancy grid, apply convolutional filters
        self.occ_grid_series = tf.unstack(self.input_grid_placeholder, axis=1)
        conv_grid_feature_series = []
        for output in self.process_grid(self.occ_grid_series):
            conv_grid_feature_series.append(tf.stop_gradient(output))

        # Set up model with variables
        with tf.variable_scope("model") as scope:

            train_encoder = self.args.end_to_end

            with tf.variable_scope('encoder_model') as scope:
                # Pedestrian grid embedding layer
                ped_grid_out_dim = 128
                self.W_pedestrian_grid = self.get_weight_variable(name='w_ped_grid',
                                                                  shape=[ped_grid_out_dim, ped_grid_out_dim],
                                                                  trainable=True)
                self.b_pedestrian_grid = self.get_bias_variable(name='b_ped_grid', shape=[1, ped_grid_out_dim], trainable=True)

                # LSTM cells to process static occupancy grid
                with tf.variable_scope('lstm_grid') as scope:
                    self.cell_grid = tf.nn.rnn_cell.LSTMCell(self.rnn_state_size_lstm_grid, name='basic_lstm_cell',
                                                             trainable=True)
                    self.cell_outputs_series_lstm_grid, self.current_state_lstm_grid = tf.contrib.rnn.static_rnn(self.cell_grid,
                                                                                                                 conv_grid_feature_series,
                                                                                                                 dtype=tf.float32,
                                                                                                                 initial_state=self.init_state_tuple_lstm_grid)

                # Process pedestrian grid
                ped_grid_series = tf.unstack(self.input_ped_grid_placeholder, axis=1)

                # LSTM cells to process pedestrian grid
                with tf.variable_scope('lstm_ped') as scope:
                    self.cell_ped = tf.nn.rnn_cell.LSTMCell(self.rnn_state_size_lstm_ped, name='basic_lstm_cell', trainable=True)
                    ped_grid_feature_series , self.current_state_lstm_ped = tf.nn.dynamic_rnn(self.cell_ped,
                                                                                                               tf.stack(ped_grid_series,axis=1),
                                                                                                               dtype=tf.float32,
                                                                                                               sequence_length=self.seq_length,
                                                                                                               initial_state=self.init_state_tuple_lstm_ped)

                    # Embedding layer of pedestrian grid
                    self.cell_outputs_series_lstm_ped = self.process_pedestrian_grid(tf.unstack(ped_grid_feature_series,axis=1))

                # LSTM cells to process pedestrian state
                with tf.variable_scope('lstm_state') as scope:
                    self.cell = tf.nn.rnn_cell.LSTMCell(self.rnn_state_size, name='basic_lstm_cell', trainable=True)
                    self.cell_outputs_series_state, self.current_state = tf.contrib.rnn.static_rnn(self.cell, inputs_series,
                                                                                                   dtype=tf.float32,
                                                                                                   initial_state=self.init_state_tuple)

                # Concatenate the outputs of the three previous LSTMs and feed in first concatenated LSTM cell
                concat_lstm = []
                for lstm_out, grid_feature, ped_feature in zip(self.cell_outputs_series_state,
                                                               self.cell_outputs_series_lstm_grid,
                                                               self.cell_outputs_series_lstm_ped):
                    concat_lstm.append(
                        tf.concat([lstm_out, grid_feature, ped_feature], axis=1, name="concatenate_lstm_out_with_grid_and_ped"))

                    # Concatenated LSTM layer
                    with tf.variable_scope('generation_model') as scope:

                        # cell_lstm_concat = tf.nn.rnn_cell.LSTMCell(self.rnn_state_size_lstm_concat, name='basic_lstm_cell')
                        cell_lstm_concat = vrnn_cell(args, args.rnn_state_size_lstm_concat, self.output_vec_dim, args.prior_size,
                                                     args.latent_space_size)
                        self.cell_outputs_series_lstm_concat, self.current_state_lstm_concat = tf.contrib.rnn.static_rnn(
                            cell_lstm_concat, concat_lstm, dtype=tf.float32, initial_state=self.init_state_tuple_lstm_concat)

                        # Output reshape
                        enc_mu_series = tf.unstack(tf.stack([o[0] for o in self.cell_outputs_series_lstm_concat]), axis=0)
                        enc_sigma_series = tf.unstack(tf.stack([o[1] for o in self.cell_outputs_series_lstm_concat]), axis=0)
                        self.dec_series = tf.unstack(tf.stack([o[2] for o in self.cell_outputs_series_lstm_concat]), axis=0)
                        prior_mu_series = tf.unstack(tf.stack([o[3] for o in self.cell_outputs_series_lstm_concat]), axis=0)
                        prior_sigma_series = tf.unstack(tf.stack([o[4] for o in self.cell_outputs_series_lstm_concat]), axis=0)

                        # Decoder layers
                        dec_fc_layer = tf.unstack(tf.stack(tf.nn.elu([output for output in self.dec_series])))
                        dec_output_layer = []
                        dec_mux_series = []
                        dec_muy_series = []
                        dec_sigmax_series = []
                        dec_sigmay_series = []
                        dec_pi_series = []
                        # Reusing weights for each truncated back propagation step
                        for t_idx in range(len(dec_fc_layer)):
                            dec_output_layer.append(linear(dec_fc_layer[t_idx], self.output_vec_dim, 50))
                            dec_pi_series.append(linear(dec_fc_layer[t_idx], self.args.n_mixtures, 100))
                        for t_idx in range(len(dec_output_layer)):
                            m_x, m_y, s_x, s_y = tf.split(dec_output_layer[t_idx], 4, axis=1)
                            dec_mux_series.append(m_x)
                            dec_muy_series.append(m_y)
                            dec_sigmax_series.append(s_x)
                            dec_sigmay_series.append(s_y)
                        dec_pi_series = [dense for dense in dec_pi_series]

                # Exponential / stepwise decrease of learning rate
                global_step = tf.Variable(0, trainable=False)
                self.learning_rate = tf.train.exponential_decay(self.learning_rate_init, self.step, decay_steps=10000,
                                                                decay_rate=0.9, staircase=False)

                self.beta = (tf.tanh((tf.to_float(self.step) - 20000) / 5000) + 1) / 2

                prediction_loss_list = []
                loss_list = []
                kl_loss_list = []
                likelihood_loss_list = []
                self.div_loss = []
                self.prediction = []
                self.likelihood = []
                pred = []
                div_loss_over_truncated_back_prop = []

                # Compute Loss Function
                for enc_mu, enc_sigma, dec_mux, dec_muy, dec_sigmax, dec_sigmay, prior_mu, prior_sigma, dec_pi, out, out_div in \
                        zip(enc_mu_series, enc_sigma_series, dec_mux_series, dec_muy_series,
                            dec_sigmax_series, dec_sigmay_series,
                            prior_mu_series, prior_sigma_series, dec_pi_series,
                            outputs_series, outputs_diversity_series):  # prediction dimension: prediction_horizon * output_states (e.g. [x_1, y_1, x_2, y_2, x_3, y_3])

                    # Apply transformations
                    mux = tf.split(dec_mux, self.prediction_horizon, axis=1)
                    muy = tf.split(dec_muy, self.prediction_horizon, axis=1)
                    sigmax = tf.exp(tf.split(dec_sigmax, self.prediction_horizon, axis=1))
                    sigmay = tf.exp(tf.split(dec_sigmay, self.prediction_horizon, axis=1))
                    pi = tf.nn.softmax(dec_pi, axis=1) # shape = [batch size, n_mixtures]

                    self.likelihood.append(pi)
                    pred = []
                    prediction_loss_list = []
                    kl_mode_loss_list = []
                    div_loss_over_horizon = []
                    norm_list = []
                    lossfunc = 0
                    for prediction_step in range(self.prediction_horizon):
                        # Compute Euclidean error for each prediction step within each tbp step
                        idx_x = prediction_step * self.output_dim
                        idx_y = idx_x + 1

                        # ground truth data
                        x_data = tf.reshape(out[:, idx_x], shape=(-1, 1))
                        y_data = tf.reshape(out[:, idx_y], shape=(-1, 1))
                        x_diversity = tf.reshape(out_div[:, idx_x], shape=(-1, 1))
                        y_diversity = tf.reshape(out_div[:, idx_y], shape=(-1, 1))

                        if prediction_step == 0:
                            pred = tf.concat([mux[prediction_step], muy[prediction_step]], 1)
                            pred = tf.concat([pred, sigmax[prediction_step]], 1)
                            pred = tf.concat([pred, sigmay[prediction_step]], 1)
                        else:
                            pred = tf.concat([pred, mux[prediction_step]], 1)
                            pred = tf.concat([pred, muy[prediction_step]], 1)
                            pred = tf.concat([pred, sigmax[prediction_step]], 1)
                            pred = tf.concat([pred, sigmay[prediction_step]], 1)

                        lossfunc = self.get_log_mixlossfunc(mux[prediction_step], muy[prediction_step],
                                                         sigmax[prediction_step], sigmay[prediction_step],
                                                         x_data, y_data, pi)

                        diversity_loss = self.get_log_mixlossfunc(mux[prediction_step], muy[prediction_step],
                                                                sigmax[prediction_step], sigmay[prediction_step],
                                                               x_diversity, y_diversity,1.0)

                        diff_x = tf.square(tf.subtract(mux[prediction_step], x_data))
                        diff_y = tf.square(tf.subtract(muy[prediction_step], y_data))
                        norm = tf.sqrt(diff_x + diff_y)

                        div_loss_over_horizon.append(diversity_loss)
                        prediction_loss_list.append(lossfunc)
                        norm_list.append(norm)

                    self.prediction.append(pred)
                    kl_loss = self.tf_kl_gaussgauss(enc_mu, enc_sigma, prior_mu, prior_sigma)
                    kl_loss_list.append(kl_loss)
                    div_loss_over_truncated_back_prop.append(tf.reduce_mean(div_loss_over_horizon))
                    loss_list.append(tf.reduce_mean(prediction_loss_list))

                    max_norms = tf.reduce_max(norm_list, axis=0)  # (batch size, n_mixtures)
                    groundtruth_likelihood = tf.nn.softmax(tf.multiply(max_norms, -1))

                    likelihood_pred = tf.distributions.Categorical(probs=pi)
                    likelihood_true = tf.distributions.Categorical(probs=groundtruth_likelihood)
                    likelihood_loss = tf.distributions.kl_divergence(likelihood_pred, likelihood_true)
                    likelihood_loss_list.append(likelihood_loss)

            self.prediction = tf.stack(self.prediction, axis=1)
            self.likelihood = tf.stack(self.likelihood, axis=1)

            # Get all trainable variables
            tvars = tf.trainable_variables()

            # L2 loss
            l2 = self.lambda_ * sum(tf.nn.l2_loss(tvar) for tvar in tvars)

            # Reduce mean in all dimensions
            self.div_loss = tf.reduce_mean(div_loss_over_truncated_back_prop)
            self.total_loss = tf.reduce_mean(loss_list, axis=0)+(tf.reduce_mean(kl_loss_list, axis=0))*self.beta
            self.reconstruction_loss = tf.reduce_mean(loss_list, axis=0)
            self.kl_loss = tf.reduce_mean(kl_loss_list, axis=0)
            self.likelihood_loss = tf.stop_gradient(likelihood_loss_list[-1])

            # Optimizer specification
            # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
            # Compute gradients
            train_vars = tf.trainable_variables()
            self.gradients = tf.gradients(self.total_loss, train_vars)
            self.likelihood_gradients = tf.gradients(self.likelihood_loss, train_vars)

            # Clip gradients
            self.grads, _ = tf.clip_by_global_norm(self.gradients, self.grads_clip)
            self.likelihood_grads, _ = tf.clip_by_global_norm(self.likelihood_gradients,
                                                              self.grads_clip)

            # Apply gradients to update model
            self.update = self.optimizer.apply_gradients(zip(self.grads, train_vars),
                                                         global_step=global_step)
            self.likelihood_update = self.optimizer.apply_gradients(
                zip(self.likelihood_grads, train_vars),
                global_step=global_step)
            # self.update = self.optimizer.minimize(self.total_loss, global_step=global_step)
            # Tensorboard summary

            ## Loss
            self.loss_summary = tf.summary.scalar('loss', self.total_loss)
            self.kl_loss_summary = tf.summary.scalar('kl-loss', self.kl_loss)
            self.div_loss_summary = tf.summary.scalar('div_loss', self.div_loss)
            self.reconstruction_loss_summary = tf.summary.scalar('reconstruction_loss', self.reconstruction_loss)
            tf.summary.scalar('learning_rate', self.learning_rate)

            self.summary_writer = tf.summary.FileWriter(logdir=os.path.join(self.log_dir, 'train'), graph=tf.Session().graph)
            self.validation_summary_writer = tf.summary.FileWriter(logdir=os.path.join(self.log_dir, 'validation'),
                                                                   graph=tf.Session().graph)

            self.summary = tf.summary.merge_all()

            # Save / restore all variables
            self.full_saver = tf.train.Saver()

            # Only save / restore the parameters responsible for occupancy grid processing
            self.convnet_saver = tf.train.Saver(self.autoencoder_var_list)
            self.autoencoder_update = self.optimizer.minimize(self.autoencoder_loss, global_step=global_step)


    def feed_dic(self, **kwargs):
        if self.args.rotated_grid:
            batch_x, batch_y = sup.rotate_batch_to_local_frame(kwargs['batch_y'], kwargs['batch_x'])
            batch_x = batch_x[:, :, 2:]
        else:
            batch_x = kwargs['batch_vel']
            batch_y = kwargs['batch_y']
        n_other_agents = np.zeros([self.args.batch_size])
        for i,other in enumerate(kwargs['other_agents_pos']):
            n_other_agents[i] = len(other)
        return {self.input_state_placeholder: batch_x,
                self.input_ped_grid_placeholder: kwargs["batch_ped_grid"],
                self.input_grid_placeholder: kwargs["batch_grid"],
                self.step: kwargs['step'],
                self.seq_length: n_other_agents,
                self.diversity_placeholder: kwargs['batch_div'],
                self.output_placeholder: batch_y,
                self.cell_state: self.cell_state_current,
                self.hidden_state: self.hidden_state_current,
                self.cell_state_lstm_grid: np.random.normal(self.cell_state_current_lstm_grid, 0),
                self.hidden_state_lstm_grid: np.random.normal(self.hidden_state_current_lstm_grid, 0),
                self.cell_state_lstm_ped: self.cell_state_current_lstm_ped,
                self.hidden_state_lstm_ped: self.hidden_state_current_lstm_ped,
                self.cell_state_lstm_concat: self.cell_state_current_lstm_concat,
                self.hidden_state_lstm_concat: self.hidden_state_current_lstm_concat,
                }


    def feed_test_dic(self, **kwargs):
        step = kwargs["step"]
        n_other_agents = np.zeros([self.args.batch_size])
        for i, other in enumerate(kwargs['other_agents_pos']):
            n_other_agents[i] = len(other)
        return {self.input_state_placeholder: np.expand_dims(kwargs["batch_vel"][:, step, :], axis=1),
                self.input_ped_grid_placeholder: np.expand_dims(kwargs["batch_ped_grid"][:, step, :], axis=1),
                self.input_grid_placeholder: np.expand_dims(kwargs["batch_grid"][:, step, :, :], axis=1),
                self.step: 0,
                self.seq_length: n_other_agents,
                self.cell_state: np.random.normal(self.test_cell_state_current.copy(), np.abs(np.mean(np.mean(self.test_cell_state_current))*kwargs["state_noise"])),
                self.hidden_state: np.random.normal(self.test_hidden_state_current.copy(), np.abs(np.mean(np.mean(self.test_cell_state_current))*kwargs["state_noise"])),
                self.cell_state_lstm_grid: np.random.normal(self.test_cell_state_current_lstm_grid.copy(), kwargs["grid_noise"]),
                self.hidden_state_lstm_grid: np.random.normal(self.test_hidden_state_current_lstm_grid.copy(), kwargs["grid_noise"]),
                self.cell_state_lstm_ped: self.test_cell_state_current_lstm_ped,
                self.hidden_state_lstm_ped: self.test_hidden_state_current_lstm_ped,
                self.cell_state_lstm_concat: self.test_cell_state_current_lstm_concat,
                self.hidden_state_lstm_concat: self.test_hidden_state_current_lstm_concat,
                }

    def feed_val_dic(self, n_other_agent = None, **kwargs):
        n_other_agents = np.zeros([self.args.batch_size])
        if n_other_agent is None:
            for i, other in enumerate(kwargs['other_agents_pos']):
                n_other_agents[i] = len(other)
        return {self.input_state_placeholder: kwargs["batch_vel"],
                self.input_grid_placeholder: kwargs["batch_grid"],
                self.input_ped_grid_placeholder: kwargs["batch_ped_grid"],
                self.diversity_placeholder: kwargs['batch_div'],
                self.output_placeholder: kwargs['batch_y'],
                self.seq_length: n_other_agents,
                self.step: 0,
                self.cell_state: np.random.normal(self.test_cell_state_current.copy(), np.abs(self.test_cell_state_current*kwargs["state_noise"])),
                self.hidden_state: np.random.normal(self.test_hidden_state_current.copy(), np.abs(self.test_cell_state_current*kwargs["state_noise"])),
                self.cell_state_lstm_grid: np.random.normal(self.test_cell_state_current_lstm_grid.copy(), kwargs["grid_noise"]),
                self.hidden_state_lstm_grid: np.random.normal(self.test_hidden_state_current_lstm_grid.copy(), kwargs["grid_noise"]),
                self.cell_state_lstm_ped: np.random.normal(self.test_cell_state_current_lstm_ped.copy(), kwargs["ped_noise"]),
                self.hidden_state_lstm_ped: np.random.normal(self.test_hidden_state_current_lstm_ped.copy(), kwargs["ped_noise"]),
                self.cell_state_lstm_concat: self.test_cell_state_current_lstm_concat,
                self.hidden_state_lstm_concat: self.test_hidden_state_current_lstm_concat,
                }

    def reset_cells(self, sequence_reset):
        # Initialize the new sequences with a hidden state of zeros, the continuing sequences get assigned the previous hidden state
        if np.any(sequence_reset):
            for sequence_idx in range(sequence_reset.shape[0]):
                if sequence_reset[sequence_idx] == 1:
                    # Hidden state of specific batch entry will be initialized with zeros if there was a jump in sequence
                    self.cell_state_current[sequence_idx, :] = np.zeros([1, self.rnn_state_size])
                    self.hidden_state_current[sequence_idx, :] = np.zeros([1, self.rnn_state_size])
                    self.cell_state_current_lstm_grid[sequence_idx, :] = np.zeros([1, self.rnn_state_size_lstm_grid])
                    self.hidden_state_current_lstm_grid[sequence_idx, :] = np.zeros([1, self.rnn_state_size_lstm_grid])
                    self.cell_state_current_lstm_ped[sequence_idx, :] = np.zeros([1, self.rnn_state_size_lstm_ped])
                    self.hidden_state_current_lstm_ped[sequence_idx, :] = np.zeros([1, self.rnn_state_size_lstm_ped])
                    self.cell_state_current_lstm_concat[sequence_idx, :] = np.zeros([1, self.rnn_state_size_lstm_concat])
                    self.hidden_state_current_lstm_concat[sequence_idx, :] = np.zeros([1, self.rnn_state_size_lstm_concat])


    def reset_test_cells(self, sequence_reset):
        # Initialize the new sequences with a hidden state of zeros, the continuing sequences get assigned the previous hidden state
        if np.any(sequence_reset):
            for sequence_idx in range(sequence_reset.shape[0]):
                if sequence_reset[sequence_idx] == 1:
                    # Hidden state of specific batch entry will be initialized with zeros if there was a jump in sequence
                    self.test_cell_state_current[sequence_idx, :] = np.zeros([1, self.rnn_state_size])
                    self.test_hidden_state_current[sequence_idx, :] = np.zeros([1, self.rnn_state_size])
                    self.test_cell_state_current_lstm_grid[sequence_idx, :] = np.zeros([1, self.rnn_state_size_lstm_grid])
                    self.test_hidden_state_current_lstm_grid[sequence_idx, :] = np.zeros([1, self.rnn_state_size_lstm_grid])
                    self.test_cell_state_current_lstm_ped[sequence_idx, :] = np.zeros([1, self.rnn_state_size_lstm_ped])
                    self.test_hidden_state_current_lstm_ped[sequence_idx, :] = np.zeros([1, self.rnn_state_size_lstm_ped])
                    self.test_cell_state_current_lstm_concat[sequence_idx, :] = np.zeros([1, self.rnn_state_size_lstm_concat])
                    self.test_hidden_state_current_lstm_concat[sequence_idx, :] = np.zeros([1, self.rnn_state_size_lstm_concat])

    def update_test_hidden_state(self):
        self.test_cell_state_current = self.cell_state_current.copy()
        self.test_hidden_state_current = self.hidden_state_current.copy()
        self.test_cell_state_current_lstm_grid = self.cell_state_current_lstm_grid.copy()
        self.test_hidden_state_current_lstm_grid = self.hidden_state_current_lstm_grid.copy()
        self.test_cell_state_current_lstm_ped = self.cell_state_current_lstm_ped.copy()
        self.test_hidden_state_current_lstm_ped = self.hidden_state_current_lstm_ped.copy()
        self.test_cell_state_current_lstm_concat = self.cell_state_current_lstm_concat.copy()
        self.test_hidden_state_current_lstm_concat = self.hidden_state_current_lstm_concat.copy()

    def validation_step(self, sess, feed_dict_validation, update=True):
        batch_loss, _model_prediction, _current_state, _current_state_lstm_grid, _current_state_lstm_ped, \
        _current_state_lstm_concat, summary = sess.run([self.reconstruction_loss,
                                                        self.prediction,
                                                        self.current_state,
                                                        self.current_state_lstm_grid,
                                                        self.current_state_lstm_ped,
                                                        self.current_state_lstm_concat,
                                                        self.loss_summary],
                                                       feed_dict=feed_dict_validation)

        if update:
            self.test_cell_state_current, self.test_hidden_state_current = _current_state
            self.test_cell_state_current_lstm_grid, self.test_hidden_state_current_lstm_grid = _current_state_lstm_grid
            self.test_cell_state_current_lstm_ped, self.test_hidden_state_current_lstm_ped = _current_state_lstm_ped
            self.test_cell_state_current_lstm_concat, self.test_hidden_state_current_lstm_concat = _current_state_lstm_concat

        return batch_loss, summary, _model_prediction

    def train_step(self, sess, feed_dict_train, step=0):

        # Diversity Training
        if step > 2000 and self.args.diversity_update:
            dict = {"batch_vel": feed_dict_train[self.input_state_placeholder],
                    "batch_grid": feed_dict_train[self.input_grid_placeholder],
                    "batch_ped_grid": feed_dict_train[self.input_ped_grid_placeholder],
                    "batch_y": feed_dict_train[self.output_placeholder],
                    "batch_div": feed_dict_train[self.diversity_placeholder],
                    "step": step,
                    "state_noise": 0.0,
                    "grid_noise": 0.0,
                    "ped_noise": 0.0
                    }
            self.update_test_hidden_state()
            feed_test_dic = self.feed_val_dic(n_other_agent=feed_dict_train[self.seq_length],**dict)
            loss, _, y_model_pred = self.validation_step(sess, feed_test_dic, False)
            batch_y_varible = np.zeros((self.args.batch_size, self.args.truncated_backprop_length,
                                        self.args.prediction_horizon * self.args.output_dim))
            # Grount truth
            batch_y = feed_dict_train[self.output_placeholder]

            # Find most different trajectory index
            for batch_id in range(self.args.batch_size):
                for t_id in range(self.args.truncated_backprop_length):
                    diversity_distance = 0
                    diff_traj_index = 0
                    for mix_id in range(self.args.n_mixtures):
                        for pred_id in range(self.args.prediction_horizon):
                            new_batch_x = y_model_pred[batch_id,t_id,pred_id * self.args.output_pred_state_dim * self.args.n_mixtures+ mix_id]
                            new_batch_y = y_model_pred[batch_id,t_id,pred_id * self.args.output_pred_state_dim * self.args.n_mixtures + self.args.n_mixtures + mix_id]
                            batch_y_varible[batch_id, t_id, 2 * pred_id:2 * pred_id + self.args.output_dim] = np.array(
                                [new_batch_x, new_batch_y])
                        if diversity_distance < np.linalg.norm(batch_y_varible[batch_id, t_id]-batch_y[batch_id, t_id]):
                            diversity_distance = np.linalg.norm(batch_y_varible[batch_id, t_id] - batch_y[batch_id, t_id])
                            diff_traj_index = mix_id
                    # Saving most different traj
                    for pred_id in range(self.args.prediction_horizon):
                        new_batch_x = y_model_pred[batch_id,t_id,pred_id * self.args.output_pred_state_dim * self.args.n_mixtures + diff_traj_index]
                        new_batch_y = y_model_pred[batch_id,t_id,pred_id * self.args.output_pred_state_dim * self.args.n_mixtures + self.args.n_mixtures + diff_traj_index]
                        batch_y_varible[batch_id, t_id, 2 * pred_id:2 * pred_id + self.args.output_dim] = np.array(
                            [new_batch_x, new_batch_y])

            # Assemble feed dict for training
            feed_dict_train[self.diversity_placeholder] = batch_y_varible

        _, _, batch_loss, kl_loss, recons_loss, lh_loss, _current_state, _current_state_lstm_grid, _current_state_lstm_ped, \
        _current_state_lstm_concat, \
        _model_prediction, _summary_str, lr, beta, output_decoder, div_loss, autoencoder_loss = sess.run(
            [self.update,
             self.likelihood_update,
             self.total_loss,
             self.kl_loss,
             self.reconstruction_loss,
             self.likelihood_loss,
             self.current_state,
             self.current_state_lstm_grid,
             self.current_state_lstm_ped,
             self.current_state_lstm_concat,
             self.prediction,
             self.summary,
             self.learning_rate,
             self.beta,
             self.deconv1,
             self.div_loss,
             self.autoencoder_loss],
            feed_dict=feed_dict_train)
        self.cell_state_current, self.hidden_state_current = _current_state
        self.cell_state_current_lstm_grid, self.hidden_state_current_lstm_grid = _current_state_lstm_grid
        self.cell_state_current_lstm_ped, self.hidden_state_current_lstm_ped = _current_state_lstm_ped
        self.cell_state_current_lstm_concat, self.hidden_state_current_lstm_concat = _current_state_lstm_concat

        # Train autoencoder if loss larger than threshold
        """"""
        print("Autoencoder loss: " + str(np.mean(autoencoder_loss)))
        if np.mean(autoencoder_loss) > 0.05:
            self.run_autoencoder(sess, feed_dict_train)
            self.convnet_saver.save(sess, self.args.pretrained_convnet_path + '/final-model.ckpt')

        out = {"batch_loss": recons_loss,
               "likelihood loss": lh_loss,
               "model_predictions": _model_prediction,
               "summary": _summary_str,
               "learning_rate": lr,
               "autoencoder_loss": autoencoder_loss,
               "log loss": batch_loss,
               "kl_loss": kl_loss,
               "output_decoder": output_decoder
               }

        return out

    def run_autoencoder(self, sess, feed_dict_train):
        _ = sess.run([self.autoencoder_update],
                     feed_dict=feed_dict_train)

    def predict(self, sess, feed_dict_train, update_state=True):
        _current_state, _current_state_lstm_grid, _current_state_lstm_ped, \
        _current_state_lstm_concat, \
        _model_prediction, output_decoder, _likelihood = sess.run([self.current_state,
                                                                   self.current_state_lstm_grid,
                                                                   self.current_state_lstm_ped,
                                                                   self.current_state_lstm_concat,
                                                                   self.prediction,
                                                                   self.deconv1,
                                                                   self.likelihood],
                                                                  feed_dict=feed_dict_train)
        if update_state:
            self.test_cell_state_current, self.test_hidden_state_current = _current_state
            self.test_cell_state_current_lstm_grid, self.test_hidden_state_current_lstm_grid = _current_state_lstm_grid
            self.test_cell_state_current_lstm_ped, self.test_hidden_state_current_lstm_ped = _current_state_lstm_ped
            self.test_cell_state_current_lstm_concat, self.test_hidden_state_current_lstm_concat = _current_state_lstm_concat

        return _model_prediction, _likelihood

    def warmstart_model(self, args, sess):
        # Restore whole model
        try:
            print('Loading session from "{}"'.format(args.model_path))
            ckpt = tf.train.get_checkpoint_state(args.model_path)
            print('Restoring model {}'.format(ckpt.model_checkpoint_path))
            self.full_saver.restore(sess, ckpt.model_checkpoint_path)
        except:
            print(Fore.RED + "Model not found..." + Style.RESET_ALL)
            sys.exit()

    def warmstart_model_encoder(self, args, sess):
        # Restore whole model
        print('Loading session from "{}"'.format(args.pretrained_encoder_path))
        ckpt = tf.train.get_checkpoint_state(args.pretrained_encoder_path)
        if ckpt == None:
            print(Fore.RED + "Error. Encoder not found...")
            sys.exit()
        else:
            print('Restoring model {}'.format(ckpt.model_checkpoint_path))
            self.encoder_saver.restore(sess, ckpt.model_checkpoint_path)

        return True

    def warmstart_convnet(self, args, sess):
        # Restore convnet parameters from pretrained ones (autoencoder encoding part)
        print('Loading convnet parameters from "{}"'.format(args.pretrained_convnet_path))
        ckpt_conv = tf.train.get_checkpoint_state(args.pretrained_convnet_path)
        if ckpt_conv == None:
            print("Error. Convnet not found...")
            exit()
        else:
            print('Restoring convnet {}'.format(ckpt_conv.model_checkpoint_path))
            self.convnet_saver.restore(sess, ckpt_conv.model_checkpoint_path)


    def get_diversity_loss(self, mux, muy, sigmax, sigmay):
        # It only works for n_mixtures =3
        div_loss = 0
        for mix_id in range(self.args.n_mixtures):
            mu1 = tf.concat([tf.expand_dims(mux[:, mix_id], axis=1), tf.expand_dims(muy[:, mix_id], axis=1)], 1)
            sigma1 = tf.concat([tf.expand_dims(sigmax[:, mix_id], axis=1), tf.expand_dims(sigmay[:, mix_id], axis=1)], 1)
            gaussian1 = tfd.MultivariateNormalDiag(
                loc=mu1,
                scale_diag=sigma1)
            if mix_id + 1 < self.args.n_mixtures:
                mu2 = tf.concat([tf.expand_dims(mux[:, mix_id + 1], axis=1), tf.expand_dims(muy[:, mix_id + 1], axis=1)], 1)
                sigma2 = tf.concat([tf.expand_dims(sigmax[:, mix_id + 1], axis=1), tf.expand_dims(sigmay[:, mix_id + 1], axis=1)],
                                   1)
                gaussian2 = tfd.MultivariateNormalDiag(
                    loc=mu2,
                    scale_diag=sigma2)
            else:
                mu2 = tf.concat([tf.expand_dims(mux[:, -1], axis=1), tf.expand_dims(muy[:, -1], axis=1)], 1)
                sigma2 = tf.concat([tf.expand_dims(sigmax[:, mix_id], axis=1), tf.expand_dims(sigmay[:, mix_id], axis=1)], 1)
                gaussian2 = tfd.MultivariateNormalDiag(
                    loc=mu2,
                    scale_diag=sigma2)

            div_loss += tfd.kl_divergence(gaussian1, gaussian2)

            return div_loss

    def tf_2d_normal(self, x, y, mux, muy, sx, sy):
        '''
        Function that implements the PDF of a 2D normal distribution
        params:
        x : input x points
        y : input y points
        mux : mean of the distribution in x
        muy : mean of the distribution in y
        sx : std dev of the distribution in x
        sy : std dev of the distribution in y
        rho : Correlation factor of the distribution
        '''
        # eq 3 in the paper
        # and eq 24 & 25 in Graves (2013)
        # Calculate (x - mux) and (y-muy)
        rho = tf.constant(0.0)  # hack because we dont want correlation now
        normx = tf.subtract(x, mux)
        normy = tf.subtract(y, muy)
        # Calculate sx*sy
        sxsy = tf.multiply(sx, sy)
        # Calculate the exponential factor
        z = tf.square(tf.div(normx, sx)) + tf.square(tf.div(normy, sy)) - 2 * tf.div(
            tf.multiply(rho, tf.multiply(normx, normy)),
            sxsy)
        negRho = 1 - tf.square(rho)
        # Numerator
        result = tf.exp(tf.div(-z, 2 * negRho))
        # Normalization constant
        denom = 2 * np.pi * tf.multiply(sxsy, tf.sqrt(negRho))
        # Final PDF calculation
        result = tf.div(result, denom)
        return result

    def tf_kl_gaussgauss(self, mu_1, sigma_1, mu_2, sigma_2):
        with tf.variable_scope("kl_gaussgauss"):
            # return tf.reduce_sum(0.5 * (
            #    2 * tf.math.log(tf.maximum(1e-9, sigma_2), name='log_sigma_2')
            #    - 2 * tf.math.log(tf.maximum(1e-9, sigma_1), name='log_sigma_1')
            #    + (tf.square(sigma_1) + tf.square(mu_1 - mu_2)) / tf.maximum(1e-9, (tf.square(sigma_2))) - 1
            # ), 1)
            # return tf.reduce_sum(0.5 * (
            #    1.0 * tf.math.log(tf.maximum(1e-9, sigma_2), name='log_sigma_2')
            #    - 1.0 * tf.math.log(tf.maximum(1e-9, sigma_1), name='log_sigma_1')
            #    + (tf.square(sigma_1) + tf.square(mu_2 - mu_1)) / tf.maximum(1e-9, tf.square(sigma_2)) - 1.0
            # ), 1)
            kl = tfd.MultivariateNormalDiag(loc=mu_1, scale_diag=sigma_1).kl_divergence(
                tfd.MultivariateNormalDiag(loc=mu_2, scale_diag=sigma_2))
            return tf.reduce_mean(kl)

    def get_log_lossfunc(self, dec_mux, dec_muy, dec_sigmax, dec_sigmay, x, y):

        # gaussian = self.tf_2d_normal(x, y, dec_mux, dec_muy, dec_sigmax, dec_sigmay)
        # gaussian = tf.reduce_sum(gaussian, axis=1)  # do inner summation

        normx = tf.subtract(x, dec_mux)
        normy = tf.subtract(y, dec_muy)
        # Calculate sx*sy
        sxsy = tf.multiply(dec_sigmax, dec_sigmay)
        # Calculate the exponential factor
        z = tf.square(tf.div(normx, dec_sigmax)) + tf.square(tf.div(normy, dec_sigmay))

        loss = z + tf.log(sxsy)

        return tf.reduce_mean(loss)

    def get_log_mixlossfunc(self, dec_mux, dec_muy, dec_sigmax, dec_sigmay, x, y, pi):

        # gaussian = self.tf_2d_normal(x, y, dec_mux, dec_muy, dec_sigmax, dec_sigmay)
        # gaussian = tf.reduce_sum(gaussian, axis=1)  # do inner summation

        normx = tf.subtract(x, dec_mux)
        normy = tf.subtract(y, dec_muy)
        # Calculate sx*sy
        sxsy = tf.multiply(dec_sigmax, dec_sigmay)
        # Calculate the exponential factor
        z = tf.square(tf.div(normx, dec_sigmax)) + tf.square(tf.div(normy, dec_sigmay))

        loss = tf.multiply(z + tf.log(sxsy), pi)
        # loss = z + tf.log(sxsy)

        return tf.reduce_mean(loss)


    def get_mixture_lossfunc(self, dec_mux, dec_muy, dec_sigmax, dec_sigmay, dec_pi, x, y):
        gaussian = self.tf_2d_normal(x, y, dec_mux, dec_muy, dec_sigmax, dec_sigmay)
        gaussian = tf.multiply(gaussian, dec_pi)  # does element-wise multiplication
        gaussian = tf.reduce_sum(gaussian, axis=1)  # do inner summation

        likelihood_loss = -tf.math.log(tf.maximum(gaussian, 1e-10))
        return tf.reduce_mean(likelihood_loss)  #

    def process_grid(self, occ_grid_series):
        """
        Process occupancy grid with series of convolutional and fc layers.
        input: occupancy grid series (shape: tbpl * [batch_size, grid_dim_x, grid_dim_y])
        output: convolutional feature vector (shape: list of tbpl elements with [batch_size, feature_vector_size] each)
        """
        # Convolutional layer 1
        conv1_kernel_size = 5
        conv1_number_filters = 64
        conv1_stride_length = 2

        # Convolutional layer 2
        conv2_kernel_size = 3
        conv2_number_filters = 32
        conv2_stride_length = 2

        # Convolutional layer 3
        conv3_kernel_size = 3
        conv3_number_filters = 8
        conv3_stride_length = 2

        # Flatten convolutional filter vector such that it can be fed to a FC layer
        conv_grid_size = 8
        fc_grid_hidden_dim = 64

        conv_feature_vectors = []
        print("Length occ grid series: {}".format(len(occ_grid_series)))
        for idx, grid_batch in enumerate(occ_grid_series):
            grid_batch = tf.expand_dims(input=grid_batch, axis=3)

            if idx == 0:
                with tf.variable_scope('auto_encoder') as scope:
                    conv1 = tf.contrib.layers.conv2d(grid_batch, conv1_number_filters, [conv1_kernel_size, conv1_kernel_size],
                                                     conv1_stride_length, trainable=True)
                    conv2 = tf.contrib.layers.conv2d(conv1, conv2_number_filters, [conv2_kernel_size, conv2_kernel_size],
                                                     conv2_stride_length, trainable=True)
                    conv3 = tf.contrib.layers.conv2d(conv2, conv3_number_filters, [conv3_kernel_size, conv3_kernel_size],
                                                     conv3_stride_length, padding='VALID', trainable=False)

                    conv3_flat = tf.reshape(conv3, [-1, (conv_grid_size - 1) * (conv_grid_size - 1) * conv3_number_filters])

                    fc_grid_weights = self.get_weight_variable(shape=[conv3_flat.get_shape()[1], fc_grid_hidden_dim],
                                                               name="fc_grid_weights",
                                                               trainable=True)
                    fc_grid_biases = self.get_bias_variable(shape=[fc_grid_hidden_dim], name="fc_grid_biases",
                                                            trainable=True)

                    fc_final = self.fc_layer(conv3_flat, fc_grid_weights, fc_grid_biases)

                    tf.summary.histogram("fc_activations", fc_final)
                    fc_grid_dec_biases = self.get_bias_variable(shape=[conv3_flat.get_shape()[1]],
                                                                name="fc_grid_decode_biases", trainable=True)
                    fc_grid_dec = self.fc_layer(fc_final, tf.transpose(fc_grid_weights), fc_grid_dec_biases)
                    print("Dim out fc_grid_dec (conv2_flat_out): {}".format(fc_grid_dec.get_shape()))

                    # Reshaping into known convolutional grid size
                    conv3_dec = tf.reshape(fc_grid_dec,
                                           shape=[-1, conv_grid_size - 1, conv_grid_size - 1, conv3_number_filters])
                    print("Dim out conv3 reshape (conv3_out): {}".format(conv3_dec.get_shape()))

                    # From output of conv3 to input of conv3 (output of conv2)
                    deconv3 = tf.contrib.layers.conv2d_transpose(conv3_dec, conv2_number_filters,
                                                                 [conv3_kernel_size, conv3_kernel_size], conv3_stride_length,
                                                                 padding='VALID', trainable=True)
                    print("Dim out deconv3 (conv2_out): {}".format(deconv3.get_shape()))

                    # From output of conv2 to input of conv2 (output of conv1)
                    deconv2 = tf.contrib.layers.conv2d_transpose(deconv3, conv1_number_filters,
                                                                 [conv2_kernel_size, conv2_kernel_size], conv2_stride_length,
                                                                 padding='SAME', trainable=True)
                    print("Dim out deconv2 (conv1_out): {}".format(deconv2.get_shape()))

                    # From output of conv1 to input of conv1 (raw grid input)
                    self.deconv1 = tf.contrib.layers.conv2d_transpose(deconv2, 1, [conv1_kernel_size, conv1_kernel_size],
                                                                      conv1_stride_length, padding='SAME', trainable=True)
                    print("Dim out deconv1 (image input): {}".format(self.deconv1.get_shape()))

                    self.autoencoder_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='auto_encoder')

                    self.autoencoder_loss = tf.reduce_mean(
                        tf.square(tf.abs(grid_batch - self.deconv1)), name="autoencoder_loss")


            else:
                conv1 = tf.contrib.layers.conv2d(grid_batch, conv1_number_filters, [conv1_kernel_size, conv1_kernel_size],
                                                 conv1_stride_length, trainable=True, reuse=True, scope='auto_encoder/Conv')
                conv2 = tf.contrib.layers.conv2d(conv1, conv2_number_filters, [conv2_kernel_size, conv2_kernel_size],
                                                 conv2_stride_length, trainable=True, reuse=True, scope='auto_encoder/Conv_1')
                conv3 = tf.contrib.layers.conv2d(conv2, conv3_number_filters, [conv3_kernel_size, conv3_kernel_size],
                                                 conv3_stride_length, padding='VALID', trainable=True, reuse=True,
                                                 scope='auto_encoder/Conv_2')

                conv3_flat = tf.reshape(conv3, [-1, (conv_grid_size - 1) * (conv_grid_size - 1) * conv3_number_filters])
                fc_final = self.fc_layer(input=conv3_flat, weights=fc_grid_weights, biases=fc_grid_biases)

            # Flatten to obtain feature vector
            conv_features = tf.contrib.layers.flatten(fc_final)
            conv_feature_vectors.append(conv_features)

        return conv_feature_vectors


    def process_pedestrian_grid(self, ped_grid_series):
        """
        Process pedestrian grid with a FC layer
        """
        fc_feature_vector = []
        for idx, grid_batch in enumerate(ped_grid_series):
            grid_batch = tf.contrib.layers.flatten(grid_batch)
            # fc_feature_vector.append(self.fc_layer(grid_batch, weights=self.W_pedestrian_grid, biases=self.b_pedestrian_grid,activation = tf.nn.relu, name="fc_ped"))
            fc_feature_vector.append(
                self.fc_layer(grid_batch, weights=self.W_pedestrian_grid, biases=self.b_pedestrian_grid, name="fc_ped"))
            if idx == 0:
                tf.summary.histogram("fc_ped_activations", fc_feature_vector)
        #       fc_feature_vector.append(grid_batch)
        return fc_feature_vector


    def get_weight_variable(self, shape, name, trainable=True, regularizer=None, summary=True):
        """
        Get weight variable with specific initializer.
        """
        # var = tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(),
        #                      regularizer=tf.contrib.layers.l2_regularizer(0.01), trainable=trainable)
        # var = tf.get_variable(name=name, shape=shape, initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
        #                      regularizer=tf.contrib.layers.l2_regularizer(0.01), trainable=trainable)
        var = tf.get_variable(name=name, shape=shape,
                              initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN',
                                                                                         uniform=False, seed=None,
                                                                                         dtype=tf.float32),
                              regularizer=tf.contrib.layers.l2_regularizer(0.01), trainable=trainable)
        if summary:
            tf.summary.histogram(name, var)
        return var


    def get_bias_variable(self, shape, name, trainable=True, regularizer=None, summary=True):
        """
        Get bias variable with specific initializer.
        """
        # var = tf.get_variable(name=name, shape=shape, initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1), trainable=trainable)
        # var = tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(),
        #                      trainable=trainable)
        var = tf.get_variable(name=name, shape=shape,
                              initializer=tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN',
                                                                                         uniform=False, seed=None,
                                                                                         dtype=tf.float32),
                              trainable=trainable)
        if summary:
            tf.summary.histogram(name, var)
        return var


    def get_bayesian_weight_variable(self, shape, name, trainable=True, regularizer=None, summary=True):
        """
        Get weight variable with specific initializer.
        """
        var = tf.Variable(tf.truncated_normal(shape, stddev=0.1), dtype=tf.float32, name=name)
        if summary:
            tf.summary.histogram(name, var)
        return var


    def get_bayesian_bias_variable(self, shape, name, trainable=True, regularizer=None, summary=True):
        """
        Get bias variable with specific initializer.
        """
        var = tf.Variable(tf.truncated_normal(shape, mean=0.1, stddev=0.1), dtype=tf.float32, name=name)
        if summary:
            tf.summary.histogram(name, var)
        return var


    def conv_layer(self, input, weights, biases, conv_stride_length=1, padding="SAME", name="conv", summary=False):
        """
        Convolutional layer including a ReLU activation but excluding pooling.
        """
        conv = tf.nn.conv2d(input, filter=weights, strides=[1, conv_stride_length, conv_stride_length, 1], padding=padding,
                            name=name)
        activations = tf.nn.relu(conv + biases)
        if summary:
            tf.summary.histogram(name, activations)
        return activations


    def deconv_layer(self, input, weights, biases, out_shape, conv_stride_length=1, padding="SAME"):
        #
        input_shape = input.get_shape()
        deconv = tf.nn.conv2d_transpose(input, weights, out_shape, [1, conv_stride_length, conv_stride_length, 1],
                                        padding=padding)
        activation = tf.nn.relu(deconv + biases)
        return activation

    def fc_layer(self, input, weights, biases, activation=None, name="fc", summary=False):
        """
        Fully connected layer with given weights and biases.
        Activation and summary can be activated with the arguments.
        """
        affine_result = tf.matmul(input, weights) + biases
        if activation is None:
            activations = affine_result
        else:
            activations = activation(affine_result)
        if summary:
            tf.summary.histogram(name + "_activations", activations)
        return activations
