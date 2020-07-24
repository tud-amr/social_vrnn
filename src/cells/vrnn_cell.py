import tensorflow as tf
import sys
if sys.version_info[0] < 3:
  from tf_utils import *
else:
  from src.models.tf_utils import *

class VariationalRNNCell(tf.contrib.rnn.RNNCell):
    """Variational RNN cell."""

    def __init__(self,args, x_dim, y_dim,h_dim ,z_dim = 100):
        self.n_h = h_dim
        self.n_x = x_dim
        self.n_z = z_dim
        self.n_x_1 = x_dim
        self.n_z_1 = z_dim
        self.n_enc_hidden = z_dim
        self.n_dec_hidden = y_dim
        self.n_prior_hidden = z_dim
        self.lstm = tf.contrib.rnn.LSTMCell(self.n_x, state_is_tuple=True,reuse=tf.AUTO_REUSE)
        self.args = args

    @property
    def state_size(self):
        return (self.n_x, self.n_x)

    @property
    def output_size(self):
        return self.n_x

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            h, c = state

            #   Eq 5
            with tf.variable_scope("Prior"):
                with tf.variable_scope("hidden"):
                    prior_hidden = tf.nn.relu(linear(h, self.n_prior_hidden))
                    #prior_hidden = linear(h, self.n_prior_hidden)
                with tf.variable_scope("mu"):
                    prior_mu = linear(prior_hidden, self.n_z)
                with tf.variable_scope("sigma"):
                    prior_sigma = tf.exp(linear(prior_hidden, self.n_z))

            with tf.variable_scope("phi_x"):
                x_1 = tf.nn.relu(linear(x, self.n_x_1))
                #x_1 = linear(x, self.n_x_1)

            with tf.variable_scope("Encoder"):
                with tf.variable_scope("hidden"):
                    enc_hidden = tf.nn.relu(linear(tf.concat(axis=1,values=(x_1, h)), self.n_enc_hidden))
                    #enc_hidden = linear(tf.concat(axis=1, values=(x_1, h)), self.n_enc_hidden)
                with tf.variable_scope("mu"):
                    enc_mu    = linear(enc_hidden, self.n_z)
                with tf.variable_scope("sigma"):
                    enc_sigma = tf.exp(linear(enc_hidden, self.n_z))
            eps = tf.random.normal(tf.shape(enc_mu), prior_mu, prior_sigma, dtype=tf.float32)
            # z = mu + sigma*epsilon
            z = tf.add(enc_mu, tf.multiply(enc_sigma, eps))
            with tf.variable_scope("phi_z"):
                z_1 = tf.nn.relu(linear(z, self.n_z_1))

            # propagate hidden state eq(7)
            output, state2 = self.lstm(tf.concat(axis=1, values=(x_1, z_1)), state) # tf.concat(axis=1, values=(x_1, z_1)

        return (enc_mu, enc_sigma, output, prior_mu, prior_sigma), state2