import logging

import numpy as np
from tqdm import tqdm
from typing import Union

from brainscore_vision.utils import fullname


# do not import tensorflow at top level to avoid forcing users to install it if they don't use this metric


class MaskRegression:
    """
    Klindt et. al, NIPS 2017
    https://papers.nips.cc/paper/6942-neural-system-identification-for-large-populations-separating-what-and-where

    TF implementation of Receptive Field factorized regression
    """

    def __init__(self, init_lr=0.01,
                 max_epochs=40, tol=0.1, batch_size=50, ls=.1, ld=.1, decay_rate=25,
                 inits: Union[None, dict] = None, log_rate=10, gpu_options=None):
        """
        mapping function class.
        :param batch_size: batch size
        :param init_lr: initial learning rate
        :param ls: regularization coefficient for spatial parameters (spatial convolution)
        :param ld: regularization coefficient for depth parameters (depth convolution)
        :param tol: tolerance - stops the optimization if reaches below tol
        :param max_epochs: maximum number of epochs to train
        :param inits: initial values for the mapping function parameters. A dictionary containing
                      any of the following keys ['s_w', 'd_w', 'bias']
        :param log_rate: rate of logging the loss values
        :param decay_rate: rate of decay for learning rate (#epochs)
        """
        self._ld = ld
        self._ls = ls
        self._tol = tol
        self._batch_size = batch_size
        self._lr = init_lr
        self._max_epochs = max_epochs
        self._inits = inits
        self._log_rate = log_rate
        self._decay_rate = decay_rate
        self._gpu_options = gpu_options

        self._graph = None
        self._lr_ph = None
        self._opt = None
        self._logger = logging.getLogger(fullname(self))

    def _iterate_minibatches(self, inputs, targets=None, batchsize=128, shuffle=False):
        """
        Iterates over inputs with minibatches
        :param inputs: input dataset, first dimension should be examples
        :param targets: [n_examples, n_neurons] response values, first dimension should be examples
        :param batchsize: batch size
        :param shuffle: flag indicating whether to shuffle the data while making minibatches
        :return: minibatch of (X, Y)
        """
        input_len = inputs.shape[0]
        if shuffle:
            indices = np.arange(input_len)
            np.random.shuffle(indices)
        for start_idx in range(0, input_len, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            if targets is None:
                yield inputs[excerpt]
            else:
                yield inputs[excerpt], targets[excerpt]

    def fit(self, X, Y):
        """
        Fits the parameters to the data
        :param X: Source data, first dimension is examples
        :param Y: Target data, first dimension is examples
        """
        assert not np.isnan(X).any() and not np.isnan(Y).any()
        self.setup()
        X = self.reindex(X)
        assert X.ndim == 4, 'Input matrix rank should be 4.'
        with self._graph.as_default():
            self._init_mapper(X, Y)
            lr = self._lr
            for epoch in tqdm(range(self._max_epochs), desc='mask epochs'):
                for counter, batch in enumerate(
                        self._iterate_minibatches(X, Y, batchsize=self._batch_size, shuffle=True)):
                    feed_dict = {self._input_placeholder: batch[0],
                                 self._target_placeholder: batch[1],
                                 self._lr_ph: lr}
                    _, loss_value, reg_loss_value = self._sess.run([self.train_op, self.l2_error, self.reg_loss],
                                                                   feed_dict=feed_dict)
                if epoch % self._log_rate == 0:
                    self._logger.debug(f'Epoch: {epoch}, Err Loss: {loss_value:.2f}, Reg Loss: {reg_loss_value:.2f}')
                if epoch % self._decay_rate == 0 and epoch != 0:
                    lr /= 10.
                if loss_value < self._tol:
                    self._logger.debug('Converged.')
                    break

    def predict(self, X):
        """
        Predicts the responses to the give input X
        :param X: Input data, first dimension is examples
        :return: predictions
        """
        assert not np.isnan(X).any()
        X = self.reindex(X)
        with self._graph.as_default():
            preds = []
            for batch in self._iterate_minibatches(X, batchsize=self._batch_size, shuffle=False):
                feed_dict = {self._input_placeholder: batch}
                preds.append(np.squeeze(self._sess.run([self._predictions], feed_dict=feed_dict)))
            return np.concatenate(preds, axis=0)

    def setup(self):
        import tensorflow as tf
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._lr_ph = tf.compat.v1.placeholder(dtype=tf.float32)
            self._opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self._lr_ph)

    def reindex(self, X):
        channel_names = ['channel', 'channel_x', 'channel_y']
        assert all(hasattr(X, coord) for coord in channel_names)
        shapes = [len(set(X[channel].values)) for channel in channel_names]
        X = np.reshape(X.values, [X.shape[0]] + shapes)
        X = np.transpose(X, axes=[0, 2, 3, 1])
        return X

    def _make_separable_map(self):
        """
        Makes the mapping function computational graph
        """
        import tensorflow as tf
        with self._graph.as_default():
            with tf.compat.v1.variable_scope('mapping'):
                input_shape = self._input_placeholder.shape
                preds = []
                for n in range(self._target_placeholder.shape[1]):
                    with tf.compat.v1.variable_scope('N_{}'.format(n)):
                        # for all variables, either use pre-defined initial value or initialize randomly
                        if self._inits is not None and 's_w' in self._inits:
                            s_w = tf.Variable(initial_value=
                                              self._inits['s_w'][n].reshape((1, input_shape[1], input_shape[2], 1)),
                                              dtype=tf.float32)
                        else:
                            s_w = tf.Variable(initial_value=np.random.randn(1, input_shape[1], input_shape[2], 1),
                                              dtype=tf.float32)
                        if self._inits is not None and 'd_w' in self._inits:
                            d_w = tf.Variable(initial_value=self._inits['d_w'][n].reshape(1, 1, input_shape[-1], 1),
                                              dtype=tf.float32)
                        else:
                            d_w = tf.Variable(initial_value=np.random.randn(1, 1, input_shape[-1], 1),
                                              dtype=tf.float32)
                        if self._inits is not None and 'bias' in self._inits:
                            bias = tf.Variable(initial_value=self._inits['bias'][n].reshape(1, 1, 1, 1),
                                               dtype=tf.float32)
                        else:
                            bias = tf.Variable(initial_value=np.zeros((1, 1, 1, 1)), dtype=tf.float32)

                        tf.compat.v1.add_to_collection('s_w', s_w)
                        out = s_w * self._input_placeholder

                        tf.compat.v1.add_to_collection('d_w', d_w)
                        out = tf.reduce_sum(input_tensor=out, axis=[1, 2], keepdims=True)
                        out = tf.nn.conv2d(input=out, filters=d_w, strides=[1, 1, 1, 1], padding='SAME')

                        tf.compat.v1.add_to_collection('bias', bias)
                        preds.append(tf.squeeze(out, axis=[1, 2]) + bias)

                self._predictions = tf.concat(preds, -1)

    def _make_loss(self):
        """
        Makes the loss computational graph
        """
        import tensorflow as tf
        with self._graph.as_default():
            with tf.compat.v1.variable_scope('loss'):
                self.l2_error = tf.norm(tensor=self._predictions - self._target_placeholder, ord=2)
                # For separable mapping
                self._s_vars = tf.compat.v1.get_collection('s_w')
                self._d_vars = tf.compat.v1.get_collection('d_w')
                self._biases = tf.compat.v1.get_collection('bias')

                # Laplacian loss
                laplace_filter = tf.constant(np.array([0, -1, 0, -1, 4, -1, 0, -1, 0]).reshape((3, 3, 1, 1)),
                                             dtype=tf.float32)
                laplace_loss = tf.reduce_sum(
                    input_tensor=[tf.norm(tensor=tf.nn.conv2d(input=t, filters=laplace_filter, strides=[1, 1, 1, 1], padding='SAME')) for t in self._s_vars])
                l2_loss = tf.reduce_sum(input_tensor=[tf.reduce_sum(input_tensor=tf.pow(t, 2)) for t in self._s_vars])
                self.reg_loss = self._ls * (l2_loss + laplace_loss) + \
                                self._ld * tf.reduce_sum(input_tensor=[tf.reduce_sum(input_tensor=tf.pow(t, 2)) for t in self._d_vars])

                self.total_loss = self.l2_error + self.reg_loss
                self.tvars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
                self.train_op = self._opt.minimize(self.total_loss, var_list=self.tvars,
                                                   global_step=tf.compat.v1.train.get_or_create_global_step())

    def _init_mapper(self, X, Y):
        """
        Initializes the mapping function graph
        :param X: input data
        """
        import tensorflow as tf
        assert len(Y.shape) == 2
        with self._graph.as_default():
            self._input_placeholder = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None] + list(X.shape[1:]))
            self._target_placeholder = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, Y.shape[1]])
            # Build the model graph
            self._make_separable_map()
            self._make_loss()

            # initialize graph
            self._logger.debug('Initializing mapper')
            init_op = tf.compat.v1.variables_initializer(tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES))
            self._sess = tf.compat.v1.Session(
                config=tf.compat.v1.ConfigProto(gpu_options=self._gpu_options) if self._gpu_options is not None else None)
            self._sess.run(init_op)

    def close(self):
        """
        Closes occupied resources
        """
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()
        self._sess.close()
