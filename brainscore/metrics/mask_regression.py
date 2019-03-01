import os
import warnings

from numpy import concatenate
from numpy import linalg as la
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import scale

from brainio_base.stimuli import StimulusSet
from brainscore.metrics.regression import pls_regression


class RegressionScore:
    def __init__(self, _model_name, _layer, _assembly, _im_dir='images', _nc_file=None, _wrapper=None):
        self.modelname = _model_name
        self.layer = _layer
        self.identifier = self.modelname + '_' + self.layer

        self.testing_stim_dir = 'TODO'
        self.assembly = _assembly

        self.extractor = _wrapper
        # for testing:
        self.extractor.identifier = self.identifier + '_kernel_PCA'

        self.regressor = None

    def __call__(self, im_path_offset=0):
        # split test/train
        train_index, val_index, resp_train, resp_val = self.split_data()
        self.train(train_index + im_path_offset, resp_train)
        predicted = self.predict(val_index + im_path_offset)
        # Scale
        scale(resp_val.values, copy=False)
        # only for vectors
        if resp_val.values.shape[0] == 1:
            resp_val.values = resp_val.values[0]
        return np.sqrt(mean_squared_error(predicted, resp_val.values))

    def train(self, train_idx, train_gt):
        scale(train_gt, copy=False)
        self.regressor = pls_regression()
        train_stim_set = self.create_stim_set(train_idx)
        # Testing activations:
        test_activations = self.extractor(train_stim_set, layers=[self.layer])

        self.regressor.fit(test_activations, train_gt)

    def predict(self, im_idx):
        validation_stim_set = self.create_stim_set(im_idx)
        # Validation activations:
        validation_activations = self.extractor(validation_stim_set, layers=[self.layer])

        return self.regressor.predict(validation_activations)

    def kfold_exp_var(self, n_folds=2, im_path_offset=0):
        kf = KFold(n_folds)
        results = None
        expected = None

        for i, (train_idx, test_idx) in enumerate(kf.split(self.assembly)):
            self.train(train_idx + im_path_offset, self.assembly[train_idx])
            predicted = self.predict(test_idx + im_path_offset)
            if results is None:
                results = predicted
                expected = self.assembly[test_idx].values
            else:
                results = np.concatenate((results, predicted))
                expected = np.concatenate((expected, self.assembly[test_idx].values))
        scores = np.array([pearsonr(results[:, i], expected[:, i])[0] for i in range(results.shape[-1])])
        return scores

    def split_data(self, test_size=0.20):
        image_index = np.array([i for i in range(self.assembly.shape[0])])
        im_train, im_val, resp_train, resp_val = train_test_split(image_index, self.assembly, test_size=test_size,
                                                                  random_state=123)
        return im_train, im_val, resp_train, resp_val

    def create_stim_set(self, im_idx):
        stim_paths = [os.path.dirname(__file__) + '/../../tests/test_metrics/image_{:05}.jpg'.format(i) for i in im_idx]
        basenames = [os.path.splitext(os.path.basename(path))[0] for path in stim_paths]
        image_ids = stim_paths
        s = StimulusSet({'image_file_path': stim_paths, 'image_file_name': basenames, 'image_id': image_ids})
        s.image_paths = {image_id: path for image_id, path in zip(image_ids, stim_paths)}
        s.name = None
        return s

    def generate_stim_paths(self, im_idx):
        stim_paths = [self.testing_stim_dir + '/image_{:05}.jpg'.format(i) for i in im_idx]
        return stim_paths


class CubeMapper(RegressionScore):
    def __init__(self, *args, **kwargs):
        super(CubeMapper, self).__init__(*args, **kwargs)
        #
        self.batch_data = BatchHook(self.extractor)
        self._is_trained = False
        self._initialized = False
        self._mapper_inits = None

    def init_mapper(self, **kwargs):
        if not self._mapper_inits:
            self._mapper_inits = kwargs
        self.regressor = Mapper(**kwargs)
        self._initialized = True
        self._is_trained = False

    def train(self, train_idx, train_gt):

        if not self._initialized:
            raise ValueError('cube map is not initialized')
        if self._is_trained:
            warnings.warn('Overwriting previous training')
            self.init_mapper(**self._mapper_inits)

        train_gt = train_gt.values
        # make sure batch array is empty
        self.batch_data.reset()

        train_stim_set = self.create_stim_set(train_idx)
        self.extractor(train_stim_set, layers=[self.layer])

        _batchArray = self.batch_data.batchArray

        self.regressor.fit(_batchArray, train_gt)
        self._is_trained = True

    def predict(self, im_idx):
        # make sure batch array is empty
        self.batch_data.reset()

        validation_stim_set = self.create_stim_set(im_idx)

        # Validation activations:
        self.extractor(validation_stim_set, layers=[self.layer])

        _batchArray = self.batch_data.batchArray
        return self.regressor.predict(_batchArray)


class BatchHook(object):
    def __init__(self, extractor):
        self.batchArray = None
        self.extractor = extractor
        self.hook()

    def reset(self):
        self.batchArray = None

    def __call__(self, batch_activations):
        activation = list(batch_activations.values())[0]
        if self.batchArray is None:
            self.batchArray = activation
        else:
            self.batchArray = concatenate((self.batchArray, activation))
        return batch_activations

    def hook(self):
        hook = self
        handle = self.extractor.register_batch_activations_hook(hook)
        hook.handle = handle
        return handle


import numpy as np
import tensorflow as tf

np.random.seed(123)


# TF implementation of RF limited Regression

class Mapper(object):
    def __init__(self, graph=None, num_neurons=65, batch_size=50, init_lr=0.01,
                 ls=0.05, ld=0.1, tol=1e-2, max_epochs=10, inits=None,
                 log_rate=100, decay_rate=200, gpu_options=None):
        """
        Mapping function class.
        :param graph: tensorflow graph to build the mapping function with
        :param num_neurons: number of neurons (response variable) to predict
        :param batch_size: batch size
        :param init_lr: initial learning rate
        :param ls: regularization coefficient for spatial parameters
        :param ld: regularization coefficient for depth parameters
        :param tol: tolerance - stops the optimization if reaches below tol
        :param max_epochs: maximum number of epochs to train
        :param inits: initial values for the mapping function parameters. A dictionary containing
                      any of the following keys ['s_w', 'd_w', 'bias']
        :param log_rate: rate of logging the loss values
        :param decay_rate: rate of decay for learning rate (#epochs)
        """
        self._ld = ld  # reg factor for depth conv
        self._ls = ls  # reg factor for spatial conv
        self._tol = tol
        self._batch_size = batch_size
        self._num_neurons = num_neurons
        self._lr = init_lr
        self._max_epochs = max_epochs
        self._inits = inits
        self._is_initialized = False
        self._log_rate = log_rate
        self._decay_rate = decay_rate
        self._gpu_options = gpu_options

        if graph is None:
            self._graph = tf.Graph()
        else:
            self._graph = graph

        with self._graph.as_default():
            self._lr_ph = tf.placeholder(dtype=tf.float32)
            self._opt = tf.train.AdamOptimizer(learning_rate=self._lr_ph)

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
        :param X: Input data, first dimension is examples
        :param Y: response values (neurons), first dimension is examples
        :return:
        """
        X = self.norm_batch_array(X)
        Y = scale(Y, copy=True)
        with self._graph.as_default():
            assert X.ndim == 4, 'Input matrix rank should be 4.'
            if self._is_initialized is False:
                self._init_mapper(X)

            for e in range(self._max_epochs):
                for counter, batch in enumerate(
                        self._iterate_minibatches(X, Y, batchsize=self._batch_size, shuffle=True)):
                    feed_dict = {self._input_ph: batch[0],
                                 self.target_ph: batch[1],
                                 self._lr_ph: self._lr}
                    _, loss_value, reg_loss_value = self._sess.run([self.train_op, self.l2_error, self.reg_loss],
                                                                   feed_dict=feed_dict)
                if e % self._log_rate == 0:
                    print('Epoch: %d, Err Loss: %.2f, Reg Loss: %.2f' % (e + 1, loss_value, reg_loss_value))
                if e % self._decay_rate == 0 and e != 0:
                    self._lr /= 10.
                if loss_value < self._tol:
                    print('Converged.')
                    break

    def predict(self, X):
        """
        Predicts the respnoses to the give input X
        :param X: Input data, first dimension is examples
        :return: predictions
        """
        X = self.norm_batch_array(X)
        with self._graph.as_default():
            if self._is_initialized is False:
                self._init_mapper(X)

            preds = []
            for batch in self._iterate_minibatches(X, batchsize=self._batch_size, shuffle=False):
                feed_dict = {self._input_ph: batch}
                preds.append(np.squeeze(self._sess.run([self._predictions], feed_dict=feed_dict)))
            return np.concatenate(preds, axis=0)

    def _make_separable_map(self):
        """
        Makes the mapping function computational graph
        :return:
        """
        with self._graph.as_default():
            with tf.variable_scope('mapping'):
                input_shape = self._input_ph.shape
                preds = []
                for n in range(self._num_neurons):
                    with tf.variable_scope('N_{}'.format(n)):
                        if self._inits is None:
                            s_w = tf.Variable(initial_value=np.random.randn(1, input_shape[1], input_shape[2], 1),
                                              dtype=tf.float32)
                            d_w = tf.Variable(initial_value=np.random.randn(1, 1, input_shape[-1], 1),
                                              dtype=tf.float32)
                            bias = tf.Variable(initial_value=np.zeros((1, 1, 1, 1)), dtype=tf.float32)
                        else:
                            if 's_w' in self._inits:
                                s_w = tf.Variable(initial_value=self._inits['s_w'][n].reshape(
                                    (1, input_shape[1], input_shape[2], 1)),
                                    dtype=tf.float32)
                            else:
                                s_w = tf.Variable(
                                    initial_value=np.random.randn(1, input_shape[1], input_shape[2], 1),
                                    dtype=tf.float32)
                            if 'd_w' in self._inits:
                                d_w = tf.Variable(
                                    initial_value=self._inits['d_w'][n].reshape(1, 1, input_shape[-1], 1),
                                    dtype=tf.float32)
                            else:
                                d_w = tf.Variable(initial_value=np.random.randn(1, 1, input_shape[-1], 1),
                                                  dtype=tf.float32)
                            if 'bias' in self._inits:
                                bias = tf.Variable(initial_value=self._inits['bias'][n].reshape(1, 1, 1, 1),
                                                   dtype=tf.float32)
                            else:
                                bias = tf.Variable(initial_value=np.zeros((1, 1, 1, 1)), dtype=tf.float32)

                        tf.add_to_collection('s_w', s_w)
                        out = s_w * self._input_ph

                        tf.add_to_collection('d_w', d_w)
                        out = tf.reduce_sum(out, axis=[1, 2], keepdims=True)
                        out = tf.nn.conv2d(out, d_w, [1, 1, 1, 1], 'SAME')

                        tf.add_to_collection('bias', bias)
                        preds.append(tf.squeeze(out, axis=[1, 2]) + bias)
                        # preds.append(tf.reduce_sum(out, axis=[1, 2]) + bias)

                self._predictions = tf.concat(preds, -1)

    def _make_loss(self):
        """
        Makes the loss computational graph
        :return:
        """
        with self._graph.as_default():
            with tf.variable_scope('loss'):
                self.l2_error = tf.norm(self._predictions - self.target_ph, ord=2)
                # For separable mapping
                self._s_vars = tf.get_collection('s_w')
                self._d_vars = tf.get_collection('d_w')
                self._biases = tf.get_collection('bias')

                # L1 reg
                # self.reg_loss = self.ls * tf.reduce_sum([tf.reduce_sum(tf.abs(t)) for t in self.s_vars]) + self.ld * tf.reduce_sum([tf.reduce_sum(tf.abs(t)) for t in self.d_vars])
                # L2 reg
                # self.reg_loss = self.ls * tf.reduce_sum([tf.reduce_sum(tf.pow(t, 2)) for t in self.s_vars]) + self.ld * tf.reduce_sum([tf.reduce_sum(tf.pow(t, 2)) for t in self.d_vars])
                #                 self.total_loss = self.l2_error + self.reg_loss

                # Laplacian loss
                laplace_filter = tf.constant(np.array([0, -1, 0, -1, 4, -1, 0, -1, 0]).reshape((3, 3, 1, 1)),
                                             dtype=tf.float32)
                laplace_loss = tf.reduce_sum(
                    [tf.norm(tf.nn.conv2d(t, laplace_filter, [1, 1, 1, 1], 'SAME')) for t in self._s_vars])
                l2_loss = tf.reduce_sum([tf.reduce_sum(tf.pow(t, 2)) for t in self._s_vars])
                self.reg_loss = self._ls * (l2_loss + laplace_loss) + \
                                self._ld * tf.reduce_sum([tf.reduce_sum(tf.pow(t, 2)) for t in self._d_vars])

                self.total_loss = self.l2_error + self.reg_loss
                self.tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                self.train_op = self._opt.minimize(self.total_loss, var_list=self.tvars,
                                                   global_step=tf.train.get_or_create_global_step())

    def _init_mapper(self, X):
        """
        Initializes the mapping function graph
        :param X: input data
        :return:
        """
        with self._graph.as_default():
            self._input_ph = tf.placeholder(dtype=tf.float32, shape=[None] + list(X.shape[1:]))
            self.target_ph = tf.placeholder(dtype=tf.float32, shape=[None, self._num_neurons])
            # Build the model graph
            self._make_separable_map()
            self._make_loss()
            self._is_initialized = True

            # initialize graph
            print('Initializing...')
            init_op = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
            if self._gpu_options is None:
                self._sess = tf.Session()
            else:
                self._sess = tf.Session(config=tf.ConfigProto(gpu_options=self._gpu_options))

            self._sess.run(init_op)

    @staticmethod
    def norm_batch_array(data):
        return data / la.norm(data, axis=-1, keepdims=True, ord=1)

    def close(self):
        """
        Closes occupied resources
        :return:
        """
        tf.reset_default_graph()
        self._sess.close()
