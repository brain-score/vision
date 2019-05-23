import logging
import warnings
from collections import OrderedDict, defaultdict

import numpy as np
import scipy.stats
import sklearn.linear_model
import sklearn.multioutput
from numpy.random.mtrand import RandomState
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.stats import spearmanr
from tqdm import tqdm

from brainio_base.assemblies import walk_coords, array_is_element, BehavioralAssembly
from brainscore.metrics import Metric, Score
from brainscore.metrics.behavior import I1
from brainscore.metrics.transformations import CrossValidation
from brainscore.utils import fullname


class OSTCorrelation(Metric):
    def __init__(self):
        self._cross_validation = CrossValidation(stratification_coord=None, splits=10, test_size=0.1)
        self._i1 = I1()
        self._tracked_bins = defaultdict(list)
        self._predicted_osts, self._target_osts = [], []

    def __call__(self, source_recordings, target_osts):
        score = self._cross_validation(source_recordings, target_osts, apply=self.apply)
        correlation = score.sel(aggregation='center')
        t, p = scipy.stats.ttest_ind(score.raw.values, [0] * len(score.raw.values))

        num_bins = 5
        non_nan = np.logical_and(~np.isnan(self._predicted_osts), ~np.isnan(self._target_osts))
        self._predicted_osts, self._target_osts = self._predicted_osts[non_nan], self._target_osts[non_nan]
        min_x, max_x = self._predicted_osts.min(), self._predicted_osts.max()
        stepsize = (max_x - min_x) / num_bins
        bins = np.arange(min_x, max_x, stepsize)
        binned_values = OrderedDict()
        for bin1, bin2 in zip(bins, bins[1:].tolist() + [np.inf]):
            mask = np.array([bin1 <= x < bin2 for x in self._predicted_osts])
            y = self._target_osts[mask]
            binned_values[bin1 + stepsize] = y
        binned_x, binned_y = list(binned_values.keys()), list(binned_values.values())
        binned_y_means, binned_y_err = [np.mean(y) for y in binned_y], [scipy.stats.sem(y) for y in binned_y]
        self.plot(binned_x, binned_y_means, yerr=binned_y_err, correlation_p=(correlation.values.tolist(), p),
                  filename='osts-interpolation-binned-cv', plot_type='errorbar')

        # plot model bins
        binned_x, binned_y = list(self._tracked_bins.keys()), list(self._tracked_bins.values())
        binned_y_means, binned_y_err = [np.mean(y) for y in binned_y], [scipy.stats.sem(y) for y in binned_y]
        self.plot(binned_x, binned_y_means, yerr=binned_y_err, correlation_p=(correlation.values.tolist(), p),
                  filename='osts-binned-cv', plot_type='errorbar')

        return score

    def apply(self, train_source, train_osts, test_source, test_osts):
        predicted_osts = self.compute_osts(train_source, test_source, test_osts)
        self._predicted_osts = np.concatenate((self._predicted_osts, predicted_osts))
        self._target_osts = np.concatenate((self._target_osts, test_osts.values))
        score = self.correlate(predicted_osts, test_osts.values)
        return score

    def compute_osts(self, train_source, test_source, test_osts):
        last_osts, hit_osts = [None] * len(test_osts), [None] * len(test_osts)
        for time_bin_start in tqdm(sorted(set(train_source['time_bin_start'].values)), desc='time_bins'):
            time_train_source = train_source.sel(time_bin_start=time_bin_start).squeeze('time_bin_end')
            time_test_source = test_source.sel(time_bin_start=time_bin_start).squeeze('time_bin_end')
            time_train_source = time_train_source.transpose('presentation', 'neuroid')
            time_test_source = time_test_source.transpose('presentation', 'neuroid')

            try:
                classifier = TFProbabilitiesClassifier()
                use_tf_classifier = True
            except ModuleNotFoundError:  # tensorflow not installed
                warnings.warn("tensorflow not installed. "
                              "falling back to sklearn classifier which does not support GPU and gets worse results")
                classifier = ProbabilitiesClassifier()
                use_tf_classifier = False
            classifier.fit(time_train_source, time_train_source['image_label'])
            prediction_probabilities = classifier.predict_proba(time_test_source)
            if use_tf_classifier:
                classifier.close()
            source_i1 = self.i1(prediction_probabilities)
            assert all(source_i1['image_id'].values == test_osts['image_id'].values)
            for i, (image_source_i1, threshold_i1) in enumerate(zip(
                    source_i1.values, test_osts['i1'].values)):
                if hit_osts[i] is None:
                    if image_source_i1 < threshold_i1 and (last_osts[i] is None or last_osts[i][1] < image_source_i1):
                        last_osts[i] = time_bin_start, image_source_i1
                    if image_source_i1 >= threshold_i1:
                        hit_osts[i] = time_bin_start, image_source_i1
            if not any(hit_ost is None for hit_ost in hit_osts):
                break

        # save binned
        for source, target in zip(hit_osts, test_osts.values):
            if source is None or np.isnan(target):
                continue  # skip
            self._tracked_bins[source[0]].append(target)

        # return np.array([ost[0] if ost is not None else np.nan for ost in hit_osts])  # ignore interpolation

        # interpolate
        predicted_osts = np.empty(len(test_osts), dtype=np.float)
        predicted_osts[:] = np.nan
        for i, (last_ost, hit_ost) in enumerate(zip(last_osts, hit_osts)):
            if hit_ost is None:
                predicted_osts[i] = np.nan
                continue
            (hit_ost_time, hit_ost_i1) = hit_ost
            if last_ost is None:
                predicted_osts[i] = hit_ost_time
                continue
            (last_ost_time, last_ost_i1) = last_ost
            fit = interp1d([last_ost_time, hit_ost_time], [last_ost_i1, hit_ost_i1], fill_value='extrapolate')
            ost = fsolve(lambda xs: [fit(x) - test_osts['i1'].values[i] for x in xs], x0=hit_ost_time)[0]
            predicted_osts[i] = ost
        return predicted_osts

    def correlate(self, predicted_osts, target_osts):
        non_nan = np.logical_and(~np.isnan(predicted_osts), ~np.isnan(target_osts))
        predicted_osts, target_osts = predicted_osts[non_nan], target_osts[non_nan]

        # correlation, p = pearsonr(predicted_osts, target_osts)
        correlation, p = spearmanr(predicted_osts, target_osts)

        # plots
        # self.plot(predicted_osts, target_osts, correlation_p=(correlation, p))

        # binned_targets = defaultdict(list)
        # for source, target in zip(predicted_osts, target_osts):
        #     binned_targets[source].append(target)
        # # self.ttest(binned_targets)
        # binned_means = OrderedDict((source, np.mean(targets)) for source, targets in binned_targets.items())
        # for source_bin, mean in binned_means.items():
        #     self._tracked_bins[source_bin].append(mean)
        # binned_x, binned_y = list(binned_means.keys()), list(binned_means.values())
        # self.plot(binned_x, binned_y, yerr=[scipy.stats.sem(targets) for source, targets in binned_targets.items()],
        #           filename='osts-binned', plot_type='errorbar', correlation_p=(correlation, p))
        # self.plot(list(binned_targets.keys()), list(binned_targets.values()),
        #           filename='osts-violin', plot_type='violinplot',)#raw_stats_values=binned_targets)
        # predicted_osts, target_osts = binned_x, binned_y

        return Score(correlation)

    def i1(self, prediction_probabilities):
        response_matrix = self._i1.target_distractor_scores(prediction_probabilities)
        response_matrix = self._i1.dprimes(response_matrix)
        response_matrix = self._i1.collapse_distractors(response_matrix)
        return response_matrix

    def plot(self, x, y, yerr=None, filename='osts', plot_type='scatter',
             raw_stats_values=None, trend_line=True, correlation_p=None):
        x, y, yerr = np.array(x), np.array(y), np.array(yerr)
        import seaborn
        seaborn.set()
        seaborn.set_context('paper', font_scale=2)
        seaborn.set_style('whitegrid', {'axes.grid': False})
        from matplotlib import pyplot
        pyplot.figure()
        plot = getattr(pyplot, plot_type)
        if plot_type == 'errorbar':
            idx = x.argsort()
            plot(x[idx], y[idx], yerr=yerr[idx], markersize=7.5, elinewidth=.5, fmt='o', color='#808080')
        elif plot_type == 'boxplot':
            plot(y, positions=x)
        elif plot_type == 'violinplot':
            plot(y, positions=x, showmeans=True, widths=8)
        else:
            plot(x, y)

        if plot_type in ['bar', 'errorbar']:
            pyplot.ylim(min(y) - 10, pyplot.ylim()[1])

        if raw_stats_values:
            significant_differences = self.ttest(raw_stats_values)
            for x1, x2 in significant_differences:
                self.significance_bar(x1, x2, 165 + np.random.randint(-10, 10), '*')

        if trend_line:
            if isinstance(y, list) and isinstance(y[0], list):
                import itertools
                x = list(itertools.chain(*[[_x] * len(_y) for _x, _y in zip(x, y)]))
                y = list(itertools.chain(*y))
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            print("trend line", p)
            trend_x = list(sorted(set(x)))
            pyplot.plot(trend_x, p(trend_x), linestyle='dashed', color='#D4145A', linewidth=4)

        if correlation_p:
            correlation, p = correlation_p
            p_magnitude = np.round(np.log10(p))
            print(f"magnitude of {p} is {p_magnitude}")
            pyplot.text(pyplot.xlim()[0] + 10, pyplot.ylim()[1] - 10, f"r={correlation:.2f} (p<{10 ** p_magnitude})")

        pyplot.xlabel('$IT_{COR}$ object solution times')
        pyplot.ylabel('$IT_{monkey}$ object solution times')

        pyplot.tight_layout()
        seaborn.despine(right=True, top=True)
        target_path = f'/braintree/home/msch/{filename}'
        for extension in ['png', 'pdf', 'svg']:
            pyplot.savefig(target_path + "." + extension)
        print(f"saved to {target_path}")

    def significance_bar(self, start, end, height, displaystring, linewidth=1.2, markersize=8, boxpad=0.3, fontsize=15,
                         color='k'):
        from matplotlib import pyplot
        from matplotlib.markers import TICKDOWN
        # draw a line with downticks at the ends
        pyplot.plot([start, end], [height] * 2,
                    '-', color=color, lw=linewidth, marker=TICKDOWN, markeredgewidth=linewidth, markersize=markersize)
        # draw the text with a bounding box covering up the line
        pyplot.text(0.5 * (start + end), height, displaystring, ha='center', va='center',
                    bbox=dict(facecolor='1.', edgecolor='none', boxstyle='Square,pad=' + str(boxpad)), size=fontsize)

    def ttest(self, bin_values):
        significant_differences = []
        bins = list(sorted(bin_values))
        for bin1 in bins:
            bins2 = [b for b in bins if b > bin1]
            for bin2 in bins2:
                a, b = np.array(bin_values[bin1]), np.array(bin_values[bin2])
                t, p = scipy.stats.ttest_ind(a[~np.isnan(a)], b[~np.isnan(b)])
                significant = p < .05
                print(f"t-test: {bin1} {'?' if not significant else '<' if t < 0 else '>'} {bin2}, (t={t}, p={p})")
                if significant:
                    significant_differences.append((bin1, bin2))
        return significant_differences


class ProbabilitiesClassifier:
    def __init__(self, classifier_c=1e-3):
        self._classifier = sklearn.linear_model.LogisticRegression(
            multi_class='multinomial', solver='newton-cg', C=classifier_c)
        self._label_mapping = None
        self._scaler = None

    def fit(self, X, Y):
        self._scaler = sklearn.preprocessing.StandardScaler().fit(X)
        X = self._scaler.transform(X)
        Y, self._label_mapping = self.labels_to_indices(Y.values)
        self._classifier.fit(X, Y)
        return self

    def predict_proba(self, X):
        assert len(X.shape) == 2, "expected 2-dimensional input"
        scaled_X = self._scaler.transform(X)
        proba = self._classifier.predict_proba(scaled_X)
        # we take only the 0th dimension because the 1st dimension is just the features
        X_coords = {coord: (dims, value) for coord, dims, value in walk_coords(X)
                    if array_is_element(dims, X.dims[0])}
        proba = BehavioralAssembly(proba,
                                   coords={**X_coords, **{'choice': list(self._label_mapping.values())}},
                                   dims=[X.dims[0], 'choice'])
        return proba

    def labels_to_indices(self, labels):
        label2index = OrderedDict()
        indices = []
        for label in labels:
            if label not in label2index:
                label2index[label] = (max(label2index.values()) + 1) if len(label2index) > 0 else 0
            indices.append(label2index[label])
        index2label = OrderedDict((index, label) for label, index in label2index.items())
        return indices, index2label


class TFProbabilitiesClassifier:
    def __init__(self,
                 init_lr=1e-4,
                 max_epochs=40,
                 zscore_feats=True,
                 train_batch_size=64,
                 eval_batch_size=240,
                 activation=None,
                 fc_weight_decay=0.463,  # 1/(C_svc * num_objectome_imgs) = 1/(1e-3 * 2160),
                 # based on https://stats.stackexchange.com/questions/216095/how-does-alpha-relate-to-c-in-scikit-learns-sgdclassifier
                 fc_dropout=1.0,
                 tol=1e-4,
                 log_rate=5, gpu_options=None):
        """
        mapping function class.
        :param train_batch_size: train batch size
        :param eval_batch_size: prediction batch size
        :param activation: what activation to use if any
        :param init_lr: initial learning rate
        :param zscore_feats: whether to zscore model features
        :param tol: tolerance - stops the optimization if reaches below tol
        :param fc_weight_decay: regularization coefficient for fully connected layers (inverse of sklearn C)
        :params fc_dropout: dropout parameter for fc layers
        :param log_rate: rate of logging the loss values (in epochs)
        """
        self._train_batch_size = train_batch_size
        self._eval_batch_size = eval_batch_size
        self._zscore_feats = zscore_feats
        self._lr = init_lr
        self._tol = tol
        self._activation = activation
        self._fc_weight_decay = fc_weight_decay
        self._fc_dropout = fc_dropout
        self._max_epochs = max_epochs
        self._log_rate = log_rate
        self._gpu_options = gpu_options

        self._graph = None
        self._lr_ph = None
        self._opt = None
        self._scaler = None
        self._logger = logging.getLogger(fullname(self))

    def _iterate_minibatches(self, inputs, targets=None, batchsize=240, shuffle=False, random_state=None):
        """
        Iterates over inputs with minibatches
        :param inputs: input dataset, first dimension should be examples
        :param targets: [n_examples, ...] response values, first dimension should be examples
        :param batchsize: batch size
        :param shuffle: flag indicating whether to shuffle the data while making minibatches
        :return: minibatch of (X, Y)
        """
        input_len = inputs.shape[0]
        if shuffle:
            indices = np.arange(input_len)
            random_state.shuffle(indices)
        for start_idx in range(0, input_len, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            if targets is None:
                yield inputs[excerpt]
            else:
                yield inputs[excerpt], targets[excerpt]

    def setup(self):
        import tensorflow as tf
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._lr_ph = tf.placeholder(dtype=tf.float32)
            self._opt = tf.train.AdamOptimizer(learning_rate=self._lr_ph)

    def initializer(self, kind='xavier', *args, **kwargs):
        import tensorflow as tf
        if kind == 'xavier':
            init = tf.contrib.layers.xavier_initializer(*args, **kwargs, seed=0)
        else:
            init = getattr(tf, kind + '_initializer')(*args, **kwargs)
        return init

    def fc(self,
           inp,
           out_depth,
           kernel_init='xavier',
           kernel_init_kwargs=None,
           bias=1,
           weight_decay=None,
           activation=None,
           name='fc'):

        import tensorflow as tf
        if weight_decay is None:
            weight_decay = 0.
        # assert out_shape is not None
        if kernel_init_kwargs is None:
            kernel_init_kwargs = {}
        resh = inp
        assert (len(resh.get_shape().as_list()) == 2)
        in_depth = resh.get_shape().as_list()[-1]

        # weights
        init = self.initializer(kernel_init, **kernel_init_kwargs)
        kernel = tf.get_variable(initializer=init,
                                 shape=[in_depth, out_depth],
                                 dtype=tf.float32,
                                 regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 name='weights')
        init = self.initializer(kind='constant', value=bias)
        biases = tf.get_variable(initializer=init,
                                 shape=[out_depth],
                                 dtype=tf.float32,
                                 regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 name='bias')

        # ops
        fcm = tf.matmul(resh, kernel)
        output = tf.nn.bias_add(fcm, biases, name=name)

        if activation is not None:
            output = getattr(tf.nn, activation)(output, name=activation)
        return output

    def _make_behavioral_map(self):
        """
        Makes the temporal mapping function computational graph
        """
        import tensorflow as tf
        num_classes = len(self._label_mapping.keys())
        with self._graph.as_default():
            with tf.variable_scope('behavioral_mapping'):
                out = self._input_placeholder
                out = tf.nn.dropout(out, keep_prob=self._fc_keep_prob, name="dropout_out")
                pred = self.fc(out,
                               out_depth=num_classes,
                               activation=self._activation,
                               weight_decay=self._fc_weight_decay, name="out")

                self._predictions = pred

    def _make_loss(self):
        """
        Makes the loss computational graph
        """
        import tensorflow as tf
        with self._graph.as_default():
            with tf.variable_scope('loss'):
                logits = self._predictions

                self.classification_error = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self._target_placeholder))
                self.reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

                self.total_loss = self.classification_error + self.reg_loss
                self.tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                self.train_op = self._opt.minimize(self.total_loss, var_list=self.tvars,
                                                   global_step=tf.train.get_or_create_global_step())

    def _init_mapper(self, X, Y):
        """
        Initializes the mapping function graph
        :param X: input data
        """
        import tensorflow as tf
        assert len(Y.shape) == 1
        with self._graph.as_default():
            self._input_placeholder = tf.placeholder(dtype=tf.float32, shape=[None] + list(X.shape[1:]))
            self._target_placeholder = tf.placeholder(dtype=tf.int32, shape=[None])
            self._fc_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='dropout_keep_prob')
            # Build the model graph
            self._make_behavioral_map()
            self._make_loss()

            # initialize graph
            self._logger.debug('Initializing mapper')
            init_op = tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
            self._sess = tf.Session(
                config=tf.ConfigProto(gpu_options=self._gpu_options) if self._gpu_options is not None else None)
            self._sess.run(init_op)

    def labels_to_indices(self, labels):
        label2index = OrderedDict()
        indices = []
        for label in labels:
            if label not in label2index:
                label2index[label] = (max(label2index.values()) + 1) if len(label2index) > 0 else 0
            indices.append(label2index[label])
        index2label = OrderedDict((index, label) for label, index in label2index.items())
        return np.array(indices), index2label

    def fit(self, X, Y):
        """
        Fits the parameters to the data
        :param X: Source data, first dimension is examples
        :param Y: Target data, first dimension is examples
        """
        if self._zscore_feats:
            import sklearn
            self._scaler = sklearn.preprocessing.StandardScaler().fit(X)
            X = self._scaler.transform(X)
        Y, self._label_mapping = self.labels_to_indices(Y.values)
        self.setup()
        assert X.ndim == 2, 'Input matrix rank should be 2.'
        random_state = RandomState(0)
        with self._graph.as_default():
            self._init_mapper(X, Y)
            lr = self._lr
            for epoch in tqdm(range(self._max_epochs), desc='epochs'):
                for counter, batch in enumerate(
                        self._iterate_minibatches(X, Y, batchsize=self._train_batch_size, shuffle=True,
                                                  random_state=random_state)):
                    feed_dict = {self._input_placeholder: batch[0],
                                 self._target_placeholder: batch[1],
                                 self._lr_ph: lr,
                                 self._fc_keep_prob: self._fc_dropout}
                    _, loss_value, reg_loss_value = self._sess.run(
                        [self.train_op, self.classification_error, self.reg_loss],
                        feed_dict=feed_dict)
                if epoch % self._log_rate == 0:
                    self._logger.debug(f'Epoch: {epoch}, Err Loss: {loss_value:.2f}, Reg Loss: {reg_loss_value:.2f}')

                if loss_value < self._tol:
                    self._logger.debug('Converged.')
                    break

    def predict_proba(self, X):
        import tensorflow as tf
        assert len(X.shape) == 2, "expected 2-dimensional input"
        if self._zscore_feats:
            scaled_X = self._scaler.transform(X)
        else:
            scaled_X = X
        with self._graph.as_default():
            preds = []
            for batch in self._iterate_minibatches(scaled_X, batchsize=self._eval_batch_size, shuffle=False):
                feed_dict = {self._input_placeholder: batch, self._fc_keep_prob: 1.0}
                preds.append(np.squeeze(self._sess.run([tf.nn.softmax(self._predictions)], feed_dict=feed_dict)))
            proba = np.concatenate(preds, axis=0)
        # we take only the 0th dimension because the 1st dimension is just the features
        X_coords = {coord: (dims, value) for coord, dims, value in walk_coords(X)
                    if array_is_element(dims, X.dims[0])}
        proba = BehavioralAssembly(proba,
                                   coords={**X_coords, **{'choice': list(self._label_mapping.values())}},
                                   dims=[X.dims[0], 'choice'])
        return proba

    def close(self):
        """
        Closes occupied resources
        """
        import tensorflow as tf
        tf.reset_default_graph()
        self._sess.close()
