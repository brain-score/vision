import functools
import itertools
import logging
from collections import OrderedDict, Counter

import numpy as np
import pandas
import scipy.stats
import sklearn.linear_model
import sklearn.multioutput

from brainscore.assemblies import walk_coords, array_is_element, DataAssembly
from brainscore.metrics import Metric
from brainscore.utils import fullname


class I2n(Metric):
    """
    Rajalingham & Issa et al., 2018 http://www.jneurosci.org/content/early/2018/07/13/JNEUROSCI.0388-18.2018
    Schrimpf & Kubilius et al., 2018 https://www.biorxiv.org/content/early/2018/09/05/407007
    """

    class MatchToSampleClassifier(object):
        def __init__(self):
            classifier_c = 1e-3
            self._classifier = sklearn.linear_model.LogisticRegression(
                multi_class='multinomial', solver='newton-cg', C=classifier_c)
            self._label_mapping = None
            self._target_class = None

        def fit(self, X, Y):
            X = self._preprocess(X)
            self._target_class = type(Y)
            Y, self._label_mapping = self.labels_to_indices(Y.values)
            self._classifier.fit(X, Y)
            return self

        def predict_proba(self, X):
            assert len(X.shape) == 2, "expected 2-dimensional input"
            proba = self._classifier.predict_proba(X)
            # we take only the 0th dimension because the 1st dimension is just the features
            X_coords = {coord: (dims, value) for coord, dims, value in walk_coords(X)
                        if array_is_element(dims, X.dims[0])}
            proba = self._target_class(proba,
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

        def _preprocess(self, X):
            scaler = sklearn.preprocessing.StandardScaler().fit(X)
            return scaler.transform(X)

    def __init__(self):
        super().__init__()
        self._source_classifier = self.MatchToSampleClassifier()
        self._logger = logging.getLogger(fullname(self))

    def __call__(self, source, target):
        # TODO: compute scores on two splits of the target -- ceiling = correlation between the two halves
        # ideally, we do the split multiple times
        # for each split, correlate only the first half, then take the variance over splits
        source_response_matrix = self.class_probabilities_from_features(source)
        source_response_matrix = self.normalize_response_matrix(source_response_matrix)
        target_response_matrix = self.build_response_matrix_from_responses(target)
        target_response_matrix = self.normalize_response_matrix(target_response_matrix)
        correlation = self.correlate(source_response_matrix, target_response_matrix)
        return correlation

    def class_probabilities_from_features(self, source):
        source_without_behavior = source  # mock TODO: use the 2400 - 240 images where we don't use behavioral responses
        target = source_without_behavior['label']
        self._source_classifier.fit(source_without_behavior, target)

        source_with_behavior = source  # mock
        # source_with_behavior = source_with_behavior.drop_duplicates('id')
        # source_features = DataAssembly(source_features,
        #                                coords={'image_id': source_with_behavior['id'].values,
        #                                        'feature_id': list(range(source_features.shape[1]))},
        #                                dims=['image_id', 'feature_id'])

        prediction = self._source_classifier.predict_proba(source_with_behavior)  # TODO: use held-out here
        truth_labels = [source_with_behavior[source_with_behavior['image_id'] == image_id]['label'].values[0]
                        for image_id in prediction['image_id'].values]
        prediction['truth'] = 'presentation', truth_labels
        # prediction = prediction.set_index(presentation=['image_id', 'truth'])
        assert prediction.shape == (240, 24)
        return prediction

    def normalize_response_matrix(self, response_matrix):
        assert response_matrix.shape == (240, 24)

        target_distractor_scores = self.compute_target_distractor_scores(response_matrix)
        assert target_distractor_scores.shape == (240, 24)

        dprime_scores = self.dprime(target_distractor_scores)

        cap = 5
        dprime_scores = dprime_scores.clip(-cap, cap)
        assert dprime_scores.shape == (240, 24)

        dprime_scores_normalized = self.subtract_mean(dprime_scores)
        assert dprime_scores_normalized.shape == (240, 24)
        return dprime_scores_normalized

    def compute_target_distractor_scores(self, object_probabilities):
        cached_object_probabilities = self._build_index(object_probabilities, ['image_id', 'choice'])

        def apply(p_choice, image_id, truth, choice, **_):
            if truth == choice:  # object == choice, ignore
                return np.nan
            # probability that something else was chosen rather than object (p_choice == p_distractor after above check)
            p_object = cached_object_probabilities[(image_id, truth)]
            p = p_object / (p_object + p_choice)
            return p

        result = object_probabilities.multi_dim_apply(['image_id', 'choice'], apply)
        return result

    def dprime(self, target_distractor_scores):
        cached_target_distractor_scores = self._build_index(target_distractor_scores, ['truth', 'choice'])

        def apply(hit_rate, choice, truth, **_):
            inverse_choice = cached_target_distractor_scores[(choice, truth)]
            assert inverse_choice.size > 0
            false_alarms_rate = 1 - np.nanmean(inverse_choice)
            dprime = self.z_score(hit_rate) - self.z_score(false_alarms_rate)
            return dprime

        result = target_distractor_scores.multi_dim_apply(['image_id', 'choice'], apply)
        return result

    def z_score(self, value):
        return scipy.stats.norm.ppf(value)

    def subtract_mean(self, scores):
        def apply(group, **_):
            return group - group.mean()

        result = scores.multi_dim_apply(['truth', 'choice'], apply)
        return result

    def build_response_matrix_from_responses(self, responses):
        num_images = Counter(responses['image_id'].values)
        num_choices = [(image_id, choice) for image_id, choice in zip(responses['image_id'].values, responses.values)]
        num_choices = Counter(num_choices)

        choices = np.unique(responses)
        image_ids, indices = np.unique(responses['image_id'], return_index=True)
        image_dim = responses['image_id'].dims
        coords = {**{coord: (dims, value) for coord, dims, value in walk_coords(responses)},
                  **{'choice': ('choice', choices)}}
        coords = {coord: (dims, value if dims != image_dim else value[indices])  # align image_dim coords with indices
                  for coord, (dims, value) in coords.items()}
        response_matrix = np.zeros((len(image_ids), len(choices)))
        for (image_index, image_id), (choice_index, choice) in itertools.product(
                enumerate(image_ids), enumerate(choices)):
            p = num_choices[(image_id, choice)] / num_images[image_id]
            response_matrix[image_index, choice_index] = p
        response_matrix = DataAssembly(response_matrix, coords=coords, dims=responses.dims + ('choice',))
        return response_matrix

    def correlate(self, source_response_matrix, target_response_matrix):
        # align
        source_response_matrix = source_response_matrix.sortby('image_id').sortby('choice')
        target_response_matrix = target_response_matrix.sortby('image_id').sortby('choice')
        # flatten and mask out NaNs
        source, target = source_response_matrix.values.flatten(), target_response_matrix.values.flatten()
        non_nan = ~np.isnan(target)
        source, target = source[non_nan], target[non_nan]
        assert not any(np.isnan(source))
        correlation = np.corrcoef(source, target)
        return correlation[0, 1]

    def _build_index(self, values, coords):
        np.testing.assert_array_equal(list(itertools.chain(*[values[coord].dims for coord in coords])), values.dims)
        return {coord_values: value for coord_values, value in zip(
            itertools.product(*[values[coord].values for coord in coords]),
            values.values.flatten())}


#
#
#
#

class I2nCopied(Metric):
    """
    Rajalingham & Issa et al., 2018 http://www.jneurosci.org/content/early/2018/07/13/JNEUROSCI.0388-18.2018
    Schrimpf & Kubilius et al., 2018 http://brain-score.org
    """

    def __init__(self, cap=5):
        super().__init__()
        self._cap = cap
        classifier_c = 1e-3
        self._source_classifier = MatchToSampleClassifier(C=classifier_c)
        self._target_classifier = MatchToSampleClassifier(C=classifier_c)
        self._logger = logging.getLogger(fullname(self))

    def __call__(self, train_source, train_target, test_source, test_target):
        self._logger.debug("Fitting")
        self.fit(train_source, train_target)
        self._logger.debug("Predicting")
        prediction = self.predict(test_source)
        self._logger.debug("Comparing")
        similarity = self.compare_prediction(prediction, test_target)
        return similarity

    def fit(self, source, target):
        source_features = source['features'].values
        self._source_classifier.fit(source_features, source['label'])
        # self._target_classifier.fit(target, target['label'])  # TODO ??

    def predict(self, test_source):
        predictions = self._source_classifier.predict_proba(test_source, targets=ground_truth, kind='2-way')
        return predictions

    def compare_prediction(self, prediction, target, axis='neuroid_id', correlation=scipy.stats.pearsonr):
        def hitrate_to_dprime(x):
            idx = x.name
            hit_rate = np.nanmean(x)

            # idx: (target, imid, distr)
            rej = df.loc[(df[target] == idx[2]) & (df['distr'] == idx[0]), value]

            fa_rate = 1 - np.nanmean(rej)

            output = scipy.stats.norm.ppf(hit_rate) - scipy.stats.norm.ppf(fa_rate)
            output = np.clip(output, -self._cap, self._cap)
            return output

        predictions_dprime = prediction.groupby(['obj', 'id', 'distr'])['acc'].apply(hitrate_to_dprime)
        predictions_dprime['acc'] = predictions_dprime.groupby(['obj', 'distr']).transform(lambda x: x - x.mean())

        df = []
        inds = np.isfinite(target[0]) & np.isfinite(target[1]) & np.isfinite(predictions_dprime)
        c0 = np.corrcoef(predictions_dprime[inds], target[0][inds])[0, 1]
        c1 = np.corrcoef(predictions_dprime[inds], target[1][inds])[0, 1]
        corr = (c0 + c1) / 2
        ic = np.corrcoef(target[0][inds], target[1][inds])[0, 1]
        df.append([iterno, ic, corr, corr / np.sqrt(ic)])
        df = pandas.DataFrame(df, columns=['split', 'internal_cons', 'r', 'cons'])
        return corr

    def _apply(self, model_feats, human_data):
        # def objectome_cons(model_feats,  # metric='i2n', kind='dprime',
        #                    target='obj', distr='distr', imid='id', value='acc', cap=20):
        target = 'obj'
        distr = 'distr'
        imid = 'id'
        value = 'acc'
        cap = 5

        obj = Objectome()
        obj24 = Objectome24s10()
        hkind = 'I2_dprime_C'
        human_data = obj24.human_data(kind=hkind)
        test_idx = pandas.read_pickle(obj24.datapath('sel240'))

        clf = MatchToSampleClassifier(C=1e-3)
        train_idx = [i for i in range(len(obj.meta.obj)) if i not in test_idx]

        # obj.meta.obj.iloc[train_idx] = object labels
        # obj.OBJS = order of objects, to make sure i2 matrix is organized in same way
        clf.fit(model_feats[train_idx], obj.meta.obj.iloc[train_idx], order=obj.OBJS)
        preds = clf.predict_proba(model_feats[test_idx],
                                  targets=obj.meta.obj.iloc[test_idx], kind='2-way')
        df = pandas.DataFrame(preds, index=obj.meta.obj.iloc[test_idx], columns=obj.OBJS).reset_index()
        df['id'] = obj.meta.id.iloc[test_idx].values
        df = df.set_index(['obj', 'id'])
        df = df.stack().reset_index()
        df = df.rename(columns={'level_2': 'distr', 0: 'acc'})

        df = df[['obj', 'id', 'distr', 'acc']]
        indices = [target, imid, distr]

        def hitrate_to_dprime(x):
            idx = x.name
            hit_rate = np.nanmean(x)

            # idx: (target, imid, distr)
            rej = df.loc[(df[target] == idx[2]) & (df[distr] == idx[0]), value]

            fa_rate = 1 - np.nanmean(rej)

            output = scipy.stats.norm.ppf(hit_rate) - scipy.stats.norm.ppf(fa_rate)
            output = np.clip(output, -cap, cap)
            return output

        dprime = df.groupby(indices)['acc'].apply(hitrate_to_dprime)
        dprime = dprime.reset_index()
        by = [target, distr]
        dprime[value] = dprime.groupby(by)[value].transform(lambda x: x - x.mean())

        # distractor and target order are different - manually re-order
        obj_order = ['lo_poly_animal_RHINO_2', 'calc01', 'womens_shorts_01M', 'zebra', 'MB27346', 'build51',
                     'weimaraner', 'interior_details_130_2', 'lo_poly_animal_CHICKDEE', 'kitchen_equipment_knife2',
                     'interior_details_103_4', 'lo_poly_animal_BEAR_BLK', 'MB30203', 'antique_furniture_item_18',
                     'lo_poly_animal_ELE_AS1', 'MB29874', 'womens_stockings_01M', 'Hanger_02', 'dromedary',
                     'MB28699', 'lo_poly_animal_TRANTULA', 'flarenut_spanner', 'MB30758', '22_acoustic_guitar']

        dprime.obj = dprime.obj.astype(pandas.api.types.CategoricalDtype(ordered=True, categories=obj_order))

        mm = obj.meta.iloc[test_idx]
        id_order = np.concatenate([mm[mm.obj == o].id for o in obj_order])
        dprime.id = dprime.id.astype(pandas.api.types.CategoricalDtype(ordered=True, categories=id_order))

        dprime = dprime.sort_values('id')
        preds = dprime.set_index(['id', 'obj', 'distr']).unstack('distr')

        preds = preds.fillna(np.nan).values

        df = []
        for iterno, split in enumerate(tqdm.tqdm(human_data)):
            inds = np.isfinite(split[0]) & np.isfinite(split[1]) & np.isfinite(preds)
            c0 = np.corrcoef(preds[inds], split[0][inds])[0, 1]
            c1 = np.corrcoef(preds[inds], split[1][inds])[0, 1]
            corr = (c0 + c1) / 2
            ic = np.corrcoef(split[0][inds], split[1][inds])[0, 1]
            df.append([iterno, ic, corr, corr / np.sqrt(ic)])
        df = pandas.DataFrame(df, columns=['split', 'internal_cons', 'r', 'cons'])
        return df


class MatchToSampleClassifierCopied(object):
    def __init__(self, norm=True, nfeats=None, seed=None, C=1):
        """
        A classifier for the Delayed Match-to-Sample task.
        It is formulated as a typical sklearn classifier with `score`, `predict_proba`
        and `fit` methods available.
        :Kwargs:
            - norm (bool, default: True)
                Whether to zscore features or not.
            - nfeats (int or None, default: None)
                The number of features to use. Useful when you want to match the
                number of features across layers. If None, all features are used.
            - seed (int or None, default: None)
                Random seed for feature selecition
        """
        self.norm = norm
        self.nfeats = nfeats
        self.seed = seed
        self.C = C

    def preproc(self, X, reset=False):
        if self.norm:
            if reset:
                self.scaler = sklearn.preprocessing.StandardScaler().fit(X)
            X = self.scaler.transform(X)
        else:
            self.scaler = None

        if self.nfeats is not None:
            if reset:
                sel = np.random.RandomState(self.seed).permutation(X.shape[1])[:self.nfeats]
            X = X[:, sel]
        return X

    def fit(self, X, y, order=None):
        """
        :Kwargs:
            - order
                Label order. If None, will be sorted alphabetically
        """
        if order is None:
            order = np.unique(y)
        self.label_dict = OrderedDict([(obj, o) for o, obj in enumerate(order)])
        y = self.labels2inds(y)
        X = self.preproc(X, reset=True)
        self.clf = sklearn.linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg', C=self.C)
        self.clf.fit(X, y)

    def _acc(self, x, y):
        return x / (x + y)

    def _dprime(self, x, y):
        return scipy.stats.norm.ppf(x) - scipy.stats.norm.ppf(y)

    def predict_proba(self, X, targets=None, distrs=None, kind='2-way', measure='acc'):
        """
        Model classification confidence (range 0-1)
        """
        if not hasattr(self, 'clf'):
            raise Exception('Must train the classifier first')

        if measure not in ['acc', 'dprime', "d'"]:
            raise ValueError('measure {} not recognized'.format(measure))

        measure_op = self._acc if measure == 'acc' else self._dprime

        X = self.preproc(X)
        conf = self.clf.predict_proba(X)

        if targets is not None:
            if isinstance(targets, str):
                targets = [targets]
            ti = self.labels2inds(targets)
            # target probability
            t = np.array([x[i] for x, i in zip(conf, ti)])

            if distrs is not None:
                if isinstance(distrs, str):
                    distrs = [distrs]
                dinds = self.labels2inds(distrs)
                # distractor probability
                d = np.array([c[di] for c, di in zip(conf, dinds)])
                acc = measure_op(t, d)

            elif kind == '2-way':
                acc = []
                for c, target in zip(conf, targets):
                    ti = self.label_dict[target]
                    c_tmp = []
                    # compute distractor probability for each distractor
                    for di in self.label_dict.values():
                        if di != ti:
                            tmp = measure_op(c[ti], c[di])
                            c_tmp.append(tmp)
                        else:
                            c_tmp.append(np.nan)
                    acc.append(c_tmp)
                acc = pandas.DataFrame(acc, index=targets, columns=list(self.label_dict.keys()))

            else:
                acc = t
        else:
            acc = conf

        return acc

    def labels2inds(self, y):
        """
        Converts class labels (usually strings) to indices
        """
        return np.array([self.label_dict[x] for x in y])

    def score(self, X, y, kind='2-way', measure='dprime', cap=5):
        """
        Classification accuracy.
        Accuracy is either 0 or 1. For a 2-way classifier, this depends on
        `predict_proba` being less or more that .5. For an n-way classifier, it
        checks if argmax of `predict_proba` gives the correct or incorrect class.
        """
        if kind == '2-way':
            acc = self.predict_proba(X, targets=y, kind=kind)
            acc[~np.isnan(acc)] = acc[~np.isnan(acc)] > .5
        else:
            conf = self.predict_proba(X, kind=kind)
            y = self.labels2inds(y)
            acc = np.argmax(conf, 1) == y

        if measure == 'dprime':
            acc = hitrate_to_dprime_o1(acc, cap=cap)

        return acc


def hitrate_to_dprime_o1(df, cap=20):
    targets = df.index.unique()
    out = pandas.Series(np.zeros(len(targets)), index=targets)
    for target in targets:
        hit_rate = np.nanmean(df.loc[df.index == target])
        fa_rate = np.nanmean(1 - df.loc[df.index != target, target])
        dprime = scipy.stats.norm.ppf(hit_rate) - scipy.stats.norm.ppf(fa_rate)
        dprime = np.clip(dprime, -cap, cap)
        out[target] = dprime
    return out


def lazy_property(function):
    """
    From: https://danijar.com/structuring-your-tensorflow-models/
    """
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


class Dataset(object):
    BUCKET = 'dicarlocox-datasets'
    COLL = 'streams'

    def home(self, *suffix_paths):
        return os.path.join(DATA_HOME, self.name, *suffix_paths)

    def datapath(self, handle, prefix=None):
        data = self.DATA[handle]
        if isinstance(data, tuple):
            s3_path, sha1, local_path = data
            local_path = os.path.join(local_path, s3_path)
            # if local_path is None:
            #     local_path = s3_path.replace(self.COLL + '/' + self.name + '/', '', 1)
        else:
            local_path = data.replace(self.COLL + '/' + self.name + '/', '', 1)
        if prefix is not None:
            local_path = '/'.join([prefix, local_path])
        return self.home(local_path)

    def fetch(self):
        return
        if not os.path.exists(self.home()):
            os.makedirs(self.home())

        session = boto3.Session()
        client = session.client('s3')

        for data in self.DATA.values():
            if isinstance(data, tuple):
                s3_path, sha1, local_path = data
                if local_path is None:
                    local_path = s3_path.replace(self.COLL + '/' + self.name + '/', '', 1)
            else:
                local_path = data.replace(self.COLL + '/' + self.name + '/', '', 1)
                s3_path = data
                sha1 = None

            local_path = self.home(local_path)
            if not os.path.exists(local_path):
                # rel_path = os.path.relpath(local_path, DATA_HOME)
                # s3_path = os.path.join(self.COLL, rel_path)
                local_dir = os.path.dirname(local_path)
                if not os.path.exists(local_dir):
                    os.makedirs(local_dir)
                client.download_file(self.BUCKET, s3_path, local_path)
                if sha1 is not None:
                    with open(local_path) as f:
                        if sha1 != hashlib.sha1(f.read()).hexdigest():
                            raise IOError("File '{}': SHA-1 does not match.".format(filename))

    def upload(self, pattern='*'):
        raise NotImplementedError
        session = boto3.Session()
        client = session.client('s3')

        uploads = []
        for root, dirs, filenames in os.walk(self.home()):
            for filename in glob.glob(os.path.join(root, pattern)):
                local_path = os.path.join(root, filename)
                rel_path = os.path.relpath(local_path, DATA_HOME)
                s3_path = os.path.join(self.COLL, rel_path)
                try:
                    client.head_object(Bucket=self.BUCKET, Key=s3_path)
                except:
                    uploads.append((local_path, s3_path))

        if len(uploads) > 0:
            text = []
            for local_path, s3_path in uploads:
                with open(local_path) as f:
                    sha1 = hashlib.sha1(f.read()).hexdigest()
                    rec = '    {} (sha-1: {})'.format(s3_path, sha1)
                text.append(rec)
            text = ['Will upload:'] + text + ['Proceed? ']
            proceed = raw_input('\n'.join(text))
            if proceed == 'y':
                for local_path, s3_path in tqdm.tqdm(uploads):
                    client.upload_file(local_path, self.BUCKET, s3_path)
        else:
            print('nothing found to upload')

    def _upload(self, filename):
        session = boto3.Session()
        client = session.client('s3')
        local_path = self.home(filename)
        rel_path = os.path.relpath(local_path, DATA_HOME)
        s3_path = os.path.join(self.COLL, rel_path)
        client.upload_file(local_path, self.BUCKET, s3_path)

    # def move(self, old_path, new_path):
    #     client.copy_object(Bucket=self.BUCKET, Key=new_path,
    #                         CopySource=self.BUCKET + '/' + old_path)
    #     client.delete_object(Bucket=self.BUCKET, Key=new_path)

    @property
    def meta(self):
        if not hasattr(self, '_meta'):
            self.fetch()
            self._meta = pandas.read_pickle(self.datapath('meta'))
        return self._meta

    def images(self, size=256):
        try:
            ims = np.load(self.datapath('images{}'.format(size)))
        except:
            ims = []
            for idd in tqdm.tqdm(self.meta.id.values, desc='processing images'):
                im = skimage.io.imread(self.home('imageset/images', idd + '.png'))
                im = skimage.transform.resize(im, (size, size))
                assert im.min() >= 0
                assert im.max() <= 1
                im = skimage.color.gray2rgb(im)
                assert im.ndims == 3
                ims.append(im)
            ims = np.array(ims)
            np.save(self.datapath('images{}'.format(size)), ims)
        return ims

    def tokens(self, size=256):
        ims = []
        for idd in self.meta.obj.unique():
            im = skimage.io.imread(self.home('imageset/images', idd + '.png'))
            im = skimage.transform.resize(im, (size, size))
            assert im.min() >= 0
            assert im.max() <= 1
            im = skimage.color.gray2rgb(im)
            assert im.ndims == 3
            ims.append(im)
        ims = np.array(ims)


class Objectome(Dataset):
    DATA = {'meta': 'streams/objectome/meta.pkl',
            'images256': 'streams/objectome/imageset/ims24s100_256.npy',
            'imageset/tfrecords': 'streams/objectome/imageset/images224.tfrecords',
            }

    OBJS = ['lo_poly_animal_RHINO_2',
            'MB30758',
            'calc01',
            'interior_details_103_4',
            'zebra',
            'MB27346',
            'build51',
            'weimaraner',
            'interior_details_130_2',
            'lo_poly_animal_CHICKDEE',
            'kitchen_equipment_knife2',
            'lo_poly_animal_BEAR_BLK',
            'MB30203',
            'antique_furniture_item_18',
            'lo_poly_animal_ELE_AS1',
            'MB29874',
            'womens_stockings_01M',
            'Hanger_02',
            'dromedary',
            'MB28699',
            'lo_poly_animal_TRANTULA',
            'flarenut_spanner',
            'womens_shorts_01M',
            '22_acoustic_guitar']

    def __init__(self):
        self.name = 'objectome'


class Objectome24s10(Objectome):
    DATA = {'meta': 'streams/objectome/meta.pkl',
            'images224': 'streams/objectome/imageset/ims24s10_224.npy',
            'sel240': 'streams/objectome/sel240.pkl',
            'metrics240': 'streams/objectome/metrics240.pkl'}
    OBJS = Objectome.OBJS

    @lazy_property
    def meta(self):
        meta = super(Objectome24s10, self).meta
        sel = pandas.read_pickle(self.datapath('sel240'))
        return meta.loc[sel]

    def human_data(self, kind='I2_dprime_C'):
        """
        Kind:
        - O1_hitrate, O1_accuracy, O1_dprime, O1_dprime_v2
        - O2_hitrate, O2_accuracy, O2_dprime,
        - I1_hitrate, I1_accuracy, I1_dprime, I1_dprime_C, I1_dprime_v2_C
        - I2_hitrate, I2_accuracy, I2_dprime, I2_dprime_C, I1_dprime_v2
        Rishi: "'v2' means averaging rather than pooling. So O1_dprime_v2 averages over all the distracter bins from O2, rather than pooling over all the trials."
        """
        data = pandas.read_pickle(self.datapath('metrics240'))
        # organized like: metric kind x 10 splits x 2 split halves
        return data[kind]
