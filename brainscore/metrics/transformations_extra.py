from collections import OrderedDict

import itertools
import logging
import math
import numpy as np
import os
import xarray as xr
import pandas as pd
from brainio_base.assemblies import DataAssembly, walk_coords
from brainio_collection.transform import subset
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, KFold, StratifiedKFold
from tqdm import tqdm

from brainscore.metrics import Score
from brainscore.metrics.utils import unique_ordered
from brainscore.utils import fullname
from brainscore.metrics.transformations import Transformation, enumerate_done, Split


class CrossValidationCustomPlusBaseline(Transformation):
    """
    Performs multiple splits over a source and target assembly.
    No guarantees are given for data-alignment, use the metadata.
    """

    def __init__(self, split_coord=Split.Defaults.split_coord,
                 stratification_coord=Split.Defaults.stratification_coord, expecting_coveriate=False, **kwargs):
        self._split_coord = split_coord
        self._stratification_coord = stratification_coord
        self._split_kwargs = kwargs or {}
        self._split = None # replacing this with an actual Split instance inside the pipe
        self._logger = logging.getLogger(fullname(self))

        # argument to provide csv files with the indexes needed
        self._given_indices_parent_folder = kwargs.get('parent_folder', None)
        self._given_indices_file = kwargs.get('csv_file', None)

        if self._given_indices_file and self._given_indices_parent_folder:
            self._train_csv, self._test_csv = self._get_csv(self._given_indices_parent_folder, self._given_indices_file)
        else:
            self._train_csv, self._test_csv = (None, None)

        # In case we also need to split a covariate
        if not expecting_coveriate:
            self.pipe = self.pipe_default
        else:
            self.pipe = self.pipe_covariate

    def _get_csv(self, parent_folder, csv_file):
        parent = parent_folder
        train_file = os.path.join(parent, 'train' + csv_file.split('/')[-1])
        test_file = os.path.join(parent, 'test' + csv_file.split('/')[-1])
        train_csv = pd.read_csv(train_file, names=['path', 'id', 'cat', 'full'])
        test_csv = pd.read_csv(test_file, names=['path', 'id', 'cat', 'full'])
        return train_csv, test_csv

    def _build_splits_file(self, train_csv, test_csv, cross_validation_values):
        all_values = list(cross_validation_values['presentation'].values)
        all_values = [a[0] for a in all_values]
        train_ids = list(train_csv['id'])
        test_ids = list(test_csv['id'])
        both_train = set(all_values).intersection(train_ids)
        both_test = set(all_values).intersection(test_ids)
        indices_train = [all_values.index(x) for x in both_train]
        indices_test = [all_values.index(x) for x in both_test]
        return [[indices_train, indices_test]]

    def pipe_covariate(self, source_assembly, covariate_assembly, target_assembly):
        # check only for equal values, alignment is given by metadata

        assert sorted(source_assembly[self._split_coord].values) == \
               sorted(covariate_assembly[self._split_coord].values) == \
               sorted(target_assembly[self._split_coord].values)

        if self._stratification_coord:
            assert hasattr(source_assembly, self._stratification_coord)
            assert sorted(source_assembly[self._stratification_coord].values) == \
                   sorted(covariate_assembly[self._stratification_coord].values) == \
                   sorted(target_assembly[self._stratification_coord].values)

        if self._train_csv is not None and self._test_csv is not None:
            train_size = len(self._train_csv)/len(source_assembly[self._split_coord].values)
            test_size = len(self._test_csv)/len(source_assembly[self._split_coord].values)

        self._split = Split(split_coord=self._split_coord,
                       stratification_coord=self._stratification_coord,
                       train_size=train_size,
                       test_size=test_size,
                       **self._split_kwargs)

        cross_validation_values, splits = self._split.build_splits(target_assembly)

        if self._train_csv is not None and self._test_csv is not None:
            splits = self._build_splits_file(self._train_csv, self._test_csv, cross_validation_values) + splits

        split_scores = []

        for split_iterator, (train_indices, test_indices), done \
                in tqdm(enumerate_done(splits), total=len(splits), desc='cross-validation'):
            train_values, test_values = cross_validation_values[train_indices], cross_validation_values[test_indices]
            train_source = subset(source_assembly, train_values, dims_must_match=False)
            train_covariate = subset(covariate_assembly, train_values, dims_must_match=False)
            train_target = subset(target_assembly, train_values, dims_must_match=False)
            assert len(train_source[self._split_coord]) == len(train_covariate[self._split_coord]) == len(
                train_target[self._split_coord])
            test_source = subset(source_assembly, test_values, dims_must_match=False)
            test_covariate = subset(covariate_assembly, test_values, dims_must_match=False)
            test_target = subset(target_assembly, test_values, dims_must_match=False)
            assert len(test_source[self._split_coord]) == len(test_covariate[self._split_coord]) == len(
                test_target[self._split_coord])

            split_score = yield from self._get_result(train_source, train_covariate, train_target, test_source,
                                                      test_covariate, test_target,
                                                      done=done)
            split_score = split_score.expand_dims('split')
            split_score['split'] = [split_iterator]
            split_scores.append(split_score)

        split_scores = Score.merge(*split_scores)
        yield split_scores

    def pipe_default(self, source_assembly, target_assembly):
        # check only for equal values, alignment is given by metadata
        assert sorted(source_assembly[self._split_coord].values) == sorted(target_assembly[self._split_coord].values)
        if self._split.do_stratify:
            assert hasattr(source_assembly, self._stratification_coord)
            assert sorted(source_assembly[self._stratification_coord].values) == \
                   sorted(target_assembly[self._stratification_coord].values)

        if self._train_csv and self._test_csv:
            train_size = len(self._train_csv)/len(source_assembly[self._split_coord].values)
            test_size = len(self._test_csv)/len(source_assembly[self._split_coord].values)

        self._split = Split(split_coord=self._split_coord,
                       stratification_coord=self._stratification_coord,
                       train_size=train_size,
                       test_size=test_size,
                       **self._split_kwargskwargs)

        cross_validation_values, splits = self._split.build_splits(target_assembly)

        if self._train_csv and self._test_csv:
            splits = self._build_splits_file(cross_validation_values) + splits

        split_scores = []

        for split_iterator, (train_indices, test_indices), done \
                in tqdm(enumerate_done(splits), total=len(splits), desc='cross-validation'):
            train_values, test_values = cross_validation_values[train_indices], cross_validation_values[test_indices]
            train_source = subset(source_assembly, train_values, dims_must_match=False)
            train_target = subset(target_assembly, train_values, dims_must_match=False)
            assert len(train_source[self._split_coord]) == len(train_target[self._split_coord])
            test_source = subset(source_assembly, test_values, dims_must_match=False)
            test_target = subset(target_assembly, test_values, dims_must_match=False)
            assert len(test_source[self._split_coord]) == len(test_target[self._split_coord])

            split_score = yield from self._get_result(train_source, train_target, test_source, test_target,
                                                      done=done)
            split_score = split_score.expand_dims('split')
            split_score['split'] = [split_iterator]
            split_scores.append(split_score)

        split_scores = Score.merge(*split_scores)
        yield split_scores

    def aggregate(self, score):
        return self._split.aggregate(score)