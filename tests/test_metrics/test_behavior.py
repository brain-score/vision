import os

import numpy as np
import pandas as pd
import xarray as xr
from pytest import approx

from brainscore.assemblies import BehavioralAssembly, walk_coords
from brainscore.metrics.behavior import I2n


class TestI2N(object):
    def test_objectome(self):
        # objectome = xr.open_dataarray(os.path.join(os.path.dirname(__file__), 'monkobjectome_behavior.nc'))
        # i2n = I2n()
        # score = i2n(train_source=objectome, train_target=objectome, test_source=objectome, test_target=objectome)
        # score = i2n.aggregate(score)
        # expected_score = 0.826
        # assert score == approx(expected_score, abs=0.01)




        # objectome = xr.open_dataarray(os.path.join(os.path.dirname(__file__), 'monkobjectome_behavior.nc'))
        # # objectome = objectome[:13]  # FIXME
        # objectome = BehavioralAssembly(objectome)
        # print("obj: {}, id: {}".format(len(np.unique(objectome['obj'])), len(np.unique(objectome['id']))))
        # # filter objects
        # objects = ['lo_poly_animal_RHINO_2',
        #            'MB30758',
        #            'calc01',
        #            'interior_details_103_4',
        #            'zebra',
        #            'MB27346',
        #            'build51',
        #            'weimaraner',
        #            'interior_details_130_2',
        #            'lo_poly_animal_CHICKDEE',
        #            'kitchen_equipment_knife2',
        #            'lo_poly_animal_BEAR_BLK',
        #            'MB30203',
        #            'antique_furniture_item_18',
        #            'lo_poly_animal_ELE_AS1',
        #            'MB29874',
        #            'womens_stockings_01M',
        #            'Hanger_02',
        #            'dromedary',
        #            'MB28699',
        #            'lo_poly_animal_TRANTULA',
        #            'flarenut_spanner',
        #            'womens_shorts_01M',
        #            '22_acoustic_guitar']
        # indexer = [obj in objects for obj in objectome['obj'].values]
        # objectome = objectome[indexer]
        # # infer labels and distractors
        # labels = [response if correct else (choice1 if response == choice2 else choice2)
        #           for correct, response, choice1, choice2 in zip(
        #         objectome.values, objectome['Response'].values, objectome['Choice_1'].values,
        #         objectome['Choice_2'].values)]
        # objectome['label'] = ('trial', labels)
        # distractors = [response if not correct else (choice1 if response == choice2 else choice2)
        #                for correct, response, choice1, choice2 in zip(
        #         objectome.values, objectome['Response'].values, objectome['Choice_1'].values,
        #         objectome['Choice_2'].values)]
        # objectome['distr'] = 'trial', distractors
        # # to pandas
        # objectome = pd.DataFrame({**{coord: values for coord, dims, values in walk_coords(objectome)},
        #                           **{'correct': objectome.values}})
        objectome = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'objectome24s100_humanpool.pkl'))
        objectome['correct'] = objectome['choice'] == objectome['sample_obj']

        # model mock
        # responses = objectome['Response'].values
        responses = objectome['choice'].values
        feature_responses = [np.array([1 / float(hash(response))]) for response in responses]  # FIXME
        # feature_responses = BehavioralAssembly(
        #     feature_responses,
        #     coords={**{coord: ('trial', values.values) for coord, values in objectome.coords.items()},
        #             **{'neuroid': [0]}},  # FIXME
        #     dims=['trial', 'neuroid'])
        feature_responses = pd.DataFrame({**{column: objectome[column] for column in objectome.columns
                                             if column not in ['correct', 'Response', 'choice']},
                                          **{'features': feature_responses}})

        i2n = I2n()
        # objectome = objectome.multi_groupby(['obj', 'distr', 'id']).mean()
        objectome = objectome.groupby(['sample_obj', 'dist_obj', 'id']).mean()
        np.testing.assert_array_equal(len(objectome), [240 * 24])
        score = i2n(train_source=feature_responses, train_target=objectome,
                    test_source=feature_responses, test_target=objectome)
        score = i2n.aggregate(score)
        expected_score = 0.826
        assert score == approx(expected_score, abs=0.01)
import os

import numpy as np
import pandas as pd
import xarray as xr
from pytest import approx

from brainscore.assemblies import BehavioralAssembly, walk_coords
from brainscore.metrics.behavior import I2n


class TestI2N(object):
    def test_objectome(self):
        # objectome = xr.open_dataarray(os.path.join(os.path.dirname(__file__), 'monkobjectome_behavior.nc'))
        # i2n = I2n()
        # score = i2n(train_source=objectome, train_target=objectome, test_source=objectome, test_target=objectome)
        # score = i2n.aggregate(score)
        # expected_score = 0.826
        # assert score == approx(expected_score, abs=0.01)




        # objectome = xr.open_dataarray(os.path.join(os.path.dirname(__file__), 'monkobjectome_behavior.nc'))
        # # objectome = objectome[:13]  # FIXME
        # objectome = BehavioralAssembly(objectome)
        # print("obj: {}, id: {}".format(len(np.unique(objectome['obj'])), len(np.unique(objectome['id']))))
        # # filter objects
        # objects = ['lo_poly_animal_RHINO_2',
        #            'MB30758',
        #            'calc01',
        #            'interior_details_103_4',
        #            'zebra',
        #            'MB27346',
        #            'build51',
        #            'weimaraner',
        #            'interior_details_130_2',
        #            'lo_poly_animal_CHICKDEE',
        #            'kitchen_equipment_knife2',
        #            'lo_poly_animal_BEAR_BLK',
        #            'MB30203',
        #            'antique_furniture_item_18',
        #            'lo_poly_animal_ELE_AS1',
        #            'MB29874',
        #            'womens_stockings_01M',
        #            'Hanger_02',
        #            'dromedary',
        #            'MB28699',
        #            'lo_poly_animal_TRANTULA',
        #            'flarenut_spanner',
        #            'womens_shorts_01M',
        #            '22_acoustic_guitar']
        # indexer = [obj in objects for obj in objectome['obj'].values]
        # objectome = objectome[indexer]
        # # infer labels and distractors
        # labels = [response if correct else (choice1 if response == choice2 else choice2)
        #           for correct, response, choice1, choice2 in zip(
        #         objectome.values, objectome['Response'].values, objectome['Choice_1'].values,
        #         objectome['Choice_2'].values)]
        # objectome['label'] = ('trial', labels)
        # distractors = [response if not correct else (choice1 if response == choice2 else choice2)
        #                for correct, response, choice1, choice2 in zip(
        #         objectome.values, objectome['Response'].values, objectome['Choice_1'].values,
        #         objectome['Choice_2'].values)]
        # objectome['distr'] = 'trial', distractors
        # # to pandas
        # objectome = pd.DataFrame({**{coord: values for coord, dims, values in walk_coords(objectome)},
        #                           **{'correct': objectome.values}})
        objectome = pd.read_pickle(os.path.join(os.path.dirname(__file__), 'objectome24s100_humanpool.pkl'))
        objectome['correct'] = objectome['choice'] == objectome['sample_obj']

        # model mock
        # responses = objectome['Response'].values
        responses = objectome['choice'].values
        feature_responses = [np.array([1 / float(hash(response))]) for response in responses]  # FIXME
        # feature_responses = BehavioralAssembly(
        #     feature_responses,
        #     coords={**{coord: ('trial', values.values) for coord, values in objectome.coords.items()},
        #             **{'neuroid': [0]}},  # FIXME
        #     dims=['trial', 'neuroid'])
        feature_responses = pd.DataFrame({**{column: objectome[column] for column in objectome.columns
                                             if column not in ['correct', 'Response', 'choice']},
                                          **{'features': feature_responses}})

        i2n = I2n()
        # objectome = objectome.multi_groupby(['obj', 'distr', 'id']).mean()
        objectome = objectome.groupby(['sample_obj', 'dist_obj', 'id']).mean()
        np.testing.assert_array_equal(len(objectome), [240 * 24])
        score = i2n(train_source=feature_responses, train_target=objectome,
                    test_source=feature_responses, test_target=objectome)
        score = i2n.aggregate(score)
        expected_score = 0.826
        assert score == approx(expected_score, abs=0.01)
