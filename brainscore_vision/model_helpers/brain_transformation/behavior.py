import os
from collections import OrderedDict
from typing import Union, List

import numpy as np
import pandas as pd
import xarray as xr
import sklearn.linear_model
import sklearn.multioutput

from brainscore_core.supported_data_standards.brainio.assemblies import walk_coords, array_is_element, BehavioralAssembly, DataAssembly
from brainscore_core.supported_data_standards.brainio.stimuli import StimulusSet
from brainscore_vision.model_helpers.utils import make_list
from brainscore_vision.model_interface import BrainModel


class BehaviorArbiter(BrainModel):
    def __init__(self, mapping):
        self.mapping = mapping
        self.current_executor = None

    def start_task(self, task: BrainModel.Task, *args, **kwargs):
        self.current_executor = self.mapping[task]
        return self.current_executor.start_task(task, *args, **kwargs)

    def look_at(self, stimuli, *args, **kwargs):
        return self.current_executor.look_at(stimuli, *args, **kwargs)


class LabelBehavior(BrainModel):
    def __init__(self, identifier, activations_model):
        self._identifier = identifier
        self.activations_model = activations_model
        self.current_task = None
        self.choice_labels = None

    @property
    def identifier(self):
        return self._identifier

    def start_task(self, task: BrainModel.Task, choice_labels):
        assert task == BrainModel.Task.label
        self.current_task = task
        self.choice_labels = choice_labels

    def look_at(self, stimuli, number_of_trials: int = 1, require_variance: bool = False):
        assert self.current_task == BrainModel.Task.label
        logits = self.activations_model(stimuli, layers=['logits'], number_of_trials=number_of_trials,
                                        require_variance=require_variance)
        choices = self.logits_to_choice(logits)
        return choices

    def logits_to_choice(self, logits):
        assert len(logits['neuroid']) == 1000
        logits = logits.transpose(..., 'neuroid')  # move neuroid dimension last
        extra_coords = {}
        if self.choice_labels == 'imagenet':
            # assuming the model was already trained on those labels, we just need to convert to synsets
            prediction_indices = logits.values.argmax(axis=1)
            with open(os.path.join(os.path.dirname(__file__), 'imagenet_classes.txt')) as f:
                synsets = f.read().splitlines()
            choices = [synsets[index] for index in prediction_indices]
            extra_coords['synset'] = ('presentation', choices)
            extra_coords['logit'] = ('presentation', prediction_indices)
        else:
            probabilities = softmax(logits)
            assert len(probabilities.dims) == 2 and probabilities.dims[-1] == 'neuroid'
            # map imagenet labels to target labels
            # from https://github.com/bethgelab/model-vs-human/blob/745046c4d82ff884af618756bd6a5f47b6f36c45/modelvshuman/datasets/decision_mappings.py#L30
            aggregated_class_probabilities = []
            for label in self.choice_labels:
                indices = LabelToImagenetIndices.label_to_indices(label)
                values = np.take(probabilities.values, indices, axis=-1)
                if 'kato2026_' in label:
                    aggregated_value = np.max(values, axis=-1)
                else:
                    aggregated_value = np.mean(values, axis=-1)
                aggregated_class_probabilities.append(aggregated_value)
            aggregated_class_probabilities = np.transpose(aggregated_class_probabilities)  # now presentation x p(label)
            top_indices = np.argmax(aggregated_class_probabilities, axis=1)
            choices = [self.choice_labels[top_index] for top_index in top_indices]

        coords = {**{coord: (dims, values) for coord, dims, values in walk_coords(logits['presentation'])},
                  **{'label': ('presentation', choices)},
                  **extra_coords}
        return BehavioralAssembly([choices], coords=coords, dims=['choice', 'presentation'])


LogitsBehavior = LabelBehavior  # LogitsBehavior is deprecated. Use LabelBehavior instead.
"""
legacy support; still used in old candidate_models submissions
https://github.com/brain-score/candidate_models/blob/fa965c452bd17c6bfcca5b991fdbb55fd5db618f/candidate_models/model_commitments/cornets.py#L13
"""


class LabelToImagenetIndices:
    airplane_indices = [404]
    bear_indices = [294, 295, 296, 297]
    bicycle_indices = [444, 671]
    bird_indices = [8, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 22, 23,
                    24, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92, 93,
                    94, 95, 96, 98, 99, 100, 127, 128, 129, 130, 131,
                    132, 133, 135, 136, 137, 138, 139, 140, 141, 142,
                    143, 144, 145]
    boat_indices = [472, 554, 625, 814, 914]
    bottle_indices = [440, 720, 737, 898, 899, 901, 907]
    car_indices = [436, 511, 817]
    cat_indices = [281, 282, 283, 284, 285, 286]
    chair_indices = [423, 559, 765, 857]
    clock_indices = [409, 530, 892]
    dog_indices = [152, 153, 154, 155, 156, 157, 158, 159, 160, 161,
                   162, 163, 164, 165, 166, 167, 168, 169, 170, 171,
                   172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
                   182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
                   193, 194, 195, 196, 197, 198, 199, 200, 201, 202,
                   203, 205, 206, 207, 208, 209, 210, 211, 212, 213,
                   214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
                   224, 225, 226, 228, 229, 230, 231, 232, 233, 234,
                   235, 236, 237, 238, 239, 240, 241, 243, 244, 245,
                   246, 247, 248, 249, 250, 252, 253, 254, 255, 256,
                   257, 259, 261, 262, 263, 265, 266, 267, 268]
    elephant_indices = [385, 386]
    keyboard_indices = [508, 878]
    knife_indices = [499]
    oven_indices = [766]
    truck_indices = [555, 569, 656, 675, 717, 734, 864, 867]

    # added from Baker et al. 2022:
    # cat and elephant indices as defined in Baker et al. 2022 are not used, instead we stick to the definition by Geirhos et al. 2021.
    # cat_indices = [281, 282, 283, 284, 285]
    # elephant_indices = [101, 385, 386]
    frog_indices = [30, 31, 32]
    lizard_indices = [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48]
    bunny_indices = [330, 331, 332]
    tiger_indices = [286, 287, 288, 289, 290, 291, 292, 293]
    turtle_indices = [33, 34, 35, 36, 37]
    wolf_indices = [269, 270, 271, 272, 273, 274, 275]

    # added from Zhu et al. 2019:
    aeroplane_indices = [404, 895]
    # car indices as defined in Zhu et al. 2019 are not used, instead we stick to the definition by Geirhos et al. 2021.
    # car_indices = [407, 436, 468, 511, 609, 627, 656, 661, 751, 817]
    motorbike_indices = [670, 665]
    bus_indices = [779, 874, 654]

    # added from the Scialom2024 benchmark:
    banana_indices = [954]
    beanie_indices = [439, 452, 515, 808]
    binoculars_indices = [447]
    boot_indices = [514]
    bowl_indices = [659, 809]
    cup_indices = [968]
    glasses_indices = [837]
    lamp_indices = [470, 607, 818, 846]
    pan_indices = [567]
    sewingmachine_indices = [786]
    shovel_indices = [792]
    # truck indices used as defined by Geirhos et al., 2021.
    
    # For Kato2026
    kato2026_corn_indices = [987, 998]
    kato2026_shoe_indices = [630, 770, 774]
    kato2026_bird_indices = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 
                             18, 19, 20, 21, 22, 23, 24, 80, 81, 82, 
                             83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 
                             93, 94, 95, 96, 97, 98, 99, 100, 127, 
                             128, 129, 130, 131, 132, 133, 134, 135, 
                             136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146]
    kato2026_monkey_indices = [370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382]
    kato2026_crocodilian_indices = [49, 50]
    kato2026_fish_indices = [0, 1, 2, 3, 4, 5, 6, 389, 390, 391, 392, 393, 394, 395, 396, 397]
    kato2026_acorn_indices = [988]
    kato2026_spider_indices = [72, 73, 74, 75, 76, 77]
    kato2026_hammer_indices = [587]
    kato2026_clock_indices = [409, 530, 892]
    kato2026_squirrel_indices = [335]
    kato2026_mug_indices = [504, 968]
    kato2026_snail_indices = [113]
    kato2026_turtle_indices = [33, 34, 35, 36, 37]
    kato2026_lobster_indices = [122, 123]
    kato2026_elephant_indices = [385, 386]
    kato2026_bear_indices = [294, 295, 296, 297]
    kato2026_horse_indices = [339]
    kato2026_swine_indices = [341, 342, 343]
    kato2026_chair_indices = [423, 559, 765, 857]
    kato2026_snake_indices = [52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68]
    kato2026_car_indices = [407, 436, 468, 511, 609, 627, 656, 661, 751, 817]
    kato2026_cat_indices = [281, 282, 283, 284, 285, 286, 287]
    kato2026_frog_indices = [30, 31, 32]
    kato2026_cabbage_indices = [936]
    kato2026_banana_indices = [954]
    kato2026_helmet_indices = [518, 560]
    kato2026_mushroom_indices = [947, 991, 992, 993, 994, 995, 996, 997]
    kato2026_apple_indices = [948]
    kato2026_dryer_indices = [589]
    kato2026_strawberry_indices = [949]
    kato2026_lemon_indices = [951]
    kato2026_pineapple_indices = [953]
    kato2026_flower_indices = [985, 986]
    kato2026_tiger_indices = [292]
    kato2026_microwave_indices = [651]
    kato2026_dog_indices = [151, 152, 153, 154, 155, 156, 157, 158, 159, 
                            160, 161, 162, 163, 164, 165, 166, 167, 168, 
                            169, 170, 171, 172, 173, 174, 175, 176, 177, 
                            178, 179, 180, 181, 182, 183, 184, 185, 186, 
                            187, 188, 189, 190, 191, 192, 193, 194, 195, 
                            196, 197, 198, 199, 200, 201, 202, 203, 204, 
                            205, 206, 207, 208, 209, 210, 211, 212, 213, 
                            214, 215, 216, 217, 218, 219, 220, 221, 222, 
                            223, 224, 225, 226, 227, 228, 229, 230, 231, 
                            232, 233, 234, 235, 236, 237, 238, 239, 240, 
                            241, 242, 243, 244, 245, 246, 247, 248, 249, 
                            250, 251, 252, 253, 254, 255, 256, 257, 258, 
                            259, 260, 261, 262, 263, 264, 265, 266, 267, 268]
    kato2026_butterfly_indices = [321, 322, 323, 324, 325, 326]
    kato2026_gun_indices = [413, 471, 763, 764]
    kato2026_bellpepper_indices = [945]
    kato2026_rabbit_indices = [330, 331, 332]
    kato2026_hippopotamus_indices = [344]
    kato2026_bee_indices = [309]
    kato2026_airplane_indices = [404]
    kato2026_camel_indices = [354]
    kato2026_guitar_indices = [402, 546]
    kato2026_crab_indices = [118, 119, 120, 121]
    kato2026_broccoli_indices = [937]
    kato2026_others_indices = [25, 26, 27, 28, 29, 38, 39, 40, 41, 42, 43, 44, 45, 46,
                       47, 48, 51, 69, 70, 71, 78, 79, 101, 102, 103, 104, 105,
                       106, 107, 108, 109, 110, 111, 112, 114, 115, 116, 117, 
                       124, 125, 126, 147, 148, 149, 150, 269, 270, 271, 272, 
                       273, 274, 275, 276, 277, 278, 279, 280, 288, 289, 290, 
                       291, 293, 298, 299, 300, 301, 302, 303, 304, 305, 306, 
                       307, 308, 310, 311, 312, 313, 314, 315, 316, 317, 318, 
                       319, 320, 327, 328, 329, 333, 334, 336, 337, 338, 340, 
                       345, 346, 347, 348, 349, 350, 351, 352, 353, 355, 356, 
                       357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 
                       368, 369, 383, 384, 387, 388, 398, 399, 400, 401, 403, 
                       405, 406, 408, 410, 411, 412, 414, 415, 416, 417, 418, 
                       419, 420, 421, 422, 424, 425, 426, 427, 428, 429, 430, 
                       431, 432, 433, 434, 435, 437, 438, 439, 440, 441, 442, 
                       443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 
                       454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 
                       465, 466, 467, 469, 470, 472, 473, 474, 475, 476, 477, 
                       478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 
                       489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 
                       500, 501, 502, 503, 505, 506, 507, 508, 509, 510, 512, 
                       513, 514, 515, 516, 517, 519, 520, 521, 522, 523, 524, 
                       525, 526, 527, 528, 529, 531, 532, 533, 534, 535, 536, 
                       537, 538, 539, 540, 541, 542, 543, 544, 545, 547, 548, 
                       549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 561, 
                       562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 
                       573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 
                       584, 585, 586, 588, 590, 591, 592, 593, 594, 595, 596, 
                       597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 
                       608, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 
                       620, 621, 622, 623, 624, 625, 626, 628, 629, 631, 632, 
                       633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 
                       644, 645, 646, 647, 648, 649, 650, 652, 653, 654, 655, 
                       657, 658, 659, 660, 662, 663, 664, 665, 666, 667, 668, 
                       669, 670, 671, 672, 673, 674, 675, 676, 677, 678, 679, 
                       680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 
                       691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 
                       702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 
                       713, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 
                       724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 
                       735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 
                       746, 747, 748, 749, 750, 752, 753, 754, 755, 756, 757, 
                       758, 759, 760, 761, 762, 766, 767, 768, 769, 771, 772, 
                       773, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 
                       785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 
                       796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 
                       807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 818, 
                       819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 
                       830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 
                       841, 842, 843, 844, 845, 846, 847, 848, 849, 850, 851, 
                       852, 853, 854, 855, 856, 858, 859, 860, 861, 862, 863, 
                       864, 865, 866, 867, 868, 869, 870, 871, 872, 873, 874, 
                       875, 876, 877, 878, 879, 880, 881, 882, 883, 884, 885, 
                       886, 887, 888, 889, 890, 891, 893, 894, 895, 896, 897, 
                       898, 899, 900, 901, 902, 903, 904, 905, 906, 907, 908, 
                       909, 910, 911, 912, 913, 914, 915, 916, 917, 918, 919, 
                       920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 
                       931, 932, 933, 934, 935, 938, 939, 940, 941, 942, 943, 
                       944, 946, 950, 952, 955, 956, 957, 958, 959, 960, 961, 
                       962, 963, 964, 965, 966, 967, 969, 970, 971, 972, 973, 
                       974, 975, 976, 977, 978, 979, 980, 981, 982, 983, 984, 989, 990, 999]

    @classmethod
    def label_to_indices(cls, label):
        # for handling multi-word labels given by models or benchmarks
        label = label.lower().replace(" ", "")

        synset_indices = getattr(cls, f"{label}_indices")
        return synset_indices


def softmax(x):
    return np.exp(x) / np.exp(x).sum(dim='neuroid')


class ProbabilitiesMapping(BrainModel):
    def __init__(self, identifier, activations_model, layer):
        """
        :param identifier: a string to identify the model
        :param activations_model: the model from which to retrieve representations for stimuli
        :param layer: the single behavioral readout layer or a list of layers to read out of.
        """
        self._identifier = identifier
        self.activations_model = activations_model
        self.readout = make_list(layer)
        self.classifier = ProbabilitiesMapping.ProbabilitiesClassifier()
        self.current_task = None

    @property
    def identifier(self):
        return self._identifier

    def start_task(self, task: BrainModel.Task, fitting_stimuli, number_of_trials=1, require_variance=False):
        assert task in [BrainModel.Task.passive, BrainModel.Task.probabilities]
        self.current_task = task

        fitting_features = self.activations_model(fitting_stimuli, layers=self.readout,
                                                  number_of_trials=number_of_trials,
                                                  require_variance=require_variance)
        fitting_features = fitting_features.transpose('presentation', 'neuroid')
        assert all(self.order_preserving_unique(fitting_features['stimulus_id'].values) == fitting_stimuli['stimulus_id'].values), \
            "stimulus_id ordering is incorrect"
        self.classifier.fit(fitting_features, fitting_features['image_label'])

    def look_at(self, stimuli, number_of_trials=1, require_variance=False):
        if self.current_task is BrainModel.Task.passive:
            return
        features = self.activations_model(stimuli, layers=self.readout, number_of_trials=number_of_trials,
                                          require_variance=require_variance)
        features = features.transpose('presentation', 'neuroid')
        prediction = self.classifier.predict_proba(features)
        return prediction

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

    @staticmethod
    def order_preserving_unique(array):
        """
        This function sorts an array and removes duplicates while preserving the order of the elements.
        This function is used in favor of np.unique to ensure that the order of the stimulus_ids is preserved, as
        np.unique performs sorting on the array.
        """
        _, indices = np.unique(array, return_index=True)
        return array[np.sort(indices)]


class OddOneOut(BrainModel):
    def __init__(self, identifier: str, activations_model, layer: Union[str, List[str]]):
        """
        :param identifier: a string to identify the model
        :param activations_model: the model from which to retrieve representations for stimuli
        :param layer: the single behavioral readout layer or a list of layers to read out of.
        """
        self._identifier = identifier
        self.activations_model = activations_model
        self.readout = make_list(layer)
        self.current_task = BrainModel.Task.odd_one_out
        self.similarity_measure = 'dot'

    @property
    def identifier(self):
        return self._identifier

    def start_task(self, task: BrainModel.Task):
        assert task == BrainModel.Task.odd_one_out
        self.current_task = task

    def look_at(self, triplets, number_of_trials: int = 1, require_variance: bool = False):
        # Compute unique features and image_paths
        stimuli = triplets.drop_duplicates(subset=['stimulus_id'])
        stimuli = stimuli.sort_values(by='stimulus_id')

        # Get features
        features = self.activations_model(stimuli, layers=self.readout, require_variance=require_variance)
        features = features.transpose('presentation', 'neuroid')

        # Compute similarity matrix
        similarity_matrix = self.calculate_similarity_matrix(features)

        # Compute choices
        triplets = np.array(triplets["stimulus_id"])
        assert len(triplets) % 3 == 0, "No. of stimuli must be a multiple of 3"
        choices = self.calculate_choices(similarity_matrix, triplets)

        # Package choices
        stimulus_ids = ['|'.join([f"{triplets[offset + i]}" for i in range(3)])
                        for offset in range(0, len(triplets) - 2, 3)]
        choices = BehavioralAssembly(
            [choices],
            coords={'stimulus_id': ('presentation', stimulus_ids)},
            dims=['choice', 'presentation'])

        return choices

    def set_similarity_measure(self, similarity_measure):
        self.similarity_measure = similarity_measure

    def calculate_similarity_matrix(self, features):
        features = features.transpose('presentation', 'neuroid')
        values = features.values
        if self.similarity_measure == 'dot':
            similarity_matrix = np.dot(values, np.transpose(values))
        elif self.similarity_measure == 'cosine':
            row_norms = np.linalg.norm(values, axis=1).reshape(-1, 1)
            norm_product = np.dot(row_norms, row_norms.T)
            dot_product = np.dot(values, np.transpose(values))
            similarity_matrix = dot_product / norm_product
        else:
            raise ValueError(
                f"Unknown similarity_measure {self.similarity_measure} -- expected one of 'dot' or 'cosine'")

        similarity_matrix = DataAssembly(similarity_matrix, coords={
            **{f"{coord}_left": ('presentation_left', values) for coord, dims, values in
               walk_coords(features) if array_is_element(dims, 'presentation')},
            **{f"{coord}_right": ('presentation_right', values) for coord, dims, values in
               walk_coords(features) if array_is_element(dims, 'presentation')}
        }, dims=['presentation_left', 'presentation_right'])
        return similarity_matrix

    def calculate_choices(self, similarity_matrix, triplets):
        triplets = np.array(triplets).reshape(-1, 3)
        # indexing via `.sel(stimulus_id_left=..., stimulus_id_right=...)` is slow.
        # To speed this up, we pre-index all stimulus ids so that we can reference directly into the .values array.
        stimulusid_index = {}
        for leftright in ['left', 'right']:
            for index, stimulus_id in enumerate(similarity_matrix[f'stimulus_id_{leftright}'].values):
                stimulusid_index[(leftright, stimulus_id)] = index
        choice_predictions = []
        for triplet in triplets:
            i, j, k = triplet
            i_index_left = stimulusid_index[('left', i)]
            j_index_right = stimulusid_index[('right', j)]
            j_index_left = stimulusid_index[('left', j)]
            k_index_right = stimulusid_index[('right', k)]
            sims = [similarity_matrix.values[i_index_left, j_index_right].item(),
                    similarity_matrix.values[i_index_left, k_index_right].item(),
                    similarity_matrix.values[j_index_left, k_index_right].item()]
            idx = triplet[2 - np.argmax(sims)]
            choice_predictions.append(idx)
        return choice_predictions
