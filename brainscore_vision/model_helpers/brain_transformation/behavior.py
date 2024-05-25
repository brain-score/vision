import os
from collections import OrderedDict
from typing import Union, List

import numpy as np
import pandas as pd
import xarray as xr
import sklearn.linear_model
import sklearn.multioutput

from brainio.assemblies import walk_coords, array_is_element, BehavioralAssembly, DataAssembly
from brainio.stimuli import StimulusSet
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

    def look_at(self, stimuli, number_of_trials=1):
        assert self.current_task == BrainModel.Task.label
        logits = self.activations_model(stimuli, layers=['logits'])
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

    @classmethod
    def label_to_indices(cls, label):
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

    def start_task(self, task: BrainModel.Task, fitting_stimuli):
        assert task in [BrainModel.Task.passive, BrainModel.Task.probabilities]
        self.current_task = task

        fitting_features = self.activations_model(fitting_stimuli, layers=self.readout)
        fitting_features = fitting_features.transpose('presentation', 'neuroid')
        assert all(fitting_features['stimulus_id'].values == fitting_stimuli['stimulus_id'].values), \
            "stimulus_id ordering is incorrect"
        self.classifier.fit(fitting_features, fitting_stimuli['image_label'])

    def look_at(self, stimuli, number_of_trials=1):
        if self.current_task is BrainModel.Task.passive:
            return
        features = self.activations_model(stimuli, layers=self.readout)
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

    def look_at(self, triplets, number_of_trials=1):
        # Compute unique features and image_pathst
        stimuli = triplets.drop_duplicates(subset=['stimulus_id'])
        stimuli = stimuli.sort_values(by='stimulus_id')

        # Get features
        features = self.activations_model(stimuli, layers=self.readout)
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
            **{f"{coord}_left": ('presentation_left', values) for coord, _, values in
               walk_coords(features['presentation'])},
            **{f"{coord}_right": ('presentation_right', values) for coord, _, values in
               walk_coords(features['presentation'])}
        }, dims=['presentation_left', 'presentation_right'])
        return similarity_matrix

    def calculate_choices(self, similarity_matrix, triplets):
        triplets = np.array(triplets).reshape(-1, 3)
        choice_predictions = []
        for triplet in triplets:
            i, j, k = triplet
            sims = [similarity_matrix.sel(stimulus_id_left=i, stimulus_id_right=j),
                    similarity_matrix.sel(stimulus_id_left=i, stimulus_id_right=k),
                    similarity_matrix.sel(stimulus_id_left=j, stimulus_id_right=k)]
            idx = triplet[2 - np.argmax(sims)]
            choice_predictions.append(idx)
        return choice_predictions
    

class ReadoutMapping(BrainModel):
    def __init__(self, identifier, activations_model, layer):
        """
        :param identifier: a string to identify the model
        :param activations_model: the model from which to retrieve representations for stimuli
        :param layer: the single behavioral readout layer or a list of layers to read out of.
        """
        self._identifier = identifier
        self.activations_model = activations_model
        self.readout = make_list(layer)
        self.classifier = ReadoutMapping.TransformerReadout()
        self.current_task = None

    @property
    def identifier(self):
        return self._identifier

    def start_task(self, task: BrainModel.Task, fitting_stimuli):
        assert task in BrainModel.Task.label
        self.current_task = task

        fitting_features = self.activations_model(fitting_stimuli['data'], layers=self.readout)
        fitting_features = fitting_features.transpose('presentation', 'neuroid')
        assert all(fitting_features['stimulus_id'].values == fitting_stimuli['stimulus_id'].values), \
            "stimulus_id ordering is incorrect"
        self.classifier.fit(fitting_features, fitting_stimuli['label'], fitting_features['stimulus_id'])

    def look_at(self, stimuli, number_of_trials=1):
        features = self.activations_model(stimuli, layers=self.readout)
        features = features.transpose('presentation', 'neuroid')
        prediction = self.classifier.predict(features)
        return prediction
        
    class TransformerReadout:
        def __init__(self, model_dim):
            self.model = ReadoutMapping.ReadoutModel(model_dim, 4, 1)
            self.num_epochs = 1000
            self.lr = 1e-4
            self.best_val_accuracy = 0
            self.convergence_thresh = 20
            self.counter_converge = 0
            self.prob_threshold = 0.5
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
            self.set_seed(43)
         
        def set_seed(self, seed_value=42):
            """Set seed for reproducibility."""
            torch.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)  # For CUDA devices
            np.random.seed(seed_value)
            random.seed(seed_value)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
        def build_loader(features, labels, mode='train', indices=None):
            train_dataset = ReadoutMapping.TransformerLoader(features, labels, indices=train_indices)
            sampler = ReadoutMapping.BalancedBatchSampler(train_dataset.positive_indices, 
                                           train_dataset.negative_indices, 
                                           batch_size=256, seed=42)

            train_loader = MultiEpochsDataLoader(train_dataset, 
                                                 batch_sampler=sampler, 
                                                 num_workers=4)
            if mode == 'train':
                val_dataset = ReadoutMapping.TransformerLoader(features, labels, indices=val_indices)
                val_loader = MultiEpochsDataLoader(val_dataset, 
                                               batch_size=256, 
                                               shuffle=False, 
                                               num_workers=4)
            else:
                val_dataset = None

            return train_loader, val_loader
            
        def fit(self, features, labels):
            train_loader, val_loader = self.build_loader(features, labels, mode='train')
            self.model = nn.DataParallel(self.model)
            self.model = self.model.to(self.device)

            logging.info(f'Total number of parameters: {total_params}')
            # Optimizer
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
            criterion = nn.BCELoss()
           
            # Define the total number of steps
            total_steps = num_epochs * len(train_loader)
            warmup_steps = int(0.05 * total_steps)  # for example, 10% of total steps

            # Initialize the warmup scheduler
            warmup_scheduler = WarmupScheduler(optimizer, warmup_steps, initial_lr=lr)

            # Replace the ReduceLROnPlateau scheduler with CosineAnnealingLR
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=0)

            for epoch in range(num_epochs):
                train_loss, train_acc = self.train(train_loader, optimizer, criterion, self.device,
                                                   epoch, scheduler, warmup_scheduler, warmup_steps)
                # optimize for prob
                if epoch % val_after == 0:
                    val_accuracy = self.validate(val_loader)  
                    logging.info(f'Epoch:{epoch+1}, Val Accuracy:{val_accuracy*100:.2f}%')

                    if val_accuracy > self.best_val_accuracy:
                        self.counter_converge = 0
                        self.best_val_accuracy = val_accuracy
                        logging.info(f'Saving best model with val accuracy:{val_accuracy:.5f}')
                        torch.save(self.model.state_dict(), 'transformer_readout.pt')
                    else:
                        self.counter_converge += 1

                if self.counter_converge >= self.convergence_thresh:
                    break
                
        def binary_accuracy(self, preds, y):
            # Round predictions to the closest integer (0 or 1)
            rounded_preds = (preds > self.prob_threshold).float()
            correct = (rounded_preds == y).float()  # Convert into float for division
            acc = correct.sum() / len(correct)
            return acc
            
        def train(self, data_loader, optimizer, criterion, 
                  epoch, scheduler, warmup_scheduler,
                  warmup_steps, log_step=50):
            self.model.train()
            total_loss = 0
            total_acc = 0

            for batch_idx, data in enumerate(data_loader):
                # Warmup for the initial warmup_steps
                if epoch * len(data_loader) + batch_idx < warmup_steps:
                    warmup_scheduler.step()
                else:
                    # Once the warmup steps are completed, use cosine annealing
                    scheduler.step(epoch * len(data_loader) + batch_idx - warmup_steps)

                inputs, labels = data['feature'], data['label']
                inputs, targets = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs, _ = self.model(inputs)  # Ensure the output is of correct shape
                outputs = outputs.squeeze()
                loss = criterion(outputs, targets.float())
                acc = self.binary_accuracy(outputs, targets)

                if batch_idx % log_step == 0:
                    logging.info(f'Epoch:{epoch+1}, Step: [{batch_idx}/{len(data_loader)}], Train Accuracy:{acc:.5f}')

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_acc += acc.item()

            # Calculate the average loss and accuracy over all batches
            avg_loss = total_loss / len(data_loader)
            avg_acc = total_acc / len(data_loader)

            return avg_loss, avg_acc
        
        def validate(val_loader):
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data in val_loader:
                    inputs, labels = data['feature'], data['label']
                    inputs, labels = inputs.to(device), labels.to(device)  
                    outputs, _ = self.model(inputs)
                    outputs = outputs.squeeze()
                    predicted = (outputs > self.prob_threshold).float() 
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            self.model.train()
            return correct / total
        
        def predict(features, labels):
            test_loader, _ = self.build_loader(features, labels, mode='test')
            self.model.load_state_dict(torch.load('transformer_readout.pt'))
            self.model.eval()  # Set the model to evaluation mode
            predictions, proba = [], []

            with torch.no_grad():  # No gradients needed
                for data in test_loader:
                    inputs, labels = data['feature'], data['label']
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs, _ = model(inputs)
                    outputs = outputs.squeeze()
                    proba.extend(outputs.tolis())
                    predicted = (outputs > self.prob_threshold).float().tolist()
                    predictions.extend(predicted)
                    
            features_coords = {coord: (dims, value) for coord, dims, value in walk_coords(features)
                        if array_is_element(dims, features.dims[0])}
            proba = BehavioralAssembly(proba,
                                       coords={**features_coords, **{'choice': predictions}},
                                       dims=[features.dims[0], 'choice'])

            return predictions
            
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000):
            super(PositionalEncoding, self).__init__()
            self.d_model = d_model

        def forward(self, x):
            device = x.device
            # Assuming x has shape [batch_size, seq_length, d_model] = [9, 10, 128] in this context
            seq_length = x.size(1)

            position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1).to(device)
            div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)).to(device)

            encoding = torch.zeros(seq_length, self.d_model, device=device)
            encoding[:, 0::2] = torch.sin(position * div_term.unsqueeze(0))
            encoding[:, 1::2] = torch.cos(position * div_term.unsqueeze(0))

            encoding = encoding.unsqueeze(0).expand(x.size(0), -1, -1)  # Expand encoding to match the batch size of x
            return x + encoding
        
    class ReadoutModel(nn.Module):
        def __init__(self, model_dim, num_heads, 
                     num_encoder_layers, num_classes=1, num_input_layers=2):
            super(ReadoutModel, self).__init__()

            self.model_dim = model_dim
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=128)
            self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
            self.position_encoder = ReadoutMapping.PositionalEncoding(model_dim)
            self.fc = nn.Linear(model_dim, num_classes)

        def forward(self, x, mode=None):
            x = x.view(x.shape[0], x.shape[1], -1)  # Flatten keeping the last dimension
            x = self.position_encoder(x)
            x = self.transformer_encoder(x)
            x = x.mean(dim=1)  # Pooling, consider masking padded values
            x = self.fc(x)
            return torch.sigmoid(x), None

    class TransformerLoader(Dataset):
        def __init__(self, features, labels, indices=None):
            self.indices = indices
            self.labels = labels
            self.features = features
            self.positive_indices, self.negative_indices = [], []
            for i, l in enumerate(labels):
                if l == 1:
                    self.positive_indices += [i]
                else:
                    self.negative_indices += [i]

        def __len__(self):
            length = len(self.indices) if self.indices is not None else self.features.shape[0]
            return length

        def __getitem__(self, idx):
            actual_idx = self.indices[idx] if self.indices is not None else idx
            feature = self.features[actual_idx]
            label = self.labels[actual_idx]

            return {
                'feature': feature,
                'label': label,
            }
        
    class BalancedBatchSampler(Sampler):
        def __init__(self, positive_indices, negative_indices, batch_size, seed=None):
            self.positive_indices = np.array(positive_indices)
            self.negative_indices = np.array(negative_indices)
            self.batch_size = batch_size
            self.seed = seed
            assert batch_size % 2 == 0, "Batch size must be even."

            # If a seed is provided, use it for random operations
            if self.seed is not None:
                np.random.seed(self.seed)
                torch.manual_seed(self.seed)

        def __iter__(self):
            np.random.shuffle(self.positive_indices)
            np.random.shuffle(self.negative_indices)
            n_batches = min(len(self.positive_indices), len(self.negative_indices)) // (self.batch_size // 2)

            for i in range(n_batches):
                batch_indices = np.concatenate([
                    self.positive_indices[i*(self.batch_size//2):(i+1)*(self.batch_size//2)],
                    self.negative_indices[i*(self.batch_size//2):(i+1)*(self.batch_size//2)]
                ])
                np.random.shuffle(batch_indices)
                yield batch_indices

        def __len__(self):
            return min(len(self.positive_indices), len(self.negative_indices)) // (self.batch_size // 2)
        
    class MultiEpochsDataLoader(torch.utils.data.DataLoader):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._DataLoader__initialized = False
            self.batch_sampler = _RepeatSampler(self.batch_sampler)
            self._DataLoader__initialized = True
            self.iterator = super().__iter__()

        def __len__(self):
            return len(self.batch_sampler.sampler)

        def __iter__(self):
            for i in range(len(self)):
                yield next(self.iterator)