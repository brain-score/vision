import os

from brainscore.model_interface import BrainModel


class LogitsBehavior(BrainModel):
    def __init__(self, identifier, activations_model):
        self.identifier = identifier
        self.activations_model = activations_model
        self.current_task = None

    def start_task(self, task: BrainModel.Task):
        assert task in [BrainModel.Task.passive, BrainModel.Task.probabilities]
        self.current_task = task

    def look_at(self, stimuli):
        softmax = self.activations_model(stimuli, layers=['logits'])
        assert len(softmax['neuroid']) == 1000
        softmax = softmax.transpose('presentation', 'neuroid')
        prediction_indices = softmax.values.argmax(axis=1)
        with open(os.path.join(os.path.dirname(__file__), 'imagenet_classes.txt')) as f:
            synsets = f.read().splitlines()
        prediction_synsets = [synsets[index] for index in prediction_indices]
        return prediction_synsets
