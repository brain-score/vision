from candidate_models.model_commitments.cornets import CORnetCommitment


class VOneRecurrenceCommitment(CORnetCommitment):
    def start_recording(self, recording_target, time_bins):
        self.recording_target = recording_target
        if recording_target == 'V1':
            self.recording_layers = ['VOneBlock.output']
        else:
            self.recording_layers = [
                layer for layer in self.layers if recording_target in layer]
        self.recording_time_bins = time_bins
