import torch as th

from brainscore_vision.model_helpers.activations.temporal.model import PytorchWrapper
from brainscore_vision.model_helpers.s3 import load_weight_file
from resnet_model import pfMAE_LSTM_physion, load_model

from torchvision import transforms

class MAELSTMWrapper(PytorchWrapper):
    def forward(self, inputs):
        tensor = th.stack(inputs)
        tensor = tensor.to(self._device)
        with torch.no_grad():
            output = self._model(tensor)
        features = output["input_states"]
        return features  # encoder only

transform_img = T.Compose([T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor()]) # ToTensor() divides by 255

def transform_video(video):
    import torch
    frames = torch.Tensor(video.to_numpy()).permute(0, 3, 1, 2)
    frames = transform_img(frames)
    return frames.permute(1, 0, 2, 3)


def get_model(identifier, num_frames=7):
    assert identifier.startswith("MAE-LSTM")
    # Instantiate the model
    
    net = pfMAE_LSTM_physion(n_past=num_frames, full_rollout=False)
    net = load_model(net, model_path)

    inferencer_kwargs = {
        "fps": 16,
        "layer_activation_format": {
            "model.dynamics": "TC",
        },
        "num_frames": num_frames,
    }

    for layer in inferencer_kwargs["layer_activation_format"].keys():
        assert "decoder" not in layer, "Decoder layers are not supported."

    wrapper = MAELSTMWrapper(identifier, net, transform_video, 
                                process_output=None,
                                **inferencer_kwargs)
    return wrapper
