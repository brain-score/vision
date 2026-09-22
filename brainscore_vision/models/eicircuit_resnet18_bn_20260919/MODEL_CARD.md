# eicircuit_resnet18_bn_20260919

ResNet-18 with batch normalization, using the saved running statistics in evaluation mode.

- Training data: ImageNet-1k training split.
- Checkpoint: epoch 88; clean ImageNet validation top-1: 70.664%.
- Parameters: 11,689,512.
- Frozen checkpoint; no brain-data training or fine-tuning.
- RGB input; resize shorter edge to 256 pixels using bilinear interpolation, center crop to 224 by 224, convert uint8 to float32 in [0, 1]. No channel mean/std normalization.
- Nine candidate neural layers: stem max-pool and the output of each of the eight residual blocks, identical choices for EI and BN.
- Brain-Score default layer selection determines region assignments using its public selection benchmarks. No region assignments are claimed in advance.
- Behavioral feature readout: features.avgpool (512 features); ImageNet labels use the original 1,000-way classifier.
- Visual degrees: 8 (conventional interface setting, not fitted).
- Computation is feedforward. This model does not implement recurrent neural dynamics.

## Weights

The code ZIP excludes the checkpoint to comply with the website's 50 MB limit.
Upload `eicircuit_resnet18_bn_20260919_weights.pt` through your Brain-Score account's Large File Upload page.
Set `folder_name` and `version_id` in `weights_config.json` to the values supplied by that page before uploading this code ZIP. The bucket is `brainscore-storage`.
The local file `weights.pt` beside `model.py` can be used for offline verification.
This export contains a tensor state_dict only, without optimizer state or training data.

## Validation

The independent PyTorch export matched the original training implementation exactly at all nine candidate layers, average pooling, and logits on a two-image CPU smoke test (maximum absolute difference 0).
Brain-Score interface validation is described in the separate bundle validation report. It is not a benchmark score.
