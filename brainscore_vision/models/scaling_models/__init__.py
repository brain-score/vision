from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, MODEL_CONFIGS

model_registry["resnet18_imagenet_full"] = lambda: ModelCommitment(
    identifier="resnet18_imagenet_full",
    activations_model=get_model("resnet18_imagenet_full"),
    layers=MODEL_CONFIGS["resnet18_imagenet_full"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["resnet18_imagenet_full"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["resnet18_imagenet_full"]["model_commitment"]["region_layer_map"]
)


model_registry["resnet34_imagenet_full"] = lambda: ModelCommitment(
    identifier="resnet34_imagenet_full",
    activations_model=get_model("resnet34_imagenet_full"),
    layers=MODEL_CONFIGS["resnet34_imagenet_full"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["resnet34_imagenet_full"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["resnet34_imagenet_full"]["model_commitment"]["region_layer_map"]
)


model_registry["resnet50_imagenet_full"] = lambda: ModelCommitment(
    identifier="resnet50_imagenet_full",
    activations_model=get_model("resnet50_imagenet_full"),
    layers=MODEL_CONFIGS["resnet50_imagenet_full"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["resnet50_imagenet_full"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["resnet50_imagenet_full"]["model_commitment"]["region_layer_map"]
)


model_registry["resnet101_imagenet_full"] = lambda: ModelCommitment(
    identifier="resnet101_imagenet_full",
    activations_model=get_model("resnet101_imagenet_full"),
    layers=MODEL_CONFIGS["resnet101_imagenet_full"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["resnet101_imagenet_full"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["resnet101_imagenet_full"]["model_commitment"]["region_layer_map"]
)


model_registry["resnet152_imagenet_full"] = lambda: ModelCommitment(
    identifier="resnet152_imagenet_full",
    activations_model=get_model("resnet152_imagenet_full"),
    layers=MODEL_CONFIGS["resnet152_imagenet_full"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["resnet152_imagenet_full"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["resnet152_imagenet_full"]["model_commitment"]["region_layer_map"]
)


model_registry["resnet18_ecoset_full"] = lambda: ModelCommitment(
    identifier="resnet18_ecoset_full",
    activations_model=get_model("resnet18_ecoset_full"),
    layers=MODEL_CONFIGS["resnet18_ecoset_full"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["resnet18_ecoset_full"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["resnet18_ecoset_full"]["model_commitment"]["region_layer_map"]
)


model_registry["resnet34_ecoset_full"] = lambda: ModelCommitment(
    identifier="resnet34_ecoset_full",
    activations_model=get_model("resnet34_ecoset_full"),
    layers=MODEL_CONFIGS["resnet34_ecoset_full"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["resnet34_ecoset_full"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["resnet34_ecoset_full"]["model_commitment"]["region_layer_map"]
)


model_registry["resnet50_ecoset_full"] = lambda: ModelCommitment(
    identifier="resnet50_ecoset_full",
    activations_model=get_model("resnet50_ecoset_full"),
    layers=MODEL_CONFIGS["resnet50_ecoset_full"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["resnet50_ecoset_full"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["resnet50_ecoset_full"]["model_commitment"]["region_layer_map"]
)


model_registry["resnet101_ecoset_full"] = lambda: ModelCommitment(
    identifier="resnet101_ecoset_full",
    activations_model=get_model("resnet101_ecoset_full"),
    layers=MODEL_CONFIGS["resnet101_ecoset_full"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["resnet101_ecoset_full"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["resnet101_ecoset_full"]["model_commitment"]["region_layer_map"]
)


model_registry["resnet152_ecoset_full"] = lambda: ModelCommitment(
    identifier="resnet152_ecoset_full",
    activations_model=get_model("resnet152_ecoset_full"),
    layers=MODEL_CONFIGS["resnet152_ecoset_full"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["resnet152_ecoset_full"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["resnet152_ecoset_full"]["model_commitment"]["region_layer_map"]
)


model_registry["resnet50_imagenet_1_seed-0"] = lambda: ModelCommitment(
    identifier="resnet50_imagenet_1_seed-0",
    activations_model=get_model("resnet50_imagenet_1_seed-0"),
    layers=MODEL_CONFIGS["resnet50_imagenet_1_seed-0"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["resnet50_imagenet_1_seed-0"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["resnet50_imagenet_1_seed-0"]["model_commitment"]["region_layer_map"]
)


model_registry["resnet50_imagenet_10_seed-0"] = lambda: ModelCommitment(
    identifier="resnet50_imagenet_10_seed-0",
    activations_model=get_model("resnet50_imagenet_10_seed-0"),
    layers=MODEL_CONFIGS["resnet50_imagenet_10_seed-0"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["resnet50_imagenet_10_seed-0"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["resnet50_imagenet_10_seed-0"]["model_commitment"]["region_layer_map"]
)


model_registry["resnet50_imagenet_100_seed-0"] = lambda: ModelCommitment(
    identifier="resnet50_imagenet_100_seed-0",
    activations_model=get_model("resnet50_imagenet_100_seed-0"),
    layers=MODEL_CONFIGS["resnet50_imagenet_100_seed-0"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["resnet50_imagenet_100_seed-0"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["resnet50_imagenet_100_seed-0"]["model_commitment"]["region_layer_map"]
)


model_registry["efficientnet_b0_imagenet_full"] = lambda: ModelCommitment(
    identifier="efficientnet_b0_imagenet_full",
    activations_model=get_model("efficientnet_b0_imagenet_full"),
    layers=MODEL_CONFIGS["efficientnet_b0_imagenet_full"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["efficientnet_b0_imagenet_full"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["efficientnet_b0_imagenet_full"]["model_commitment"]["region_layer_map"]
)


model_registry["efficientnet_b1_imagenet_full"] = lambda: ModelCommitment(
    identifier="efficientnet_b1_imagenet_full",
    activations_model=get_model("efficientnet_b1_imagenet_full"),
    layers=MODEL_CONFIGS["efficientnet_b1_imagenet_full"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["efficientnet_b1_imagenet_full"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["efficientnet_b1_imagenet_full"]["model_commitment"]["region_layer_map"]
)


model_registry["efficientnet_b2_imagenet_full"] = lambda: ModelCommitment(
    identifier="efficientnet_b2_imagenet_full",
    activations_model=get_model("efficientnet_b2_imagenet_full"),
    layers=MODEL_CONFIGS["efficientnet_b2_imagenet_full"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["efficientnet_b2_imagenet_full"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["efficientnet_b2_imagenet_full"]["model_commitment"]["region_layer_map"]
)


model_registry["deit_small_imagenet_full_seed-0"] = lambda: ModelCommitment(
    identifier="deit_small_imagenet_full_seed-0",
    activations_model=get_model("deit_small_imagenet_full_seed-0"),
    layers=MODEL_CONFIGS["deit_small_imagenet_full_seed-0"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["deit_small_imagenet_full_seed-0"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["deit_small_imagenet_full_seed-0"]["model_commitment"]["region_layer_map"]
)


model_registry["deit_base_imagenet_full_seed-0"] = lambda: ModelCommitment(
    identifier="deit_base_imagenet_full_seed-0",
    activations_model=get_model("deit_base_imagenet_full_seed-0"),
    layers=MODEL_CONFIGS["deit_base_imagenet_full_seed-0"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["deit_base_imagenet_full_seed-0"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["deit_base_imagenet_full_seed-0"]["model_commitment"]["region_layer_map"]
)


model_registry["deit_large_imagenet_full_seed-0"] = lambda: ModelCommitment(
    identifier="deit_large_imagenet_full_seed-0",
    activations_model=get_model("deit_large_imagenet_full_seed-0"),
    layers=MODEL_CONFIGS["deit_large_imagenet_full_seed-0"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["deit_large_imagenet_full_seed-0"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["deit_large_imagenet_full_seed-0"]["model_commitment"]["region_layer_map"]
)


model_registry["deit_small_imagenet_1_seed-0"] = lambda: ModelCommitment(
    identifier="deit_small_imagenet_1_seed-0",
    activations_model=get_model("deit_small_imagenet_1_seed-0"),
    layers=MODEL_CONFIGS["deit_small_imagenet_1_seed-0"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["deit_small_imagenet_1_seed-0"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["deit_small_imagenet_1_seed-0"]["model_commitment"]["region_layer_map"]
)


model_registry["deit_small_imagenet_10_seed-0"] = lambda: ModelCommitment(
    identifier="deit_small_imagenet_10_seed-0",
    activations_model=get_model("deit_small_imagenet_10_seed-0"),
    layers=MODEL_CONFIGS["deit_small_imagenet_10_seed-0"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["deit_small_imagenet_10_seed-0"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["deit_small_imagenet_10_seed-0"]["model_commitment"]["region_layer_map"]
)


model_registry["deit_small_imagenet_100_seed-0"] = lambda: ModelCommitment(
    identifier="deit_small_imagenet_100_seed-0",
    activations_model=get_model("deit_small_imagenet_100_seed-0"),
    layers=MODEL_CONFIGS["deit_small_imagenet_100_seed-0"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["deit_small_imagenet_100_seed-0"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["deit_small_imagenet_100_seed-0"]["model_commitment"]["region_layer_map"]
)


model_registry["convnext_tiny_imagenet_full_seed-0"] = lambda: ModelCommitment(
    identifier="convnext_tiny_imagenet_full_seed-0",
    activations_model=get_model("convnext_tiny_imagenet_full_seed-0"),
    layers=MODEL_CONFIGS["convnext_tiny_imagenet_full_seed-0"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["convnext_tiny_imagenet_full_seed-0"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["convnext_tiny_imagenet_full_seed-0"]["model_commitment"]["region_layer_map"]
)


model_registry["convnext_small_imagenet_full_seed-0"] = lambda: ModelCommitment(
    identifier="convnext_small_imagenet_full_seed-0",
    activations_model=get_model("convnext_small_imagenet_full_seed-0"),
    layers=MODEL_CONFIGS["convnext_small_imagenet_full_seed-0"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["convnext_small_imagenet_full_seed-0"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["convnext_small_imagenet_full_seed-0"]["model_commitment"]["region_layer_map"]
)


model_registry["convnext_base_imagenet_full_seed-0"] = lambda: ModelCommitment(
    identifier="convnext_base_imagenet_full_seed-0",
    activations_model=get_model("convnext_base_imagenet_full_seed-0"),
    layers=MODEL_CONFIGS["convnext_base_imagenet_full_seed-0"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["convnext_base_imagenet_full_seed-0"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["convnext_base_imagenet_full_seed-0"]["model_commitment"]["region_layer_map"]
)


model_registry["convnext_large_imagenet_full_seed-0"] = lambda: ModelCommitment(
    identifier="convnext_large_imagenet_full_seed-0",
    activations_model=get_model("convnext_large_imagenet_full_seed-0"),
    layers=MODEL_CONFIGS["convnext_large_imagenet_full_seed-0"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["convnext_large_imagenet_full_seed-0"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["convnext_large_imagenet_full_seed-0"]["model_commitment"]["region_layer_map"]
)


model_registry["convnext_small_imagenet_1_seed-0"] = lambda: ModelCommitment(
    identifier="convnext_small_imagenet_1_seed-0",
    activations_model=get_model("convnext_small_imagenet_1_seed-0"),
    layers=MODEL_CONFIGS["convnext_small_imagenet_1_seed-0"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["convnext_small_imagenet_1_seed-0"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["convnext_small_imagenet_1_seed-0"]["model_commitment"]["region_layer_map"]
)


model_registry["convnext_small_imagenet_10_seed-0"] = lambda: ModelCommitment(
    identifier="convnext_small_imagenet_10_seed-0",
    activations_model=get_model("convnext_small_imagenet_10_seed-0"),
    layers=MODEL_CONFIGS["convnext_small_imagenet_10_seed-0"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["convnext_small_imagenet_10_seed-0"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["convnext_small_imagenet_10_seed-0"]["model_commitment"]["region_layer_map"]
)


model_registry["convnext_small_imagenet_100_seed-0"] = lambda: ModelCommitment(
    identifier="convnext_small_imagenet_100_seed-0",
    activations_model=get_model("convnext_small_imagenet_100_seed-0"),
    layers=MODEL_CONFIGS["convnext_small_imagenet_100_seed-0"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["convnext_small_imagenet_100_seed-0"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["convnext_small_imagenet_100_seed-0"]["model_commitment"]["region_layer_map"]
)


