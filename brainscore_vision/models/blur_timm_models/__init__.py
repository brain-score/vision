from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, MODEL_CONFIGS

model_registry["blur25_convnext_large_mlp:clip_laion2b_augreg_ft_in1k_384"] = lambda: ModelCommitment(
    identifier="blur25_convnext_large_mlp:clip_laion2b_augreg_ft_in1k_384",
    activations_model=get_model("blur25_convnext_large_mlp:clip_laion2b_augreg_ft_in1k_384"),
    layers=MODEL_CONFIGS["blur25_convnext_large_mlp:clip_laion2b_augreg_ft_in1k_384"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["blur25_convnext_large_mlp:clip_laion2b_augreg_ft_in1k_384"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["blur25_convnext_large_mlp:clip_laion2b_augreg_ft_in1k_384"]["model_commitment"]["region_layer_map"]
)


model_registry["blur25_convnext_tiny:in12k_ft_in1k"] = lambda: ModelCommitment(
    identifier="blur25_convnext_tiny:in12k_ft_in1k",
    activations_model=get_model("blur25_convnext_tiny:in12k_ft_in1k"),
    layers=MODEL_CONFIGS["blur25_convnext_tiny:in12k_ft_in1k"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["blur25_convnext_tiny:in12k_ft_in1k"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["blur25_convnext_tiny:in12k_ft_in1k"]["model_commitment"]["region_layer_map"]
)


model_registry["blur25_convnext_xlarge:fb_in22k_ft_in1k"] = lambda: ModelCommitment(
    identifier="blur25_convnext_xlarge:fb_in22k_ft_in1k",
    activations_model=get_model("blur25_convnext_xlarge:fb_in22k_ft_in1k"),
    layers=MODEL_CONFIGS["blur25_convnext_xlarge:fb_in22k_ft_in1k"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["blur25_convnext_xlarge:fb_in22k_ft_in1k"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["blur25_convnext_xlarge:fb_in22k_ft_in1k"]["model_commitment"]["region_layer_map"]
)


model_registry["blur25_convnext_xxlarge:clip_laion2b_soup_ft_in1k"] = lambda: ModelCommitment(
    identifier="blur25_convnext_xxlarge:clip_laion2b_soup_ft_in1k",
    activations_model=get_model("blur25_convnext_xxlarge:clip_laion2b_soup_ft_in1k"),
    layers=MODEL_CONFIGS["blur25_convnext_xxlarge:clip_laion2b_soup_ft_in1k"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["blur25_convnext_xxlarge:clip_laion2b_soup_ft_in1k"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["blur25_convnext_xxlarge:clip_laion2b_soup_ft_in1k"]["model_commitment"]["region_layer_map"]
)


model_registry["blur25_vit_huge_patch14_clip_224:laion2b_ft_in12k_in1k"] = lambda: ModelCommitment(
    identifier="blur25_vit_huge_patch14_clip_224:laion2b_ft_in12k_in1k",
    activations_model=get_model("blur25_vit_huge_patch14_clip_224:laion2b_ft_in12k_in1k"),
    layers=MODEL_CONFIGS["blur25_vit_huge_patch14_clip_224:laion2b_ft_in12k_in1k"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["blur25_vit_huge_patch14_clip_224:laion2b_ft_in12k_in1k"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["blur25_vit_huge_patch14_clip_224:laion2b_ft_in12k_in1k"]["model_commitment"]["region_layer_map"]
)


model_registry["blur25_vit_large_patch14_clip_224:laion2b_ft_in12k_in1k"] = lambda: ModelCommitment(
    identifier="blur25_vit_large_patch14_clip_224:laion2b_ft_in12k_in1k",
    activations_model=get_model("blur25_vit_large_patch14_clip_224:laion2b_ft_in12k_in1k"),
    layers=MODEL_CONFIGS["blur25_vit_large_patch14_clip_224:laion2b_ft_in12k_in1k"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["blur25_vit_large_patch14_clip_224:laion2b_ft_in12k_in1k"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["blur25_vit_large_patch14_clip_224:laion2b_ft_in12k_in1k"]["model_commitment"]["region_layer_map"]
)


model_registry["blur25_vit_large_patch14_clip_224:openai_ft_in12k_in1k"] = lambda: ModelCommitment(
    identifier="blur25_vit_large_patch14_clip_224:openai_ft_in12k_in1k",
    activations_model=get_model("blur25_vit_large_patch14_clip_224:openai_ft_in12k_in1k"),
    layers=MODEL_CONFIGS["blur25_vit_large_patch14_clip_224:openai_ft_in12k_in1k"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["blur25_vit_large_patch14_clip_224:openai_ft_in12k_in1k"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["blur25_vit_large_patch14_clip_224:openai_ft_in12k_in1k"]["model_commitment"]["region_layer_map"]
)


model_registry["blur25_vit_large_patch14_clip_336:openai_ft_in12k_in1k"] = lambda: ModelCommitment(
    identifier="blur25_vit_large_patch14_clip_336:openai_ft_in12k_in1k",
    activations_model=get_model("blur25_vit_large_patch14_clip_336:openai_ft_in12k_in1k"),
    layers=MODEL_CONFIGS["blur25_vit_large_patch14_clip_336:openai_ft_in12k_in1k"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["blur25_vit_large_patch14_clip_336:openai_ft_in12k_in1k"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["blur25_vit_large_patch14_clip_336:openai_ft_in12k_in1k"]["model_commitment"]["region_layer_map"]
)