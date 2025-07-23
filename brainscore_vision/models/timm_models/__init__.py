from brainscore_vision import model_registry
from brainscore_vision.model_helpers.brain_transformation import ModelCommitment
from .model import get_model, MODEL_CONFIGS

model_registry["convnext_base:clip_laiona_augreg_ft_in1k_384"] = lambda: ModelCommitment(
    identifier="convnext_base:clip_laiona_augreg_ft_in1k_384",
    activations_model=get_model("convnext_base:clip_laiona_augreg_ft_in1k_384"),
    layers=MODEL_CONFIGS["convnext_base:clip_laiona_augreg_ft_in1k_384"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["convnext_base:clip_laiona_augreg_ft_in1k_384"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["convnext_base:clip_laiona_augreg_ft_in1k_384"]["model_commitment"]["region_layer_map"]
)


model_registry["convnext_femto_ols:d1_in1k"] = lambda: ModelCommitment(
    identifier="convnext_femto_ols:d1_in1k",
    activations_model=get_model("convnext_femto_ols:d1_in1k"),
    layers=MODEL_CONFIGS["convnext_femto_ols:d1_in1k"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["convnext_femto_ols:d1_in1k"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["convnext_femto_ols:d1_in1k"]["model_commitment"]["region_layer_map"]
)


model_registry["convnext_large:fb_in22k_ft_in1k"] = lambda: ModelCommitment(
    identifier="convnext_large:fb_in22k_ft_in1k",
    activations_model=get_model("convnext_large:fb_in22k_ft_in1k"),
    layers=MODEL_CONFIGS["convnext_large:fb_in22k_ft_in1k"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["convnext_large:fb_in22k_ft_in1k"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["convnext_large:fb_in22k_ft_in1k"]["model_commitment"]["region_layer_map"]
)


model_registry["convnext_large_mlp:clip_laion2b_augreg_ft_in1k_384"] = lambda: ModelCommitment(
    identifier="convnext_large_mlp:clip_laion2b_augreg_ft_in1k_384",
    activations_model=get_model("convnext_large_mlp:clip_laion2b_augreg_ft_in1k_384"),
    layers=MODEL_CONFIGS["convnext_large_mlp:clip_laion2b_augreg_ft_in1k_384"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["convnext_large_mlp:clip_laion2b_augreg_ft_in1k_384"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["convnext_large_mlp:clip_laion2b_augreg_ft_in1k_384"]["model_commitment"]["region_layer_map"]
)


model_registry["convnext_tiny:in12k_ft_in1k"] = lambda: ModelCommitment(
    identifier="convnext_tiny:in12k_ft_in1k",
    activations_model=get_model("convnext_tiny:in12k_ft_in1k"),
    layers=MODEL_CONFIGS["convnext_tiny:in12k_ft_in1k"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["convnext_tiny:in12k_ft_in1k"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["convnext_tiny:in12k_ft_in1k"]["model_commitment"]["region_layer_map"]
)


model_registry["convnext_xlarge:fb_in22k_ft_in1k"] = lambda: ModelCommitment(
    identifier="convnext_xlarge:fb_in22k_ft_in1k",
    activations_model=get_model("convnext_xlarge:fb_in22k_ft_in1k"),
    layers=MODEL_CONFIGS["convnext_xlarge:fb_in22k_ft_in1k"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["convnext_xlarge:fb_in22k_ft_in1k"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["convnext_xlarge:fb_in22k_ft_in1k"]["model_commitment"]["region_layer_map"]
)


model_registry["convnext_xxlarge:clip_laion2b_soup_ft_in1k"] = lambda: ModelCommitment(
    identifier="convnext_xxlarge:clip_laion2b_soup_ft_in1k",
    activations_model=get_model("convnext_xxlarge:clip_laion2b_soup_ft_in1k"),
    layers=MODEL_CONFIGS["convnext_xxlarge:clip_laion2b_soup_ft_in1k"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["convnext_xxlarge:clip_laion2b_soup_ft_in1k"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["convnext_xxlarge:clip_laion2b_soup_ft_in1k"]["model_commitment"]["region_layer_map"]
)


model_registry["swin_small_patch4_window7_224:ms_in22k_ft_in1k"] = lambda: ModelCommitment(
    identifier="swin_small_patch4_window7_224:ms_in22k_ft_in1k",
    activations_model=get_model("swin_small_patch4_window7_224:ms_in22k_ft_in1k"),
    layers=MODEL_CONFIGS["swin_small_patch4_window7_224:ms_in22k_ft_in1k"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["swin_small_patch4_window7_224:ms_in22k_ft_in1k"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["swin_small_patch4_window7_224:ms_in22k_ft_in1k"]["model_commitment"]["region_layer_map"]
)


model_registry["vit_base_patch16_clip_224:openai_ft_in12k_in1k"] = lambda: ModelCommitment(
    identifier="vit_base_patch16_clip_224:openai_ft_in12k_in1k",
    activations_model=get_model("vit_base_patch16_clip_224:openai_ft_in12k_in1k"),
    layers=MODEL_CONFIGS["vit_base_patch16_clip_224:openai_ft_in12k_in1k"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["vit_base_patch16_clip_224:openai_ft_in12k_in1k"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["vit_base_patch16_clip_224:openai_ft_in12k_in1k"]["model_commitment"]["region_layer_map"]
)


model_registry["vit_base_patch16_clip_224:openai_ft_in1k"] = lambda: ModelCommitment(
    identifier="vit_base_patch16_clip_224:openai_ft_in1k",
    activations_model=get_model("vit_base_patch16_clip_224:openai_ft_in1k"),
    layers=MODEL_CONFIGS["vit_base_patch16_clip_224:openai_ft_in1k"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["vit_base_patch16_clip_224:openai_ft_in1k"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["vit_base_patch16_clip_224:openai_ft_in1k"]["model_commitment"]["region_layer_map"]
)


model_registry["vit_huge_patch14_clip_224:laion2b_ft_in12k_in1k"] = lambda: ModelCommitment(
    identifier="vit_huge_patch14_clip_224:laion2b_ft_in12k_in1k",
    activations_model=get_model("vit_huge_patch14_clip_224:laion2b_ft_in12k_in1k"),
    layers=MODEL_CONFIGS["vit_huge_patch14_clip_224:laion2b_ft_in12k_in1k"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["vit_huge_patch14_clip_224:laion2b_ft_in12k_in1k"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["vit_huge_patch14_clip_224:laion2b_ft_in12k_in1k"]["model_commitment"]["region_layer_map"]
)


model_registry["vit_huge_patch14_clip_336:laion2b_ft_in12k_in1k"] = lambda: ModelCommitment(
    identifier="vit_huge_patch14_clip_336:laion2b_ft_in12k_in1k",
    activations_model=get_model("vit_huge_patch14_clip_336:laion2b_ft_in12k_in1k"),
    layers=MODEL_CONFIGS["vit_huge_patch14_clip_336:laion2b_ft_in12k_in1k"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["vit_huge_patch14_clip_336:laion2b_ft_in12k_in1k"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["vit_huge_patch14_clip_336:laion2b_ft_in12k_in1k"]["model_commitment"]["region_layer_map"]
)


model_registry["vit_large_patch14_clip_224:laion2b_ft_in12k_in1k"] = lambda: ModelCommitment(
    identifier="vit_large_patch14_clip_224:laion2b_ft_in12k_in1k",
    activations_model=get_model("vit_large_patch14_clip_224:laion2b_ft_in12k_in1k"),
    layers=MODEL_CONFIGS["vit_large_patch14_clip_224:laion2b_ft_in12k_in1k"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["vit_large_patch14_clip_224:laion2b_ft_in12k_in1k"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["vit_large_patch14_clip_224:laion2b_ft_in12k_in1k"]["model_commitment"]["region_layer_map"]
)


model_registry["vit_large_patch14_clip_224:laion2b_ft_in1k"] = lambda: ModelCommitment(
    identifier="vit_large_patch14_clip_224:laion2b_ft_in1k",
    activations_model=get_model("vit_large_patch14_clip_224:laion2b_ft_in1k"),
    layers=MODEL_CONFIGS["vit_large_patch14_clip_224:laion2b_ft_in1k"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["vit_large_patch14_clip_224:laion2b_ft_in1k"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["vit_large_patch14_clip_224:laion2b_ft_in1k"]["model_commitment"]["region_layer_map"]
)


model_registry["vit_large_patch14_clip_224:openai_ft_in12k_in1k"] = lambda: ModelCommitment(
    identifier="vit_large_patch14_clip_224:openai_ft_in12k_in1k",
    activations_model=get_model("vit_large_patch14_clip_224:openai_ft_in12k_in1k"),
    layers=MODEL_CONFIGS["vit_large_patch14_clip_224:openai_ft_in12k_in1k"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["vit_large_patch14_clip_224:openai_ft_in12k_in1k"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["vit_large_patch14_clip_224:openai_ft_in12k_in1k"]["model_commitment"]["region_layer_map"]
)


model_registry["vit_large_patch14_clip_224:openai_ft_in1k"] = lambda: ModelCommitment(
    identifier="vit_large_patch14_clip_224:openai_ft_in1k",
    activations_model=get_model("vit_large_patch14_clip_224:openai_ft_in1k"),
    layers=MODEL_CONFIGS["vit_large_patch14_clip_224:openai_ft_in1k"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["vit_large_patch14_clip_224:openai_ft_in1k"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["vit_large_patch14_clip_224:openai_ft_in1k"]["model_commitment"]["region_layer_map"]
)


model_registry["vit_large_patch14_clip_336:laion2b_ft_in1k"] = lambda: ModelCommitment(
    identifier="vit_large_patch14_clip_336:laion2b_ft_in1k",
    activations_model=get_model("vit_large_patch14_clip_336:laion2b_ft_in1k"),
    layers=MODEL_CONFIGS["vit_large_patch14_clip_336:laion2b_ft_in1k"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["vit_large_patch14_clip_336:laion2b_ft_in1k"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["vit_large_patch14_clip_336:laion2b_ft_in1k"]["model_commitment"]["region_layer_map"]
)


model_registry["vit_large_patch14_clip_336:openai_ft_in12k_in1k"] = lambda: ModelCommitment(
    identifier="vit_large_patch14_clip_336:openai_ft_in12k_in1k",
    activations_model=get_model("vit_large_patch14_clip_336:openai_ft_in12k_in1k"),
    layers=MODEL_CONFIGS["vit_large_patch14_clip_336:openai_ft_in12k_in1k"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["vit_large_patch14_clip_336:openai_ft_in12k_in1k"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["vit_large_patch14_clip_336:openai_ft_in12k_in1k"]["model_commitment"]["region_layer_map"]
)


model_registry["vit_relpos_base_patch16_clsgap_224:sw_in1k"] = lambda: ModelCommitment(
    identifier="vit_relpos_base_patch16_clsgap_224:sw_in1k",
    activations_model=get_model("vit_relpos_base_patch16_clsgap_224:sw_in1k"),
    layers=MODEL_CONFIGS["vit_relpos_base_patch16_clsgap_224:sw_in1k"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["vit_relpos_base_patch16_clsgap_224:sw_in1k"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["vit_relpos_base_patch16_clsgap_224:sw_in1k"]["model_commitment"]["region_layer_map"]
)


model_registry["vit_relpos_base_patch32_plus_rpn_256:sw_in1k"] = lambda: ModelCommitment(
    identifier="vit_relpos_base_patch32_plus_rpn_256:sw_in1k",
    activations_model=get_model("vit_relpos_base_patch32_plus_rpn_256:sw_in1k"),
    layers=MODEL_CONFIGS["vit_relpos_base_patch32_plus_rpn_256:sw_in1k"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["vit_relpos_base_patch32_plus_rpn_256:sw_in1k"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["vit_relpos_base_patch32_plus_rpn_256:sw_in1k"]["model_commitment"]["region_layer_map"]
)


model_registry["vit_tiny_r_s16_p8_384:augreg_in21k_ft_in1k"] = lambda: ModelCommitment(
    identifier="vit_tiny_r_s16_p8_384:augreg_in21k_ft_in1k",
    activations_model=get_model("vit_tiny_r_s16_p8_384:augreg_in21k_ft_in1k"),
    layers=MODEL_CONFIGS["vit_tiny_r_s16_p8_384:augreg_in21k_ft_in1k"]["model_commitment"]["layers"],
    behavioral_readout_layer=MODEL_CONFIGS["vit_tiny_r_s16_p8_384:augreg_in21k_ft_in1k"]["model_commitment"]["behavioral_readout_layer"],
    region_layer_map=MODEL_CONFIGS["vit_tiny_r_s16_p8_384:augreg_in21k_ft_in1k"]["model_commitment"]["region_layer_map"]
)


