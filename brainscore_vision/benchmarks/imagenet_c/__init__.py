from brainscore_vision import benchmark_registry
from .benchmark import Imagenet_C_Noise, Imagenet_C_Blur, Imagenet_C_Weather, Imagenet_C_Digital

benchmark_registry['ImageNet-C-noise-top1'] = Imagenet_C_Noise
benchmark_registry['ImageNet-C-blur-top1'] = Imagenet_C_Blur
benchmark_registry['ImageNet-C-weather-top1'] = Imagenet_C_Weather
benchmark_registry['ImageNet-C-digital-top1'] = Imagenet_C_Digital
