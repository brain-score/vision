from brainscore_vision import benchmark_registry
from .benchmark import Imagenet_C_Noise, Imagenet_C_Blur, Imagenet_C_Weather, Imagenet_C_Digital

benchmark_registry['dietterich.Hendrycks2019-noise-top1'] = Imagenet_C_Noise
benchmark_registry['dietterich.Hendrycks2019-blur-top1'] = Imagenet_C_Blur
benchmark_registry['dietterich.Hendrycks2019-weather-top1'] = Imagenet_C_Weather
benchmark_registry['dietterich.Hendrycks2019-digital-top1'] = Imagenet_C_Digital
