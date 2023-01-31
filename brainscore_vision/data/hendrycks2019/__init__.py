from brainscore_vision import stimulus_set_registry
from brainscore_vision.data_helpers.s3 import load_stimulus_set_from_s3

BIBTEX = """@article{hendrycks2019benchmarking,
  title={Benchmarking neural network robustness to common corruptions and perturbations},
  author={Hendrycks, Dan and Dietterich, Thomas},
  journal={arXiv preprint arXiv:1903.12261},
  year={2019}
}"""

# stimulus set: dietterich.Hendrycks2019.noise
stimulus_set_registry['dietterich.Hendrycks2019.noise'] = lambda: load_stimulus_set_from_s3(
    identifier="dietterich.Hendrycks2019.noise",
    bucket="brainio-brainscore",
    csv_sha1="56f445e058b4d825e7731711c824918812ed2d2d",
    zip_sha1="e3c46b81bfd8a522cadcd8a4bb0c67bb5ccb4c6a",
    csv_version_id="oR0_anEdzvyqgLSWz8.uLbzYeMQR2uem",
    zip_version_id="Hcbdp9m83Ajoxc5nqWDB0zi9UFplB1w7")

# stimulus set: dietterich.Hendrycks2019.blur
stimulus_set_registry['dietterich.Hendrycks2019.blur'] = lambda: load_stimulus_set_from_s3(
    identifier="dietterich.Hendrycks2019.blur",
    bucket="brainio-brainscore",
    csv_sha1="e7a537bb2f3f94b9cd3819a529de9f4349e58bd2",
    zip_sha1="85bac6d7c9b9646b22480f65cf6f1486fcf4b488",
    csv_version_id="WegyPp9lBJ5jw_hQDhRN8mKQ8aRsCIzv",
    zip_version_id="IIF74vl5z75b65umHvgn5nds8XHz05Jd")

# stimulus set: dietterich.Hendrycks2019.weather
stimulus_set_registry['dietterich.Hendrycks2019.weather'] = lambda: load_stimulus_set_from_s3(
    identifier="dietterich.Hendrycks2019.weather",
    bucket="brainio-brainscore",
    csv_sha1="02fa92430be754163ecd06ee9211c1edd8984207",
    zip_sha1="18ef85343b10a4a326b5fc0758b9ae1da58f7d90",
    csv_version_id=".2f.Ze0pLG0.qgPxiVAjMQAI.QMzSg5S",
    zip_version_id="hy.5apU1C13feZSHOR90daDHDc5oRtVk")

# stimulus set: dietterich.Hendrycks2019.digital
stimulus_set_registry['dietterich.Hendrycks2019.digital'] = lambda: load_stimulus_set_from_s3(
    identifier="dietterich.Hendrycks2019.digital",
    bucket="brainio-brainscore",
    csv_sha1="1e9611b560333989d4673bfd019160a27842f89b",
    zip_sha1="a10e8d7eabce191d1f8e92983c08b4f0fe0f435d",
    csv_version_id="f_xxGK_dwzOyy8tM4hmSnnIlRa1RwCiF",
    zip_version_id="3eblycjaIWN7E4lCcQwMRtRBSotiCxun")
