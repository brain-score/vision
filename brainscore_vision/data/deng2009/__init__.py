from brainscore_vision import stimulus_set_registry
from brainscore_vision.data_helpers.s3 import load_stimulus_set_from_s3

BIBTEX = """@INPROCEEDINGS{5206848,
  author={Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Kai Li and Li Fei-Fei},
  booktitle={2009 IEEE Conference on Computer Vision and Pattern Recognition}, 
  title={ImageNet: A large-scale hierarchical image database}, 
  year={2009},
  volume={},
  number={},
  pages={248-255},
  doi={10.1109/CVPR.2009.5206848}}"""


# stimulus set
stimulus_set_registry['imagenet_val'] = lambda: load_stimulus_set_from_s3(
    identifier="imagenet_val",
    bucket="brainio-brainscore",
    csv_sha1="ff79dcf6b0d115e6e8aa8d0fbba3af11dc649e57",
    zip_sha1="78172d752d8216a00833cfa34be67c8532ad7330",
    csv_version_id="7TGln_E5k.V8NdNYGB55mX9JYF0s7R4N",
    zip_version_id="7DJHWmhFvqTYblo8amGykHR3LYyS.3UC")
