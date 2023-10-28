from brainio.assemblies import NeuronRecordingAssembly

from brainscore_vision import data_registry, stimulus_set_registry, load_stimulus_set
from brainscore_vision.metrics.internal_consistency.ceiling import InternalConsistency
from brainscore_vision.metric_helpers.transformations import CrossValidation
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

BIBTEX = """@article {19,
    title = {Evidence that recurrent circuits are critical to the ventral stream{\textquoteright}s execution of core object recognition behavior},
    journal = {bioRxiv},
    year = {2018},
    month = {06/2018},
    type = {preprint},
    abstract = {<p>Non-recurrent deep convolutional neural networks (DCNNs) are currently the best models of core object recognition; a behavior supported by the densely recurrent primate ventral stream, culminating in the inferior temporal (IT) cortex. Are these recurrent circuits critical to ventral stream\&$\#$39;s execution of this behavior? We reasoned that, if recurrence is critical, then primates should outperform feedforward-only DCNNs for some images, and that these images should require additional processing time beyond the feedforward IT response. Here we first used behavioral methods to discover hundreds of these \&quot;challenge\&quot; images. Second, using large-scale IT electrophysiology in animals performing core recognition tasks, we observed that behaviorally-sufficient, linearly-decodable object identity solutions emerged ~30ms (on average) later in IT for challenge images compared to DCNN and primate performance-matched \&quot;control\&quot; images. We observed these same late solutions even during passive viewing. Third, consistent with a failure of feedforward computations, the behaviorally-critical late-phase IT population response patterns evoked by the challenge images were poorly predicted by DCNN activations. Interestingly, deeper CNNs better predicted these late IT responses, suggesting a functional equivalence between recurrence and additional nonlinear transformations. Our results argue that automatically-evoked recurrent circuits are critical even for rapid object identification. By precisely comparing current DCNNs, primate behavior and IT population dynamics, we provide guidance for future recurrent model development.</p>
},
    doi = {https://doi.org/10.1101/354753},
    url = {https://www.biorxiv.org/content/10.1101/354753v1.full.pdf},
    author = {Kar, Kohitij and Kubilius, Jonas and Schmidt, Kailyn and Issa, Elias B and DiCarlo, James J.}
}"""

# assemblies: hvm
data_registry['dicarlo.Kar2018hvm'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.Kar2018hvm",
    version_id="4sytUtSGiyB.G0oBmPCVnnQ6l4FChj8z",
    sha1="96ccacc76c5fa30ee68bdc8736d1d43ace93f3e7",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('dicarlo.hvm'),
)

# assemblies: cocogray
data_registry['dicarlo.Kar2018cocogray'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.Kar2018cocogray",
    version_id="RxiK296HHAe2ql_STmZ2K..uEsfCuHtF",
    sha1="4202cb3992a5d71f71a7ca9e28ba3f8b27937b43",
    bucket="brainio-brainscore",
    cls=NeuronRecordingAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('dicarlo.Kar2018cocogray'),
)

# stimulus set: cocogray
stimulus_set_registry['dicarlo.Kar2018cocogray'] = lambda: load_stimulus_set_from_s3(
    identifier="dicarlo.Kar2018cocogray",
    bucket="brainio-brainscore",
    csv_sha1="be9bb267b80fd7ee36a88d025b73ae8a849165da",
    zip_sha1="1457003ee9b27ed51c018237009fe148c6e71fd3",
    csv_version_id="3Y1o11OuW8dmEyqJ7WHxrKCTqacBC20_",
    zip_version_id="bnYW48AQ7DoLBzWY5Bx0IZ9r8_bwvWRC")


def filter_neuroids(assembly, threshold):
    ceiler = InternalConsistency()
    ceiling = ceiler(assembly)
    ceiling = ceiling.raw
    ceiling = CrossValidation().aggregate(ceiling)
    ceiling = ceiling.sel(aggregation='center')
    pass_threshold = ceiling >= threshold
    assembly = assembly[{'neuroid': pass_threshold}]
    return assembly
