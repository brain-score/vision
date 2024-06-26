import functools
import importlib
from brainscore_vision.model_helpers.activations.pytorch import load_preprocess_images
from brainscore_vision.model_helpers.check_submission import check_models
import torch.hub
import ssl
from brainscore_vision.model_helpers.s3 import load_weight_file
from torch.nn import Module
from .helpers.helpers import TemporalPytorchWrapper


ssl._create_default_https_context = ssl._create_unverified_context


TIME_MAPPINGS = {
        'V1': (50, 100, 1),
        'V2': (70, 100, 2),
        # 'V2': (20, 50, 2),  # MS: This follows from the movshon anesthesized-monkey recordings, so might not hold up
        'V4': (90, 50, 4),
        'IT': (100, 100, 2),
    }


def get_model(name):
    assert name == 'cornet_s'

    class Wrapper(Module):
        def __init__(self, model):
            super(Wrapper, self).__init__()
            self.module = model

    mod = importlib.import_module(f'cornet.cornet_s')
    model_ctr = getattr(mod, 'CORnet_S')
    model = model_ctr()
    model = Wrapper(model)  # model was wrapped with DataParallel, so weights require `module.` prefix
    weights_path = load_weight_file(bucket="brainscore-vision", folder_name="models",
                                    relative_path="cornet_s/cornet_s_epoch43.pth.tar",
                                    version_id="4EAQnCqTy.2MCKiXTJ4l02iG8l3e.yfQ",
                                    sha1="a4bfd8eda33b45fd945da1b972ab0b7cad38d60f")
    checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)  # map onto cpu
    model.load_state_dict(checkpoint['state_dict'])
    model = model.module  # unwrap
    preprocessing = functools.partial(load_preprocess_images, image_size=224)
    wrapper = TemporalPytorchWrapper(identifier="CORnet-S", model=model, preprocessing=preprocessing,
                                     separate_time=True)
    wrapper.image_size = 224
    return wrapper


def get_layers(name):
    assert name == 'cornet_s'
    return (['V1.output-t0'] +
               [f'{area}.output-t{timestep}'
                for area, timesteps in [('V2', range(2)), ('V4', range(4)), ('IT', range(2))]
                for timestep in timesteps] +
               ['decoder.avgpool-t0']
            )


def get_bibtex(model_identifier):
    return """@inproceedings{KubiliusSchrimpf2019CORnet,
            abstract = {Deep convolutional artificial neural networks (ANNs) are the leading class of candidate models of the mechanisms of visual processing in the primate ventral stream. While initially inspired by brain anatomy, over the past years, these ANNs have evolved from a simple eight-layer architecture in AlexNet to extremely deep and branching architectures, demonstrating increasingly better object categorization performance, yet bringing into question how brain-like they still are. In particular, typical deep models from the machine learning community are often hard to map onto the brain's anatomy due to their vast number of layers and missing biologically-important connections, such as recurrence. Here we demonstrate that better anatomical alignment to the brain and high performance on machine learning as well as neuroscience measures do not have to be in contradiction. We developed CORnet-S, a shallow ANN with four anatomically mapped areas and recurrent connectivity, guided by Brain-Score, a new large-scale composite of neural and behavioral benchmarks for quantifying the functional fidelity of models of the primate ventral visual stream. Despite being significantly shallower than most models, CORnet-S is the top model on Brain-Score and outperforms similarly compact models on ImageNet. Moreover, our extensive analyses of CORnet-S circuitry variants reveal that recurrence is the main predictive factor of both Brain-Score and ImageNet top-1 performance. Finally, we report that the temporal evolution of the CORnet-S "IT" neural population resembles the actual monkey IT population dynamics. Taken together, these results establish CORnet-S, a compact, recurrent ANN, as the current best model of the primate ventral visual stream.},
            archivePrefix = {arXiv},
            arxivId = {1909.06161},
            author = {Kubilius, Jonas and Schrimpf, Martin and Hong, Ha and Majaj, Najib J. and Rajalingham, Rishi and Issa, Elias B. and Kar, Kohitij and Bashivan, Pouya and Prescott-Roy, Jonathan and Schmidt, Kailyn and Nayebi, Aran and Bear, Daniel and Yamins, Daniel L. K. and DiCarlo, James J.},
            booktitle = {Neural Information Processing Systems (NeurIPS)},
            editor = {Wallach, H. and Larochelle, H. and Beygelzimer, A. and D'Alch{\'{e}}-Buc, F. and Fox, E. and Garnett, R.},
            pages = {12785----12796},
            publisher = {Curran Associates, Inc.},
            title = {{Brain-Like Object Recognition with High-Performing Shallow Recurrent ANNs}},
            url = {http://papers.nips.cc/paper/9441-brain-like-object-recognition-with-high-performing-shallow-recurrent-anns},
            year = {2019}
            }"""


if __name__ == '__main__':
    check_models.check_base_models(__name__)