from brainio.assemblies import DataAssembly

from brainscore_vision import data_registry, stimulus_set_registry, load_stimulus_set
from brainscore_vision.data_helpers.s3 import load_assembly_from_s3, load_stimulus_set_from_s3

BIBTEX = """@Article{Kar2019,
            author={Kar, Kohitij
            and Kubilius, Jonas
            and Schmidt, Kailyn
            and Issa, Elias B.
            and DiCarlo, James J.},
            title={Evidence that recurrent circuits are critical to the ventral stream's execution of core object recognition behavior},
            journal={Nature Neuroscience},
            year={2019},
            month={Jun},
            day={01},
            volume={22},
            number={6},
            pages={974-983},
            abstract={Non-recurrent deep convolutional neural networks (CNNs) are currently the best at modeling core object recognition, a behavior that is supported by the densely recurrent primate ventral stream, culminating in the inferior temporal (IT) cortex. If recurrence is critical to this behavior, then primates should outperform feedforward-only deep CNNs for images that require additional recurrent processing beyond the feedforward IT response. Here we first used behavioral methods to discover hundreds of these `challenge' images. Second, using large-scale electrophysiology, we observed that behaviorally sufficient object identity solutions emerged {\textasciitilde}30{\thinspace}ms later in the IT cortex for challenge images compared with primate performance-matched `control' images. Third, these behaviorally critical late-phase IT response patterns were poorly predicted by feedforward deep CNN activations. Notably, very-deep CNNs and shallower recurrent CNNs better predicted these late IT responses, suggesting that there is a functional equivalence between additional nonlinear transformations and recurrence. Beyond arguing that recurrent circuits are critical for rapid object identification, our results provide strong constraints for future recurrent model development.},
            issn={1546-1726},
            doi={10.1038/s41593-019-0392-5},
            url={https://doi.org/10.1038/s41593-019-0392-5}
            }"""

# assembly
data_registry['Kar2019'] = lambda: load_assembly_from_s3(
    identifier="dicarlo.Kar2019",
    version_id="mQZ1nNRfE_m_IjFbox.xvlJvbudFwsWo",
    sha1="147717ce397e11d56164d472063a84a83bbcbb94",
    bucket="brainio-brainscore",
    cls=DataAssembly,
    stimulus_set_loader=lambda: load_stimulus_set('Kar2019'),
)

# stimulus set
stimulus_set_registry['Kar2019'] = lambda: load_stimulus_set_from_s3(
    identifier="Kar2019",
    bucket="brainio-brainscore",
    csv_sha1="7f705bdea02c0a72a76d7f5e7b6963531df818a6",
    zip_sha1="75ab7b8b499fc8e86c813f717b79d268bcb986be",
    csv_version_id="RxErKK258xYBuO6.ThuQU0IACEWkZ20o",
    zip_version_id="3ArjwRwvdMgjwA2VP63zQ4rhF5WIbr.L")
