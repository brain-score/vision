from brainscore_vision import metric_registry
from .metric import CrossRegressedCorrelation, pls_regression, ridge_regression, single_regression, linear_regression,\
    pearsonr_correlation

metric_registry['pls'] = lambda *args, **kwargs: CrossRegressedCorrelation(
    regression=pls_regression(), correlation=pearsonr_correlation(), *args, **kwargs)
metric_registry['ridge'] = lambda *args, **kwargs: CrossRegressedCorrelation(
    regression=ridge_regression(), correlation=pearsonr_correlation(), *args, **kwargs)
metric_registry['neuron_to_neuron'] = lambda *args, **kwargs: CrossRegressedCorrelation(
    regression=single_regression(), correlation=pearsonr_correlation(), *args, **kwargs)
metric_registry['linear_predictivity'] = lambda *args, **kwargs: CrossRegressedCorrelation(
    regression=linear_regression(), correlation=pearsonr_correlation(), *args, **kwargs)

BIBTEX = """@article{schrimpf2018brain,
  title={Brain-score: Which artificial neural network for object recognition is most brain-like?},
  author={Schrimpf, Martin and Kubilius, Jonas and Hong, Ha and Majaj, Najib J and Rajalingham, Rishi and Issa, Elias B and Kar, Kohitij and Bashivan, Pouya and Prescott-Roy, Jonathan and Geiger, Franziska and others},
  journal={BioRxiv},
  pages={407007},
  year={2018},
  publisher={Cold Spring Harbor Laboratory}
}"""

BIBTEX_PLS = """@article{yamins2014performance,
  title={Performance-optimized hierarchical models predict neural responses in higher visual cortex},
  author={Yamins, Daniel LK and Hong, Ha and Cadieu, Charles F and Solomon, Ethan A and Seibert, Darren and DiCarlo, James J},
  journal={Proceedings of the national academy of sciences},
  volume={111},
  number={23},
  pages={8619--8624},
  year={2014},
  publisher={National Acad Sciences}
}"""

BIBTEX_NEURONTONEURON = """@techreport{arend2018single,
  title={Single units in a deep neural network functionally correspond with neurons in the brain: preliminary results},
  author={Arend, Luke and Han, Yena and Schrimpf, Martin and Bashivan, Pouya and Kar, Kohitij and Poggio, Tomaso and DiCarlo, James J and Boix, Xavier},
  year={2018},
  institution={Center for Brains, Minds and Machines (CBMM)}
}"""
