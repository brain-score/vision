from brainscore_vision import metric_registry
from .metric import CrossRegressedCorrelation, pls_regression, ridge_regression, single_regression, linear_regression,\
    pearsonr_correlation

#metrics using cross-validation to generate multiple train-test splits from a monolithic dataset

metric_registry['pls-cv'] = lambda *args, **kwargs: CrossRegressedCorrelation(
    regression=pls_regression(), correlation=pearsonr_correlation(), *args, **kwargs)
metric_registry['ridge-cv'] = lambda *args, **kwargs: CrossRegressedCorrelation(
    regression=ridge_regression(), correlation=pearsonr_correlation(), *args, **kwargs)
metric_registry['neuron_to_neuron-cv'] = lambda *args, **kwargs: CrossRegressedCorrelation(
    regression=single_regression(), correlation=pearsonr_correlation(), *args, **kwargs)
metric_registry['linear_predictivity-cv'] = lambda *args, **kwargs: CrossRegressedCorrelation(
    regression=linear_regression(), correlation=pearsonr_correlation(), *args, **kwargs)

# metrics using seperate train and test sets
from .metric import FixedTrainTestSplitCorrelation
metric_registry['pls-split'] = lambda *args, **kwargs: FixedTrainTestSplitCorrelation(
    regression=pls_regression(), correlation=pearsonr_correlation(), *args, **kwargs)
metric_registry['ridge-split'] = lambda *args, **kwargs: FixedTrainTestSplitCorrelation(
    regression=ridge_regression(), correlation=pearsonr_correlation(), *args, **kwargs)
metric_registry['neuron_to_neuron-split'] = lambda *args, **kwargs: FixedTrainTestSplitCorrelation(
    regression=single_regression(), correlation=pearsonr_correlation(), *args, **kwargs)
metric_registry['linear_predictivity-split'] = lambda *args, **kwargs: FixedTrainTestSplitCorrelation(
    regression=linear_regression(), correlation=pearsonr_correlation(), *args, **kwargs)


#backwards compatibility
metric_registry['pls'] = metric_registry['pls-cv']
metric_registry['ridge'] = metric_registry['ridge-cv']
metric_registry['neuron_to_neuron'] = metric_registry['neuron_to_neuron-cv']
metric_registry['linear_predictivity'] = metric_registry['linear_predictivity-cv']


# temporal metrics
from .metric import SpanTimeCrossRegressedCorrelation

metric_registry['spantime_pls'] = lambda *args, **kwargs: SpanTimeCrossRegressedCorrelation(
  regression=pls_regression(), correlation=pearsonr_correlation(), *args, **kwargs)
metric_registry['spantime_ridge'] = lambda *args, **kwargs: SpanTimeCrossRegressedCorrelation(
  regression=ridge_regression(), correlation=pearsonr_correlation(), *args, **kwargs)


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
