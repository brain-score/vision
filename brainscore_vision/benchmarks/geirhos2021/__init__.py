# from brainscore_vision import benchmark_registry
# from .benchmark import DicarloSanghavi2020V4PLS, DicarloSanghavi2020ITPLS
#
# DATASETS = ['colour', 'contrast', 'cue-conflict', 'edge',
#             'eidolonI', 'eidolonII', 'eidolonIII',
#             'false-colour', 'high-pass', 'low-pass', 'phase-scrambling', 'power-equalisation',
#             'rotation', 'silhouette', 'sketch', 'stylized', 'uniform-noise']
#
# for dataset in DATASETS:
#     # behavioral benchmark
#     identifier = f"Geirhos2021{dataset.replace('-', '')}ErrorConsistency"
#     globals()[identifier] = lambda dataset=dataset: _Geirhos2021ErrorConsistency(dataset)
#     # engineering benchmark
#     identifier = f"Geirhos2021{dataset.replace('-', '')}Accuracy"
#     globals()[identifier] = lambda dataset=dataset: _Geirhos2021Accuracy(dataset)
#
# benchmark_registry['dicarlo.Sanghavi2020.V4-pls'] = DicarloSanghavi2020V4PLS
# benchmark_registry['dicarlo.Sanghavi2020.IT-pls'] = DicarloSanghavi2020ITPLS