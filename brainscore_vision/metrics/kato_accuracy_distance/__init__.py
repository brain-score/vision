from brainscore_vision import metric_registry
from .metric import KatoAccuracyDistance

metric_registry['kato_accuracy_distance'] = KatoAccuracyDistance

BIBTEX = """@article{Kato2026,
    title = {Systematic image perturbations reveal persistent gaps between human and machine vision},
    volume = {XX},
    doi = {XX},
    journal = {iScience},
    author = {Kato, Mugihiko, and He, Biyu},
    year = {2026},
    }"""
