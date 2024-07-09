from brainscore_vision import metric_registry
from .metric import ConfusionSimilarity, TasksConsistency

BIBTEX = """@article {Maniquet2024.04.02.587669,
        author = {Maniquet, Tim and de Beeck, Hans Op and Costantino, Andrea Ivan},
        title = {Recurrent issues with deep neural network models of visual recognition},
        elocation-id = {2024.04.02.587669},
        year = {2024},
        doi = {10.1101/2024.04.02.587669},
        publisher = {Cold Spring Harbor Laboratory},
        URL = {https://www.biorxiv.org/content/early/2024/04/10/2024.04.02.587669},
        eprint = {https://www.biorxiv.org/content/early/2024/04/10/2024.04.02.587669.full.pdf},
        journal = {bioRxiv}
}"""

metric_registry['confusion_similarity'] = ConfusionSimilarity
metric_registry['tasks_consistency'] = TasksConsistency


