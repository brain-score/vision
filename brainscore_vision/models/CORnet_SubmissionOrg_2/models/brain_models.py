from candidate_models.model_commitments import brain_translated_pool
from model_tools.check_submission import check_models

"""
Template module for a brain model submission to brain-score
"""

def get_bibtex(model_identifier):
    return """@incollection{NIPS2012_4824,
                title = {ImageNet Classification with Deep Convolutional Neural Networks},
                author = {Alex Krizhevsky and Sutskever, Ilya and Hinton, Geoffrey E},
                booktitle = {Advances in Neural Information Processing Systems 25},
                editor = {F. Pereira and C. J. C. Burges and L. Bottou and K. Q. Weinberger},
                pages = {1097--1105},
                year = {2012},
                publisher = {Curran Associates, Inc.},
                url = {http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf}
                }"""


def get_model_list():
    """
    This method defines all submitted model names. It returns a list of model names.
    The name is then used in the get_model method to fetch the actual model instance.
    If the submission contains only one model, return a one item list.
    :return: a list of model string names
    """
    return ['CORnet-S']


def get_model(name):
    """
    This method fetches an instance of a brain model. The instance has to implement the BrainModel interface in the
    brain-score project(see imports). To get a detailed explanation of how the interface hast to be implemented,
    check out the brain-score project(https://github.com/brain-score/brain-score), examples section :param name: the
    name of the model to fetch
    :return: the model instance, which implements the BrainModel interface
    """
    return brain_translated_pool[name]


if __name__ == '__main__':
    st = ' '
    print(st.join(get_model_list()))
    # Use this method to ensure the correctness of the brain model implementations.
    # It executes a mock run of brain-score benchmarks.
    check_models.check_brain_models(__name__)
