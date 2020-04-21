.. _benchmarks:

Benchmarks
----------

A ``Benchmark`` runs an experiment on a model_ and tests the resulting measurements against primate data_.
This comparison is done by :ref:`metrics` which output a score of how well model and data match.
This score is normalized with data ceilings and the benchmark returns this ceiled score.

.. automodule:: brainscore.benchmarks
    :members:
    :undoc-members:
.. _model: https://github.com/brain-score/brain-score/blob/master/brainscore/model_interface.py
.. _data: https://github.com/brain-score/brainio_collection
