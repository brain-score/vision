.. _benchmarks:

Benchmarks
----------

``Benchmark``s run an experiment on a model_ and test the resulting measurements against primate data_.
This comparison is done by :ref:`metrics` which output a score.
This score is normalized with data ceilings and hte output ultimately outputs this ceiled score.

.. automodule:: brainscore.benchmarks
    :members:
    :undoc-members:
.. _data: https://github.com/brain-score/brainio_collection
.. _model: https://github.com/brain-score/brain-score/blob/master/brainscore/model_interface.py
