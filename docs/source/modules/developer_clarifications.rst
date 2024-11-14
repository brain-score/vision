.. _interface:

************************
Developer Clarifications
************************

The Following documentation stores commonly-asked developer questions. We hope this will be useful to
anyone interested in contributing to Brain-Score's codebase or scientific workings.



1. **For a given model, are activations different on each benchmark? How?**


    Activations per model are generated based on benchmark stimuli; not every benchmark has unique stimuli. For most
    model-benchmark pairs, activations will be different because stimuli will be different. The exceptions to this
    are the benchmarks that use the same stimuli, such as the `MajajHong20215` family of benchmarks.

2. **Result Caching**

    Result Caching is a Brain-Score `repo <https://github.com/brain-score/result_caching>`_ that allows model activations (and other functions) to be cached
    to disk, in order to speed up the process of rescoring models. It contains a decorator that can be attached to a function
    right before it is defined. On the first run of that function, `result_caching` will save to disk the result of tha function
    and will load that result from disk in future calls with the same parameters. All files are saved in the user's `~/result_caching`
    folder, and they are persistent, as there is no garbage collection built in. You can deactivate
    `result_caching` by simply setting the environment flag `RESULTCACHING_DISABLE` to `1`. Please see the link above
    for more detailed documentation.

3. **Model Mapping Procedure**

    In general, there are different methods that are used in the Brain-Score code to instruct the model to "begin recording",
    observe stimuli, and to generate scores. Models follow the `ModelCommitment` to conform to the `BrainModel` API. A
    `BrainModel` is any model that has a `region_layer_map`. This allows the layers in the model to be mapped to layers in
    the ventral visual stream, and is chosen by scoring models on the public version of a benchmark (the private
    benchmark data is heldout for the BrainModel to be scored on). See the more technical docs
    `here <https://brain-score.readthedocs.io/en/latest/modules/model_interface.html>`_  for additional notes.
