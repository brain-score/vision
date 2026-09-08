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

    Note that `RESULTCACHING_DISABLE=1` disables *every* cached function, not just activations -- including
    `place_on_screen`. To disable only the activation cache (for example, to check that a score is unaffected by
    caching), name the module instead::

        RESULTCACHING_DISABLE=brainscore_vision.model_helpers.activations

    In production scoring, activations are additionally cached to a *shared* S3 bucket so a second metric on the same
    model and stimuli reuses the forward pass instead of repeating it. Because that cache is long-lived and shared,
    its keys carry a content revision of the model plugin and of the stimulus-set plugin: a plugin can be revised
    while keeping the same identifier, and without a revision the cache would serve activations produced by code that
    no longer exists. If a revision cannot be resolved, brain-score **refuses to cache** that request rather than
    write an ambiguous key -- correct, but it means the work is recomputed every time. See item 4.

4. **Naming a stimulus set so it can be cached**

    Benchmarks often build a stimulus set at runtime rather than using a registered one directly -- a merged
    train+test pool, a per-subject slice, a filtered subset. Such a name is not a registered plugin, so no revision
    can be resolved for it, so the shared activation cache refuses it and every extraction is recomputed at full
    cost. The only symptom is one warning line in a scoring log.

    Name a synthesised stimulus set after the registered set it derives from::

        stim.identifier = f"{registered_identifier}--{marker}"

    Everything from the first ``--`` onward stays in the cache key, so derivatives remain distinct from one another;
    only the part before it is used to look up the revision. `place_on_screen` already follows this convention
    (appending ``--target<deg>--source<deg>``), and its suffix composes with a benchmark's own marker.

    For example, a benchmark that loads `Zerbe2026_fmri_stim_full` and builds a per-subject merged pool from it
    should name that pool ``Zerbe2026_fmri_stim_full--rdm-sub-01``, not ``Zerbe2026_fmri_rdm_full_sub-01``. The
    second form resolves to nothing and disabled caching for an entire benchmark family until it was caught.

    Assert this in your plugin's tests so it is caught at PR time rather than in a production scoring log::

        from brainscore_vision.benchmark_helpers.cache_contract import (
            assert_stimulus_identifier_is_cacheable)

        def test_stimulus_identifier_is_cacheable():
            assert_stimulus_identifier_is_cacheable("MyData_stim_full--my-variant")

    This only matters for stimulus sets a benchmark *synthesises*. A benchmark that uses a registered stimulus set
    directly already resolves.

3. **Model Mapping Procedure**

    In general, there are different methods that are used in the Brain-Score code to instruct the model to "begin recording",
    observe stimuli, and to generate scores. Models follow the `ModelCommitment` to conform to the `BrainModel` API. A
    `BrainModel` is any model that has a `region_layer_map`. This allows the layers in the model to be mapped to layers in
    the ventral visual stream, and is chosen by scoring models on the public version of a benchmark (the private
    benchmark data is heldout for the BrainModel to be scored on). See the more technical docs
    `here <https://brain-score.readthedocs.io/en/latest/modules/model_interface.html>`_  for additional notes.
