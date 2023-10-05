.. _Benchmark_Tutorial:
.. |UnitTestSupport| replace:: We realize that unit tests can be a hurdle and we can take over this task for you.
                                 Please let us know of any hurdles and we will do our best to support.

==================
Benchmark Tutorial
==================

Benchmarks are at the core of Brain-Score and test models' match to experimental observations.
New benchmarks keep models in check and require them to generalize to new experiments.

A benchmark reproduces the experimental paradigm on a model candidate, the experimentally observed data,
and a metric to compare model with experimental observations.

To submit a new benchmark, there are three steps:
1. packaging stimuli and data,
2. creating the benchmark with experimental paradigm and metric to compare against data, and
3. opening a pull request on the github repository to commit the updates from 1 and 2
In order to ensure the continued validity of the benchmark, we require unit tests for all components
(stimuli and data as well as the benchmark itself).

1. Package stimuli and data
===========================
We require a certain format for stimuli and data so that we can maintain them for long-term use.
In particular, we use BrainIO for data management. BrainIO uses
`StimulusSet <https://github.com/brain-score/brainio/blob/main/brainio/stimuli.py>`_ (a subclass of
`pandas DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html>`_) to maintain stimuli, and
`DataAssembly <https://github.com/brain-score/brainio/blob/main/brainio/assemblies.py>`_
(a subclass of `xarray DataArray <http://xarray.pydata.org/en/stable/generated/xarray.DataArray.html>`_)
to maintain experimental measurements.
Aside from unifying data from different sources, the advantage of these formats is that all data are kept together with
metadata such as image parameters, electrode locations, and details on behavioral choices.
For both StimulusSet and DataAssembly, BrainIO provides packaging methods that upload to S3 cloud storage, and add the
entries to `lookup.csv <https://github.com/brain-score/brain-score/blob/master/brainscore/lookup.csv>`_ from which they
can later be accessed.

Data and stimuli can be made public or kept private. It is your choice if you wish to release the data itself or only
the benchmark. If you choose to keep the data private, model submissions can be scored on the data, but the actual data
itself will not be visible. Publicly released data can also be scored against, but will be fully accessible.

Getting started, please create a new folder :code:`<authoryear>/__init__.py` in the :code:`packaging` directory in
which you keep all your packaging scripts.
If your code depends on additional requirements, it is good practice to additionally keep a :code:`requirements.txt`
or :code:`setup.py` file specifying the dependencies.

Before executing the packaging methods to actually upload to S3, please check in with us
(msch@mit.edu, mferg@mit.edu, jjpr@mit.edu) so that we can give you access. With the credentials, you can then
configure the awscli (:code:`pip install awscli`, :code:`aws configure` using region :code:`us-east-1`,
output format :code:`json`) to make the packaging methods upload successfully.

**StimulusSet**:
The StimulusSet contains the stimuli that were used in the experiment as well as any kind of metadata for the stimuli.
Here is a slim example of creating and uploading a StimulusSet:

.. code-block:: python

    from pathlib import Path
    from brainio.stimuli import StimulusSet
    from brainio.packaging import package_stimulus_set

    stimuli = []  # collect meta
    stimulus_paths = {}  # collect mapping of stimulus_id to filepath
    for filepath in Path(stimuli_directory).glob('*.png'):
        stimulus_id = filepath.stem
        object_name = filepath.stem.split('_')[0]  # if the filepath contains meta, this can come from anywhere
        # ...and other metadata
        stimulus_paths[stimulus_id] = filepath
        stimuli.append({
            'stimulus_id': stimulus_id,
            'object_name': object_name,
            # ...and other metadata
            # optionally you can set 'stimulus_path_within_store' to define the filename in the packaged stimuli
        })
    stimuli = StimulusSet(stimuli)
    stimuli.stimulus_paths = stimulus_paths
    stimuli.name = '<AuthorYear>'  # give the StimulusSet an identifier name

    assert len(stimuli) == 1600  # make sure the StimulusSet is what you would expect

    package_stimulus_set(catalog_name='brainio_brainscore', proto_stimulus_set=stimuli, stimulus_set_identifier=stimuli.name) # upload to S3 and add entry to local lookup.csv


**DataAssembly**:

DataAssemblies contain the actual experimental measurements as well as any metadata on them.
Note that these do not necessarily have to be raw data, but can also be previously published characterizations of the
data such as preference distributions.
As such, the person submitting the data to Brain-Score does not have to be involved in the data collection.
If you package someone else's data, we do however recommend checking the specifics with them to avoid mis-interpretation.
So far, we have encountered data in three forms:

* NeuroidAssembly: neural data recorded from "neuroids" -- neurons or their analogues such as multi-unit activity from
  Utah array electrodes. These assemblies typically contain spike rates structured in three dimensions
  :code:`presentation x neuroid x time_bin` where
  the :code:`presentation` dimension represents stimulus presentations (e.g. images x trials),
  the :code:`neuroid` dimension represents e.g. electrodes (with metadata such as neuroid_id and location), and
  the :code:`time_bin` dimension contains information about the start (:code:`time_bin_start`) and
  end (:code:`time_bin_end`) of a time bin of spike rates.
* BehavioralAssembly: behavioral measurements, typically choices in a task structured in one dimension
  :code:`presentation` that represents stimulus presentations (e.g. images x trials, with metadata on the task such
  as the sample object and the distractor object in a match-to-sample task) with the actual choices
  (e.g. "dog"/"cat", "left"/"right") in the assembly values.
* PropertiesAssembly: any kind of data in a pre-processed form, such as a surround suppression index per :code:`neuroid`.

Here is an example of a BehavioralAssembly:

.. code-block:: python

    from brainio.assemblies import BehavioralAssembly
    from brainio.packaging import package_data_assembly

    assembly = BehavioralAssembly(['dog', 'dog', 'cat', 'dog', ...],
                                   coords={
                                       'stimulus_id': ('presentation', ['image1', 'image2', 'image3', 'image4', ...]),
                                       'sample_object': ('presentation', ['dog', 'cat', 'cat', 'dog', ...]),
                                       'distractor_object': ('presentation', ['cat', 'dog', 'dog', 'cat', ...]),
                                       # ...more meta
                                       # Note that meta from the StimulusSet will automatically be merged into the
                                       #  presentation dimension:
                                       #  https://github.com/brain-score/brainio/blob/d0ac841779fb47fa7b8bdad3341b68357c8031d9/brainio/fetch.py#L125-L132
                                   },
                                   dims=['presentation'])
    assembly.name = '<authoryear>'  # give the assembly an identifier name

    # make sure the assembly is what you would expect
    assert len(assembly['presentation']) == 179660
    assert len(set(assembly['stimulus_id'].values)) == 1600
    assert len(set(assembly['choice'].values)) == len(set(assembly['sample_object'].values)) \
           == len(set(assembly['distractor_object'].values)) == 2

    # upload to S3
    package_data_assembly(assembly, assembly_identifier=assembly.name, ,
                          assembly_class='BehavioralAssembly'
                          stimulus_set_identifier=stimuli.name)  # link to the StimulusSet

In our experience, it is generally a good idea to include as much metadata as possible (on both StimulusSet and
Assembly). This will increase the utility of the data and make it a more valuable long-term contribution.


**Unit Tests**:
We ask that packaged stimuli and assemblies are tested so that their validity can be confirmed for a long time, even as
details in the system might change. For instance, we want to avoid accidental overwrite of a packaged experiment,
and the unit tests guard against that.

|UnitTestSupport|

There are already generic tests in place to which you can add your StimulusSet and assembly identifiers:

#. :meth:`tests.test_stimuli.test_list_stimulus_set`
#. :meth:`tests.test_assemblies.test_list_assembly`
#. :meth:`tests.test_assemblies.test_existence`

Simply add your identifiers to the list.

Additionally, you can write your own test method to run some more detailed checks on the validity of StimulusSet and
assembly:

.. code-block:: python

    # in test_stimuli.py
    def test_<authoryear>:
        stimulus_set = brainio.get_stimulus_set('<authoryear>')
        assert len(stimulus_set) == 123  # check number of stimuli
        assert len(set(stimulus_set['stimulus_id'])) == 12  # check number of unique stimuli
        assert set(stimulus_set['object_name']) == {'dog', 'cat'}
        # etc


    # in test_assemblies.py
    def test_<authoryear>:
        assembly = brainscore.get_assembly('<authoryear>')
        np.testing.assert_array_equal(assembly.dims, ['presentation'])
        assert len(set(assembly['stimulus_id'].values)) == 123  # check number of stimuli
        assert len(assembly) == 123456  # check number of trials
        assert assembly.stimulus_set is not None
        assert len(assembly.stimulus_set) == 123  # make sure number of stimuli in stimulus_set lines up with assembly
        # etc


2. Create the benchmark
=======================
The :class:`~brainscore.benchmarks.Benchmark` brings together the experimental paradigm with stimuli,
and a :class:`~brainscore.metrics.Metric` to compare model measurements against experimental data.
The paradigm typically involves telling the model candidate to perform a task or start recording in a particular area,
while looking at images from the previously packaged StimulusSet.
Interacting with the model candidate is agnostic of the specific model and is guided by the
:class:`~brainscore.model_interface.BrainModel` -- all models implement this interface,
and through this interface the benchmark can interact with all current and future model candidates.

Typically, all benchmarks inherit from :class:`~brainscore.benchmarks.BenchmarkBase`, a super-class requesting the
commmonly used attributes. These attributes include

* the *identifier* which uniquely designates the benchmark
* the *version* number which increases when changes to the benchmark are made
* a *ceiling_func* that, when run, returns a ceiling for this benchmark
* the benchmark's *parent* to group under e.g. V1, V2, V4, IT, behavior, or engineering (machine learning benchmarks)
* a *bibtex* that is used to link to the publication from the benchmark and website for further details
  (we are working on crediting benchmark submitters more prominently in addition to only the data source.)

Here is an example of a behavioral benchmark that uses an already defined metric,
:class:`~brainscore.metrics.image_level_behavior.I2n`, to compare image-level behaviors:

.. code-block:: python

    import brainscore
    from brainscore.benchmarks import BenchmarkBase
    from brainscore.benchmarks.screen import place_on_screen
    from brainscore.metrics.image_level_behavior import I2n
    from brainscore.model_interface import BrainModel
    from brainscore.utils import LazyLoad

    # the BIBTEX will be used to link to the publication from the benchmark for further details
    BIBTEX = """@article {AuthorYear,
                    author = {Author},
                    title = {title},
                    year = {2021},
                    url = {link},
                    journal = {bioRxiv}
                }"""


    class AuthorYearI2n(BenchmarkBase):
        def __init__(self):
            self._metric = I2n()  # use a previously defined metric
            # we typically use the LazyLoad wrapper to only load the assembly on demand
            self._fitting_stimuli = LazyLoad(lambda: brainscore.get_stimulus_set('<authoryear>'))
            self._assembly = LazyLoad(lambda: brainscore.get_assembly('<authoryear>'))
            # at what degree visual angle stimuli were presented
            self._visual_degrees = 8
            # how many repeated trials each stimulus was shown for
            self._number_of_trials = 2
            super(AuthorYearI2n, self).__init__(
                identifier='<AuthorYear>-i2n',
                # the version number increases when changes to the benchmark are made; start with 1
                version=1,
                # the ceiling function outputs a ceiling estimate of how reliable the data is, or in other words, how
                # well we would expect the perfect model to perform on this benchmark
                ceiling_func=lambda: self._metric.ceiling(self._assembly),
                parent='behavior',
                bibtex=BIBTEX,
            )

        # The __call__ method takes as input a candidate BrainModel and outputs a similarity score of how brain-like
        # the candidate is under this benchmark.
        # A candidate here could be a model such as CORnet or brain-mapped Alexnet, but importantly the benchmark can be
        # agnostic to the details of the candidate and instead only engage with the BrainModel interface.
        def __call__(self, candidate: BrainModel):
            # based on the visual degrees of the candidate
            fitting_stimuli = place_on_screen(self._fitting_stimuli, target_visual_degrees=candidate.visual_degrees(),
                                              source_visual_degrees=self._visual_degrees)
            candidate.start_task(BrainModel.Task.probabilities, fitting_stimuli)
            stimulus_set = place_on_screen(self._assembly.stimulus_set, target_visual_degrees=candidate.visual_degrees(),
                                           source_visual_degrees=self._visual_degrees)
            probabilities = candidate.look_at(stimulus_set, number_of_trials=self._number_of_trials)
            score = self._metric(probabilities, self._assembly)
            score = self._metric.ceil_score(score, self.ceiling)
            return score


We also need to register the benchmark in the benchmark pool in order to make it accessible by its identifier.
Register the benchmark in the experimental benchmark pool first, we will then run existing models on it, and later
transfer it to the main benchmark pool.

.. code-block:: python

    # in brainscore/benchmarks/__init__.py

    def _experimental_benchmark_pool():
        # ...
        from .<authoryear> import <AuthorYear>I2n
        pool['<authoryear>-i2n'] = LazyLoad(<AuthorYear>I2n)

        return pool


**Unit Tests**

Like with the stimuli and data, we want to ensure the continued validity of the benchmark so that it remains valuable
and can be maintained.

|UnitTestSupport|

We ask that all benchmarks test at least two things:

#. The ceiling value of the benchmark for which the benchmark identifier and expected ceiling can simply be added to
   the :meth:`tests.test_benchmarks.test___init__.test_ceilings` method
#. The score of a couple of models with precomputed features:

The idea for scores of precomputed features is to run a few models on the benchmark, store their features, and test that
the stored features run on the benchmark will reproduce the same score.
These tests are organized in :class:`tests.test_benchmarks.test___init__.TestPrecomputed` where, for both neural and
behavioral benchmarks, standardized functions exist to make these tests as easy as possible.

To add a new test, first store the features of select models.
For neural benchmarks, run the benchmark on a model, then convert the pickled activations from :code:`result_caching`
into :code:`netcdf (.nc)` files:

.. code-block:: python

    import pandas as pd  # makes pickle file loading a little easier, even if not a pandas object

    activations_dir = '~/.result_caching/model_tools.activations.core.ActivationsExtractorHelper._from_paths_stored'
    activations = pd.read_pickle(activations_dir + '/identifier=alexnet,stimuli_identifier=<authoryear>.pkl')['data']
    activations.reset_index(activations.dims).to_netcdf('~/alexnet-<authoryear>.nc')


For behavioral benchmarks, the only way right now is to store the predictions as you score a model on the benchmark,
i.e. either by putting a breakpoint and storing the features or by adding code to the benchmark's :code:`__call__`
method (before model predictions are scored with the metric).

.. code-block:: python

    probabilities.reset_index(probabilities.dims).rename({'choice_':'choice'})\
        .to_netcdf('~/brain-score/tests/test_benchmarks/CORnet-Z-<authoryear>.nc')


Next, upload these precomputed features to :code:`S3://brainscore-unittests/tests/test_benchmarks/`
(account 613927419654).
Please reach out to us so that we can help you with the upload.

To have these precomputed features downloaded when unit tests are run, please add the filenames to the
:code:`test_setup.sh` file.

Finally, add a new method :code:`test_<authoryear-metric>` in
:class:`tests.test_benchmarks.test___init__.TestPrecomputed` which points to the precomputed features file, and tests
that an expected score is output by the benchmark.


3. Submit a pull request with your changes and iterate to finalize
==================================================================
Finally, submit a pull request on https://github.com/brain-score/brain-score/compare with all your changes.
This will trigger server-side unit tests which ensure that all previous as well as newly added unit tests pass
successfully.
Often, this step can highlight some issues in the code, so it can take some iterations on the code to make sure
everything runs smoothly.
Looking at other merged pull requests for reference could be helpful here:
https://github.com/brain-score/brain-score/pulls.
We will also manually review the pull request before merging.

If any stimuli or data should be made public, please let us know so that we can change the corresponding S3 bucket
policy.

After the PR has been merged, we will run all existing models on the new benchmark before making the benchmark public
and integrating it into the set of standardly evaluated benchmarks.
