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

Before executing the packaging methods to actually upload to S3, please check in with us via
`Slack or Github Issue <https://www.brain-score.org/community>`_ so that we can give you access.
With the credentials, you can then configure the awscli (:code:`pip install awscli`, :code:`aws configure` using region :code:`us-east-1`,
output format :code:`json`) to make the packaging methods upload successfully.

**StimulusSet**:
The StimulusSet contains the stimuli that were used in the experiment as well as any kind of metadata for the stimuli.
Below is a slim example of creating and uploading a StimulusSet. The :code:`package_stimulus_set` method returns the
AWS metadata needed in the :code:`data/__init__.py` file (such as :code:`sha1` and the :code:`version_id`).
In this example, we store the metadata in the :code:`packaged_stimulus_metadata` variable.

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

    packaged_stimulus_metadata = package_stimulus_set(catalog_name=None, proto_stimulus_set=stimuli,
                                 stimulus_set_identifier=stimuli.name, bucket_name="brainio-brainscore")  # upload to S3


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
    packaged_assembly_metadata = package_data_assembly(proto_data_assembly=assembly, assembly_identifier=assembly.name,
                                 stimulus_set_identifier=stimuli.name,  # link to the StimulusSet packaged above
                                 assembly_class_name="BehavioralAssembly", bucket_name="brainio-brainscore",
                                 catalog_identifier=None)

In our experience, it is generally a good idea to include as much metadata as possible (on both StimulusSet and
Assembly). This will increase the utility of the data and make it a more valuable long-term contribution.
Please note that, like in :code:`package_stimulus_set`, The :code:`package_data_assembly` method returns the
AWS metadata needed in the :code:`data/__init__.py` file (such as :code:`sha1` and the :code:`version_id`).
In this example, we store the metadata in the :code:`packaged_assembly_metadata` variable.

You can also put both of these packaging methods inside of one Python file, called e.g. :code:`data_packaging.py`. This file
would then package and upload both the stimulus_set and assembly.

**Unit Tests (test.py)**:
We ask that packaged stimuli and assemblies are tested so that their validity can be confirmed for a long time, even as
details in the system might change. For instance, we want to avoid accidental overwrite of a packaged experiment,
and the unit tests guard against that.

When creating your benchmark, we require you to include a :code:`test.py` file. For what this  file should contain, see
below.

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



**Adding your data to Brain-Score**:
You will also need an :code:`__init__.py` file to go along with your submission. The purpose of this file is to register the
benchmark inside the Brain-Score ecosystem. This involves adding both the stimuli and the data to the
:code:`stimulus_set_registry` and :code:`data_registry` respectively. See below for an example from the data for :code:`Geirhos2021`:

.. code-block:: python

    # assembly
    data_registry['Geirhos2021_colour'] = lambda: load_assembly_from_s3(
        identifier='brendel.Geirhos2021_colour',
        version_id="RDjCFAFt_J5mMwFBN9Ifo0OyNPKlToqf",
        sha1="258862d82467614e45cc1e488a5ac909eb6e122d",
        bucket="brainio-brainscore",
        cls=BehavioralAssembly,
        stimulus_set_loader=lambda: load_stimulus_set('Geirhos2021_colour'),
    )

    # stimulus set
    stimulus_set_registry['Geirhos2021_colour'] = lambda: load_stimulus_set_from_s3(
        identifier='Geirhos2021_colour',
        bucket="brainio-brainscore",
        csv_sha1="9c97c155fd6039a95978be89eb604c6894c5fa16",
        zip_sha1="d166f1d3dc3d00c4f51a489e6fcf96dbbe778d2c",
        csv_version_id="Rz_sX3_48Lg3vtvfT63AFiFslyXaRy.Y",
        zip_version_id="OJh8OmoKjG_7guxLW2fF_GA7ehxbJrvG")


**Data Packaging Summary**:
Part 1 of creating a benchmark involves packaging the stimuli and data, adding a :code:`test.py` file, and adding these stimuli
and data to the :code:`data_registry`. The summary of what to submit is seen below with an example structure of an example
submission structure:

.. code-block:: python

    MyBenchmark2024_stimuli_and_data/
        data/
            data_packaging.py
            test.py
            __init__.py

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


We also need to register the benchmark in the benchmark registry in order to make it accessible by its identifier.
This is done in the :code:`__init__.py` file inside the benchmark directory:

.. code-block:: python

    # in brainscore_vision/benchmarks/mybenchmark/__init__.py

    from brainscore_vision import benchmark_registry

    benchmark_registry['mybenchmark-i2n'] = AuthorYearI2n  # specify the class and not the object, i.e. without `()`


**Unit Tests**

Like with the stimuli and data, we want to ensure the continued validity of the benchmark so that it remains valuable
and can be maintained.
All tests are in your plugin folder's ``test.py``, e.g. ``brainscore_vision/benchmarks/mybenchmark/test.py``.

|UnitTestSupport|

We ask that all benchmarks test at least two things:

#. The ceiling value of the benchmark:

.. code-block:: python

    benchmark = load_benchmark('mybenchmark')
    assert benchmark.ceiling == expected


#. The score of one or more models:

The idea for scores of existing models is to run a few models on the benchmark,
and test that running them on the benchmark will reproduce the same score.

.. code-block:: python

    from brainscore_vision import score

    actual_score = score(model_identifier='your-favorite-model', benchmark_identifier='mybenchmark')
    assert actual_score == expected

**Benchmark Summary**:
To summarize, Part 2 of creating a benchmark involves making the actual benchmark package. This is done by adding the
:code:`benchmark.py` file, the :code:`test.py` file, and registering the benchmark via the :code:`__init__.py` file.

The summary of what to submit is seen below with an example structure of an example
submission structure:

.. code-block:: python

    MyBenchmark2024_stimuli_and_data/
            benchmarks/
                benchmark.py
                test.py
                __init__.py



3. Submit the benchmark and iterate to finalize
==================================================================
Finally, submit your entire model plugin.
You can do this by either opening a pull request on https://github.com/brain-score/vision/compare
or by submitting a zip file containing your plugin (``<zip>/benchmarks/mybenchmark``) on the website.

This will trigger server-side unit tests which ensure that all unit tests pass successfully.
Often, this step can highlight some issues in the code, so it can take some iterations on the code to make sure
everything runs smoothly.
Please open an issue if you run into trouble or get stuck.

If any stimuli or data should be made public, please let us know so that we can change the corresponding S3 bucket
policy.

After the PR has been merged, the submission system will automatically run all existing models on the new benchmark.
