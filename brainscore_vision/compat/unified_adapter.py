"""The adapter that makes a legacy vision ``BrainModel`` a subject.

A session reaches the legacy model as the ``start_task``, ``start_recording``
and ``look_at`` calls a legacy benchmark makes. A legacy benchmark that is
given the adapter directly calls ``look_at`` on it and gets the same calls.
"""

from typing import Any, Dict, List, Set

from brainscore_core.compatibility import ExecutionNotFeasibleError
from brainscore_core.contract import Subject, TaskContext
from brainscore_core.streaming import StreamEvent
from brainscore_core.streaming_helpers import (
    declared_only,
    drain_session_inputs,
    measurement_settings,
    session_stimuli,
)


class VisionModelAdapter(Subject):

    def __init__(self, legacy_model):
        self._legacy = legacy_model
        # Setup the legacy model currently has, so a repeated identical
        # configuration (a facade session re-sends it per measurement) is
        # applied once, as the legacy benchmark applied it.
        self._configured_task = None
        # The task key holds id(fitting); keeping the object alive keeps that
        # id from being reused by another fitting set.
        self._configured_fitting = None
        self._configured_recording = None

    @property
    def identifier(self) -> str:
        # Vision's identifier is a @property
        return self._legacy.identifier

    def _legacy_layer_map(self):
        if hasattr(self._legacy, 'layer_model') and hasattr(self._legacy.layer_model, 'region_layer_map'):
            return self._legacy.layer_model.region_layer_map
        return {}

    @property
    def region_layer_map(self) -> Dict[str, str]:
        """The region -> layer commitments the legacy model has made so far."""
        return dict(self._legacy_layer_map())

    def _mappable_regions(self) -> List[str]:
        """Every region the legacy model is wired to map, without a layer search.

        The committed regions come first. A lazy map (a ``ModelCommitment``
        with no ``region_layer_map/<id>.json`` and no ``region_layer_map=``)
        also declares, in its public ``region_benchmarks``, the regions its
        layer selection can search. The search itself runs where it always ran:
        at the first legacy ``look_at`` of a region.
        """
        layer_map = self._legacy_layer_map()
        regions = list(layer_map)  # dict iteration: committed regions only, no search
        # Read as a public declaration; a plain committed dict has none.
        searchable = getattr(layer_map, 'region_benchmarks', None) or {}
        regions += [region for region in searchable if region not in regions]
        return regions

    @property
    def probe_safe_regions(self) -> List[str]:
        """Regions with a committed layer: recording them starts no layer search.

        The memory pre-flight probes only these regions on a benchmark that
        does not name the region it measures.
        """
        return list(self._legacy_layer_map())

    @property
    def in_channels(self) -> Set[str]:
        return {'vision:image'}

    @property
    def required_channels(self) -> Set[str]:
        # Legacy vision models are unimodal: they can only process images.
        # Requiring images makes pre-flight reject a benchmark that does not
        # provide them.
        return {'vision:image'}

    @property
    def out_channels(self) -> Set[str]:
        # A legacy BrainModel answers behavioral tasks as well as recording.
        # Neural channels come from the regions the legacy model can map, so a
        # lazy layer search declares the same channels before and after it runs.
        # A Choice task is answered on behavior:choice and behavior:probabilities from
        # the legacy model's own tasks (label, probabilities).
        return ({f'neural:{region}' for region in self._mappable_regions()}
                | {'behavior', 'behavior:choice', 'behavior:probabilities'})

    def process(self, stimuli, number_of_trials=1, require_variance=False) -> Any:
        if require_variance:
            return self._legacy.look_at(stimuli, number_of_trials, require_variance=True)
        return self._legacy.look_at(stimuli, number_of_trials)

    def participate(self, session) -> None:
        """Run one session as the legacy ``look_at`` calls it stands for.

        The session's task, recording target, time bins, trial count and
        variance request reach the legacy model through the same calls a
        legacy benchmark makes. Setup is applied when it changes, not on every
        measurement. Only declared input streams reach the legacy model.
        A session that requests several ``neural:<region>`` channels gets, for
        each region in request order, the ``start_recording`` and ``look_at``
        that a session for that region alone gets.
        """
        session = session.declare_inputs(self.in_channels)
        channels = tuple(session.outputs or ())
        if not channels:
            raise ValueError('A session for a legacy vision model must request an output channel.')
        if len(channels) > 1 and not all(c.startswith('neural:') for c in channels):
            # A refusal at session start, before any input is read: an error of the run.
            task = session.task
            raise ExecutionNotFeasibleError(
                f'Session refused (task type '
                f'{getattr(task, "task_type", None) if task is not None else "none"!r}): a legacy '
                f'vision model answers one output channel per look_at, or several '
                f'neural:<region> channels; this session requests {sorted(channels)}. The pair '
                f'passed the declared compatibility check, so this limit is not expressed in '
                f'the declarations of the subject or the benchmark; there is no result.',
                channels=channels,
                task_type=getattr(task, 'task_type', None) if task is not None else None)
        undeclared = [channel for channel in channels if channel not in self.out_channels]
        if undeclared:
            raise ValueError(f'Undeclared output channel {undeclared[0]!r}; declared: '
                             f'{sorted(self.out_channels)}')
        task = session.task
        settings = measurement_settings(session)
        if channels == ('behavior:choice',):
            self._answer_choice(session, task, settings)
            return
        if channels == ('behavior:probabilities',):
            self._answer_probabilities(session, task, settings)
            return
        if task is not None and task.task_type != 'passive':
            self.start_task(task)
        elif self._configured_task is not None and self._configured_task[0] != 'passive':
            # An earlier session left the legacy model answering a task. A
            # passive session leaves it the legacy way, as a legacy benchmark
            # does with start_task('passive'). A fresh subject gets no call.
            self.start_task('passive')
        events = drain_session_inputs(session)
        stimuli = session_stimuli(session, events)
        for channel in channels:
            if channel.startswith('neural:'):
                self.start_recording(channel.split(':', 1)[1], time_bins=session.time_bins)
            output = self.process(stimuli,
                                  number_of_trials=settings.get('number_of_trials', 1),
                                  require_variance=settings.get('require_variance', False))
            session.respond(StreamEvent(channel, output,
                                        events[-1].t_ms if events else None,
                                        meta={'driver': 'legacy'}))

    def _answer_choice(self, session, task, settings) -> None:
        """Answer a Choice task on ``behavior:choice`` from the legacy model's own task.

        With fitting stimuli the legacy model runs the task a legacy benchmark sends
        for a fitted forced choice, ``start_task('probabilities', fitting_stimuli)``;
        its probability rows are turned into the choice by core's ``argmax`` in the
        legacy model's column order (derived). Without fitting stimuli it runs the
        label task, ``start_task('label', labels)``, whose answer is one label per
        stimulus (native): the stored rows are one-hot over the task's labels and keep
        every presentation coordinate of the legacy answer. An error of the set-up or
        the fit passes through unchanged, as on the bare-behavior path.
        """
        _shared_labels(task, self.identifier, 'behavior:choice')
        if task.fitting_stimuli is None:
            self.start_task('label', list(task.response.labels))
            answer = lambda rows: _choice_from_labels(rows, task.response.labels)
        else:
            self.start_task(_legacy_choice_task(task))
            answer = lambda rows: _choice_from_rows(rows, task.response.labels)
        events = drain_session_inputs(session)
        stimuli = session_stimuli(session, events)
        rows = self.process(stimuli, number_of_trials=settings.get('number_of_trials', 1),
                            require_variance=settings.get('require_variance', False))
        session.respond(StreamEvent('behavior:choice', answer(rows),
                                    events[-1].t_ms if events else None, meta={'driver': 'legacy'}))

    def _answer_probabilities(self, session, task, settings) -> None:
        """Answer a Choice task on ``behavior:probabilities`` from the legacy probabilities task.

        The rows are the legacy model's own, put in the order of the task's labels and
        cast to float32, the dtype of the format (the record says so). The labels must
        be the labels of the fitting stimuli; otherwise the task is refused before the
        legacy model is set up.
        """
        _shared_labels(task, self.identifier, 'behavior:probabilities')
        if task.fitting_stimuli is None:
            raise ExecutionNotFeasibleError(
                f"Legacy model '{self.identifier}' gives behavior:probabilities from its "
                "probabilities task, which is fitted on TaskContext.fitting_stimuli; the task "
                "gives none.", channels={'behavior:probabilities'}, task_type=task.task_type)
        fitted = set(str(label) for label in task.fitting_stimuli['image_label'])
        if fitted != set(task.response.labels):
            raise ExecutionNotFeasibleError(
                f"Legacy model '{self.identifier}': the probabilities task is fitted on the "
                f"labels {sorted(fitted)}, but the Choice task names {sorted(task.response.labels)}.",
                channels={'behavior:probabilities'}, task_type=task.task_type)
        self.start_task(_legacy_choice_task(task))
        events = drain_session_inputs(session)
        stimuli = session_stimuli(session, events)
        rows = self.process(stimuli, number_of_trials=settings.get('number_of_trials', 1),
                            require_variance=settings.get('require_variance', False))
        session.respond(StreamEvent('behavior:probabilities',
                                    _probabilities_from_rows(rows, task.response.labels),
                                    events[-1].t_ms if events else None, meta={'driver': 'legacy'}))

    # Backwards-compatible: existing benchmarks call look_at() directly
    def look_at(self, stimuli, number_of_trials=1, require_variance=False):
        return self.process(stimuli, number_of_trials, require_variance)

    def start_task(self, task_or_type, fitting_stimuli=None,
                   **kwargs) -> None:
        # Accept both calling conventions:
        #   New: start_task(TaskContext(...)); fitting settings a legacy
        #        benchmark passed travel in metadata['task_settings']
        #   Legacy: start_task(task, fitting_stimuli, **settings)
        if isinstance(task_or_type, TaskContext):
            task_type = task_or_type.task_type
            fitting = task_or_type.fitting_stimuli
            settings = dict(task_or_type.metadata.get('task_settings', {}))
        else:
            task_type, fitting, settings = task_or_type, fitting_stimuli, dict(kwargs)
        key = (task_type, id(fitting), repr(sorted(settings.items())))
        if key == self._configured_task:
            return
        # Forward what the benchmark passed: a task without fitting data
        # (Hebart2023 odd_one_out) reaches a start_task(self, task) unchanged.
        if fitting is None:
            self._legacy.start_task(task_type, **settings)
        else:
            self._legacy.start_task(task_type, declared_only(fitting, self.in_channels), **settings)
        self._configured_task = key
        self._configured_fitting = fitting

    def start_recording(self, recording_target: str,
                        time_bins=None, recording_type=None, **kwargs) -> None:
        # Legacy BrainModel.start_recording(recording_target, time_bins)
        # Default time_bins for vision is [(70, 170)]
        # Explicit check: an ndarray of bins (temporal benchmarks) has no truth value.
        bins = [(70, 170)] if time_bins is None or len(time_bins) == 0 else time_bins
        key = (repr(recording_target), _time_bins_key(bins))
        if key == self._configured_recording:
            return
        self._legacy.start_recording(recording_target, bins)
        self._configured_recording = key

    def visual_degrees(self):
        return self._legacy.visual_degrees()

    def reset(self) -> None:
        reset = getattr(self._legacy, 'reset', None)
        if callable(reset):
            reset()
        self._configured_task = None
        self._configured_fitting = None
        self._configured_recording = None

    @property
    def legacy_model(self):
        """The wrapped legacy model, for tools that need its own attributes.

        Unknown attributes are no longer forwarded implicitly: a benchmark
        sees only what the adapter declares (Contract Rules v3, rule 2.1).
        """
        return self._legacy


def _time_bins_key(bins):
    """An exact setup key for time bins.

    Lists and tuples keep their ``repr`` key. An ndarray (temporal benchmarks)
    gets a structural key that does not depend on NumPy display precision.
    """
    import numpy as np
    if not isinstance(bins, np.ndarray):
        return repr(bins)
    return _exact(bins)


def _exact(value):
    import numpy as np
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return ('ndarray', 'O', value.shape, tuple(_exact(item) for item in value.ravel()))
        return ('ndarray', value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(_exact(item) for item in value))
    if isinstance(value, np.generic):
        return ('scalar', value.dtype.str, value.tobytes())
    return (type(value).__name__, value)


def _shared_labels(task, identifier, channel):
    """Refuse a task that is not a Choice with one list of options for every stimulus."""
    from brainscore_core.responses import Choice
    response = getattr(task, 'response', None)
    if not isinstance(response, Choice) or isinstance(response.labels, str):
        raise ExecutionNotFeasibleError(
            f"Legacy model '{identifier}' answers {channel} only for a Choice task with "
            f"shared options; got {response!r}.", channels={channel},
            task_type=getattr(task, 'task_type', None))


def _legacy_choice_task(task):
    """The legacy task for a fitted Choice task: ``'probabilities'``, as a legacy forced choice."""
    return TaskContext(task_type='probabilities', fitting_stimuli=task.fitting_stimuli,
                       label_set=list(task.response.labels), instruction=task.instruction)


def _presentation_columns(assembly):
    """Every presentation coordinate of ``assembly`` in its order: the index levels,
    then the other coordinates on the presentation dimension."""
    index = assembly.indexes.get('presentation')
    names = [name for name in (getattr(index, 'names', None) or ()) if name and name != 'presentation']
    names += [name for name, coord in assembly.coords.items()
              if coord.dims == ('presentation',) and name != 'presentation' and name not in names]
    if not names and index is not None:
        return {'stimulus_id': ('presentation', list(index))}
    return {name: ('presentation', list(assembly[name].values)) for name in names}


def _choice_from_labels(answer, labels):
    """The legacy label task's answer as a stored choice: one-hot rows over ``labels``.

    The legacy answer has dims ``('choice', 'presentation')`` and one label per
    stimulus. The stored rows keep every presentation coordinate of the answer, in its
    order, and the label itself as the raw answer. A label outside ``labels`` is an
    invalid row (NaN).
    """
    import numpy as np
    from brainscore_core.response_formats import response_record
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    chosen = [str(label) for label in np.asarray(answer.values).reshape(-1)]
    labels = list(labels)
    one_hot = np.full((len(chosen), len(labels)), np.nan, dtype=np.float32)
    for row, label in enumerate(chosen):
        if label in labels:
            one_hot[row] = 0.0
            one_hot[row, labels.index(label)] = 1.0
    coords = _presentation_columns(answer)
    coords.update({'choice': labels, 'raw_answer': ('presentation', chosen)})
    return BehavioralAssembly(one_hot, dims=['presentation', 'choice'], coords=coords,
                              attrs={'response': response_record('behavior:choice', 'legacy')})


def _probabilities_from_rows(rows, labels):
    """The legacy probabilities task's rows as stored probabilities: the task's label
    order, float32 (the format's dtype; the record names the cast), the legacy
    presentation coordinates and each source row as its raw answer."""
    import numpy as np
    from brainscore_core.response_formats import response_record
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    rows = rows.transpose('presentation', 'choice')
    columns = [str(c) for c in rows['choice'].values]
    source = np.asarray(rows.values)
    ordered = source[:, [columns.index(label) for label in labels]]
    coords = _presentation_columns(rows)
    coords.update({'choice': list(labels),
                   'raw_answer': ('presentation', [str([float(v) for v in row]) for row in source])})
    return BehavioralAssembly(ordered.astype(np.float32), dims=['presentation', 'choice'], coords=coords,
                              attrs={'response': response_record(
                                  'behavior:probabilities', 'legacy',
                                  parameters={'cast': f'{source.dtype} to float32',
                                              'option_order': 'task'})})


def _choice_from_rows(rows, labels):
    """The legacy model's probability rows as a stored choice: one-hot rows over ``labels``.

    The choice is core's ``argmax`` of each row in the legacy model's own column order
    (ties to the first column, as the old benchmark's argmax); the record says the
    legacy model's task gave the rows and the argmax derived the choice.
    """
    import numpy as np
    from brainscore_core.response_formats import argmax, response_record
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    values = np.asarray(rows.transpose('presentation', 'choice').values, dtype=float)
    columns = [str(c) for c in rows['choice'].values]
    labels = list(labels)
    indices, valid = argmax(values)
    one_hot = np.full((len(values), len(labels)), np.nan, dtype=np.float32)
    for row, (index, ok) in enumerate(zip(indices, valid)):
        if ok and columns[index] in labels:
            one_hot[row] = 0.0
            one_hot[row, labels.index(columns[index])] = 1.0
    coords = _presentation_columns(rows)
    coords.update({'choice': labels,
                   'raw_answer': ('presentation', [str([float(v) for v in row]) for row in values])})
    return BehavioralAssembly(one_hot, dims=['presentation', 'choice'], coords=coords,
        attrs={'response': response_record(
            'behavior:choice', 'legacy', conversion='argmax', source_format='behavior:probabilities',
            parameters={'tie': 'first', 'option_order': 'legacy'})})
