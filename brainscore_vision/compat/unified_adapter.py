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

        The rows are the legacy model's own, in their own float type (float64), put in
        the order of the task's labels; the format keeps them as given. The labels must
        be the labels of the fitting stimuli; otherwise the task is refused before the
        legacy model is set up. The meta of the answer event holds the adapter's own record of
        what the old model returned: the old model's own answer object
        (``meta['legacy_answer']``). The legacy facade gives that object to the old benchmark.
        """
        _shared_labels(task, self.identifier, 'behavior:probabilities')
        if task.fitting_stimuli is None:
            raise ExecutionNotFeasibleError(
                f"Legacy model '{self.identifier}' gives behavior:probabilities from its "
                "probabilities task, which is fitted on TaskContext.fitting_stimuli; the task "
                "gives none.", channels={'behavior:probabilities'}, task_type=task.task_type)
        # The label column of an old probabilities task: image_label, else label (the facade's rule).
        fitting = task.fitting_stimuli
        column = 'image_label' if 'image_label' in fitting.columns else 'label'
        fitted = set(str(label) for label in fitting[column])
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
        # The adapter's own record of what the old model returned, in every session; the legacy
        # facade hands this very object to an old benchmark. The stored answer is the format's.
        meta = {'driver': 'legacy', 'legacy_answer': rows}
        session.respond(StreamEvent('behavior:probabilities',
                                    _probabilities_from_rows(rows, task.response.labels, stimuli),
                                    events[-1].t_ms if events else None, meta=meta))

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
    """The legacy task for a fitted Choice task: ``'probabilities'``, as a legacy forced choice,
    with the fitting settings of the task (``metadata['task_settings']``, as an old benchmark
    passed them to ``start_task``)."""
    return TaskContext(task_type='probabilities', fitting_stimuli=task.fitting_stimuli,
                       label_set=list(task.response.labels), instruction=task.instruction,
                       metadata=dict(task.metadata))


def _presentation_columns(assembly):
    """Every presentation coordinate of ``assembly`` in its order: the index levels,
    then the other coordinates on the presentation dimension.

    An assembly built with ``stimulus_id`` as its only presentation coordinate keeps it as
    the simple index of the dimension; that index is the stimulus ids, also when other
    presentation coordinates were added to it later (``assign_coords``)."""
    index = assembly.indexes.get('presentation')
    names = [name for name in (getattr(index, 'names', None) or ()) if name and name != 'presentation']
    names += [name for name, coord in assembly.coords.items()
              if coord.dims == ('presentation',) and name != 'presentation' and name not in names]
    columns = {name: ('presentation', list(assembly[name].values)) for name in names}
    simple = index is not None and getattr(index, 'nlevels', 1) == 1 and index.name in (None, 'presentation')
    if simple and 'stimulus_id' not in columns:
        columns = {'stimulus_id': ('presentation', list(index)), **columns}
    return columns


def _choice_from_labels(answer, labels):
    """The legacy label task's answer as a stored choice: one-hot rows over ``labels``.

    The legacy answer has dims ``('choice', 'presentation')`` and one label per
    stimulus. A label is inside ``labels`` when it equals one of them (plain equality on
    the old model's own value: the integer 1 is not the text '1', bytes are not text);
    a label outside ``labels`` is an invalid row (NaN). The stored rows keep every
    presentation coordinate of the answer, in its order, the old model's own value of
    each label as the raw answer, and the attributes of the answer except ``response``
    (the key of the stored answer's record, which core sets and compares). Two limits: a
    null label (None or a float NaN, core's test for a raw answer with no answer) is refused,
    since the stored choice format would make it a missing row and the row would not reach
    the old benchmark; a run recorder refuses an answer whose labels are bytes (its codec has
    no encoding for bytes).
    """
    import numpy as np
    from brainscore_core.response_formats import response_record
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    given = list(np.asarray(answer.values).reshape(-1))
    labels = list(labels)
    coords = _presentation_columns(answer)
    one_hot = np.full((len(given), len(labels)), np.nan, dtype=np.float32)
    for row, value in enumerate(given):
        inside = [column for column, label in enumerate(labels) if bool(value == label)]
        if inside:
            one_hot[row] = 0.0
            one_hot[row, inside[0]] = 1.0
    coords.update({'choice': labels, 'raw_answer': ('presentation', given)})
    attrs = {key: value for key, value in answer.attrs.items() if key != 'response'}
    attrs['response'] = response_record('behavior:choice', 'legacy')
    stored = BehavioralAssembly(one_hot, dims=['presentation', 'choice'], coords=coords, attrs=attrs)
    # Core's null test for a raw answer, on the values as the stored answer holds them (the
    # coordinate can promote a value, for example a float32 NaN in an object array to a float).
    for row, value in enumerate(stored['raw_answer'].values):
        if value is None or (isinstance(value, float) and value != value):
            stimulus = coords['stimulus_id'][1][row] if 'stimulus_id' in coords else row
            raise ValueError(
                f"Legacy model's label for stimulus {stimulus!r} is {value!r}: the stored choice "
                "format cannot carry a null label (it would be a missing answer). The old model "
                "must return a label for every row.")
    return stored


# The coordinates core adds to a stored answer; an old answer's own coordinate of such a name
# is left out of the stored answer (core reads its own column of that name).
_STORED_NAMES = ('raw_answer', 'response_state', 'response_reason', 'answer_valid', 'parse_error')


def _probabilities_from_rows(rows, labels, stimuli=None):
    """The legacy probabilities task's rows as stored probabilities: the task's label
    order, the legacy model's own values and float type (float64), the legacy
    presentation coordinates (without the names core uses for its own columns) and each
    source row as its raw answer. The old answer object itself is never changed.

    The stimulus ids are the values that resolve against the presented table's ids (a
    ``stimulus_id`` coordinate or level, or the simple presentation index), not a name or a
    shape. Rows whose ids resolve are answered; the table's other trials are stored as
    missing by the session. An old answer with no resolvable ids answers the table in its
    order when it has one row for each trial: it takes the table's ids. With fewer rows and
    no resolvable ids no row can be matched to a trial: every trial is stored as missing,
    with that reason, and nothing is refused. The format holds one answer for each table row:
    a stimulus answered more often than the table shows it (several trials of one image:
    ``number_of_trials`` with ``require_variance``) has its trials stored as missing, with
    that reason; the old object keeps every trial."""
    import numpy as np
    from brainscore_core.response_formats import response_record
    from brainscore_core.supported_data_standards.brainio.assemblies import BehavioralAssembly
    record = response_record('behavior:probabilities', 'legacy', parameters={'option_order': 'task'})
    from collections import Counter
    repeated = {}
    rows = rows.transpose('presentation', 'choice')
    columns = [str(c) for c in rows['choice'].values]
    source = np.asarray(rows.values)
    ordered = source[:, [columns.index(label) for label in labels]]
    coords = {name: column for name, column in _presentation_columns(rows).items()
              if name not in _STORED_NAMES}
    presented = None if stimuli is None else list(stimuli['stimulus_id'])
    if presented is not None:
        known = set(presented)
        ids = list(coords.get('stimulus_id', ('presentation', []))[1])
        kept = [index for index, stimulus_id in enumerate(ids) if _hashable(stimulus_id) and stimulus_id in known]
        if not kept:
            coords.pop('stimulus_id', None)
            if len(source) != len(presented):
                reason = (f'the old answer has {len(source)} rows for {len(presented)} trials and no '
                          'stimulus id that names a trial of this session, so no row can be matched to a trial')
                return BehavioralAssembly(np.full((len(presented), len(labels)), np.nan), dims=['presentation', 'choice'],
                                          coords={'stimulus_id': ('presentation', presented),
                                                  'raw_answer': ('presentation', [None] * len(presented)),
                                                  'response_reason': ('presentation', [reason] * len(presented)),
                                                  'choice': list(labels)}, attrs={'response': record})
            coords = {'stimulus_id': ('presentation', presented), **coords}
            kept = list(range(len(source)))
        else:
            # One answer for each table row: a stimulus the old model answered more often than
            # the table shows it (several trials of one image, number_of_trials with
            # require_variance) cannot be stated as valid rows; its trials are missing, with the cause.
            shown, given = Counter(presented), Counter(ids[index] for index in kept)
            repeated = {stimulus_id: count for stimulus_id, count in given.items() if count > shown[stimulus_id]}
            kept = [index for index in kept if ids[index] not in repeated]
            if len(kept) < len(ids):
                coords = {name: ('presentation', [column[1][index] for index in kept]) for name, column in coords.items()}
        source, ordered = source[kept], ordered[kept]
    coords.update({'choice': list(labels),
                   'raw_answer': ('presentation', [str([float(v) for v in row]) for row in source])})
    if presented is not None and repeated:
        missing = [stimulus_id for stimulus_id in presented if stimulus_id in repeated]
        reasons = [f'the old model gave {repeated[stimulus_id]} answers for this stimulus and the table shows it '
                   f'{shown[stimulus_id]} time(s); the format holds one answer for each trial' for stimulus_id in missing]
        for name, column in coords.items():
            if name != 'choice':
                extra = missing if name == 'stimulus_id' else [None] * len(missing)
                coords[name] = ('presentation', list(column[1]) + extra)
        coords['response_reason'] = ('presentation', [''] * len(source) + reasons)
        ordered = np.concatenate([ordered, np.full((len(missing), len(labels)), np.nan)])
    return BehavioralAssembly(ordered, dims=['presentation', 'choice'], coords=coords, attrs={'response': record})


def _hashable(value) -> bool:
    try:
        hash(value)
    except TypeError:
        return False
    return True


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
    return BehavioralAssembly(one_hot, dims=['presentation', 'choice'], coords={
        'stimulus_id': ('presentation', list(rows['stimulus_id'].values)), 'choice': labels,
        'raw_answer': ('presentation', [str([float(v) for v in row]) for row in values])},
        attrs={'response': response_record(
            'behavior:choice', 'legacy', conversion='argmax', source_format='behavior:probabilities',
            parameters={'tie': 'first', 'option_order': 'legacy'})})
