# Use vision models with UMI

`BrainModel` and `look_at` remain supported for existing vision plugins. Loading these plugins through `brainscore.load_model` supplies the appropriate UMI adapter.

| Goal | Interface |
| --- | --- |
| Score an existing model | `brainscore.score(model_identifier, benchmark_identifier)` |
| Reuse extraction and task helpers | `brainscore.BrainScoreModel`, run through sessions (`neural_response`, `behavioral_response`) |
| Define custom session behavior | `Subject.participate(session)`, with declared input/output channels |
| Combine recording and interventions | `Experiment` with a compatible protocol and tools |

`brainscore.BrainScoreModel` is a `Subject` implementation. A native `Subject` does not need `process()` or a region mapping: `brainscore.score` gives an existing benchmark that calls such methods a facade that runs each call as a session through `participate`.

See the [UMI manual](https://github.com/KartikP/brainscore-unified/blob/unified-model-interface-v2/docs/README.md). The domain documentation remains useful for existing benchmark and submission workflows.
