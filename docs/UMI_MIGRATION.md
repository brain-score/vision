# Use vision models with UMI

`BrainModel` and `look_at` remain supported for existing vision plugins. Loading these plugins through `brainscore.load_model` supplies the appropriate UMI adapter.

| Goal | Interface |
| --- | --- |
| Score an existing model | `brainscore.score(model_identifier, benchmark_identifier)` |
| Reuse extraction and task helpers | `BrainScoreModel`, with `process`, task setup and recording methods |
| Define custom session behavior | `Subject.interact(session)`, with declared input/output channels |
| Combine recording and interventions | `Experiment` with a compatible protocol and tools |

`BrainScoreModel` is a `Subject` implementation. A native `Subject` does not need `process()` or a region mapping; a benchmark using those methods requires an implementation that provides them.

See the [UMI getting-started guide](https://github.com/KartikP/brainscore-unified/blob/unified-model-interface-v2/docs/getting_started.md). The domain documentation remains useful for existing benchmark and submission workflows.
