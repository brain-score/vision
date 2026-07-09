# Vision interface migration to UMI

BrainModel and look_at are the legacy vision API. They remain valid for
existing vision plugins and are adapted when loaded through the unified
registry.

For new cross-domain work:

| Legacy vision API | Unified Model Interface |
| --- | --- |
| BrainModel | Subject or BrainScoreModel |
| candidate.look_at(stimuli) | candidate.process(stimuli) |
| vision-only load_model | brainscore.load_model |
| benchmark(candidate) | brainscore.score(model_id, benchmark_id) |

Continue in the distribution's unified/docs/getting_started.md and
unified/docs/umi_api_reference.md. The legacy Read the Docs pages are still
useful for vision-specific benchmark and submission details.
