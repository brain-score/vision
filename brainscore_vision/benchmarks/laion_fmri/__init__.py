"""LAION-fMRI benchmark registration.

Benchmark registry: 36 headline variants matching the authors' re:vision framework.

Headline ridge encoding variants — predict per-voxel responses from model features.
Two regression flavours per cell: `-ridge` (fixed alpha=1, dual form, fastest) and
`-ridgecv` (per-fit CV alpha selection from a 21-value log-spaced sweep).
  - Per-subject pool (5,833 stim/subject, drives the leaderboard rank):
      LAION_fMRI_persubject.{V1,V2,V4,IT}-{tau,ood}-{ridge,ridgecv}  (16 variants)
  - Shared pool (1,492 stim, cross-subject comparison):
      LAION_fMRI.{V1,V2,V4,IT}-{tau,ood}-{ridge,ridgecv}             (16 variants)

Headline RSA variants — representational alignment via RDM Spearman:
  - Shared pool only (Nili ceiling requires shared stim across subjects):
      LAION_fMRI.{V1,V2,V4,IT}-rdm-pearson                            (4 variants)

Total registered: 36.

Available via the public API but intentionally not registered to keep the
leaderboard focused:
  - `cluster_k5` splits (re:vision Method 2; tracks `ood` ~1:1 in IT, 5x compute):
      LAIONfMRIClusterCV('V1') and friends
  - 9 per-OOD-category breakdowns:
      LAIONfMRI('IT', 'ood_shape'), LAIONfMRI('IT', 'ood_gabor'), ...
  - IT_full ablation (V4 ∪ IT union):
      LAIONfMRI('IT_full', 'tau'), LAIONfMRIRSA('IT_full'), ...
  - V3 — not in Brain-Score's canonical region set; skipped here

NOTE: Brain-Score's plugin discovery does a literal text-grep for
`benchmark_registry['<identifier>']` lines, so each variant must be listed
explicitly here.
"""

from brainscore_vision import benchmark_registry
from .benchmark import LAIONfMRI, LAIONfMRIRSA

# ── Per-subject pool ridge variants (HEADLINE) ────────────────────────────
# Each subject's 5,833-image pool (1,121 shared + 4,712 subject-unique).
# Exposes generalization to subject-specific stimulus distributions —
# where weak models collapse and only strong models hold their score.
benchmark_registry['LAION_fMRI_persubject.V1-tau-ridge'] = lambda: LAIONfMRI('V1', 'tau', dataset_prefix='LAION_fMRI_persubject')
benchmark_registry['LAION_fMRI_persubject.V1-ood-ridge'] = lambda: LAIONfMRI('V1', 'ood', dataset_prefix='LAION_fMRI_persubject')
benchmark_registry['LAION_fMRI_persubject.V2-tau-ridge'] = lambda: LAIONfMRI('V2', 'tau', dataset_prefix='LAION_fMRI_persubject')
benchmark_registry['LAION_fMRI_persubject.V2-ood-ridge'] = lambda: LAIONfMRI('V2', 'ood', dataset_prefix='LAION_fMRI_persubject')
benchmark_registry['LAION_fMRI_persubject.V4-tau-ridge'] = lambda: LAIONfMRI('V4', 'tau', dataset_prefix='LAION_fMRI_persubject')
benchmark_registry['LAION_fMRI_persubject.V4-ood-ridge'] = lambda: LAIONfMRI('V4', 'ood', dataset_prefix='LAION_fMRI_persubject')
benchmark_registry['LAION_fMRI_persubject.IT-tau-ridge'] = lambda: LAIONfMRI('IT', 'tau', dataset_prefix='LAION_fMRI_persubject')
benchmark_registry['LAION_fMRI_persubject.IT-ood-ridge'] = lambda: LAIONfMRI('IT', 'ood', dataset_prefix='LAION_fMRI_persubject')

benchmark_registry['LAION_fMRI_persubject.V1-tau-ridgecv'] = lambda: LAIONfMRI('V1', 'tau', metric_type='ridgecv', dataset_prefix='LAION_fMRI_persubject')
benchmark_registry['LAION_fMRI_persubject.V1-ood-ridgecv'] = lambda: LAIONfMRI('V1', 'ood', metric_type='ridgecv', dataset_prefix='LAION_fMRI_persubject')
benchmark_registry['LAION_fMRI_persubject.V2-tau-ridgecv'] = lambda: LAIONfMRI('V2', 'tau', metric_type='ridgecv', dataset_prefix='LAION_fMRI_persubject')
benchmark_registry['LAION_fMRI_persubject.V2-ood-ridgecv'] = lambda: LAIONfMRI('V2', 'ood', metric_type='ridgecv', dataset_prefix='LAION_fMRI_persubject')
benchmark_registry['LAION_fMRI_persubject.V4-tau-ridgecv'] = lambda: LAIONfMRI('V4', 'tau', metric_type='ridgecv', dataset_prefix='LAION_fMRI_persubject')
benchmark_registry['LAION_fMRI_persubject.V4-ood-ridgecv'] = lambda: LAIONfMRI('V4', 'ood', metric_type='ridgecv', dataset_prefix='LAION_fMRI_persubject')
benchmark_registry['LAION_fMRI_persubject.IT-tau-ridgecv'] = lambda: LAIONfMRI('IT', 'tau', metric_type='ridgecv', dataset_prefix='LAION_fMRI_persubject')
benchmark_registry['LAION_fMRI_persubject.IT-ood-ridgecv'] = lambda: LAIONfMRI('IT', 'ood', metric_type='ridgecv', dataset_prefix='LAION_fMRI_persubject')

# ── Shared pool ridge variants (cross-subject comparison) ─────────────────
# Same 1,492 images across every subject — comparable to Allen2022 / Hebart2023.
benchmark_registry['LAION_fMRI.V1-tau-ridge'] = lambda: LAIONfMRI('V1', 'tau')
benchmark_registry['LAION_fMRI.V1-ood-ridge'] = lambda: LAIONfMRI('V1', 'ood')
benchmark_registry['LAION_fMRI.V2-tau-ridge'] = lambda: LAIONfMRI('V2', 'tau')
benchmark_registry['LAION_fMRI.V2-ood-ridge'] = lambda: LAIONfMRI('V2', 'ood')
benchmark_registry['LAION_fMRI.V4-tau-ridge'] = lambda: LAIONfMRI('V4', 'tau')
benchmark_registry['LAION_fMRI.V4-ood-ridge'] = lambda: LAIONfMRI('V4', 'ood')
benchmark_registry['LAION_fMRI.IT-tau-ridge'] = lambda: LAIONfMRI('IT', 'tau')
benchmark_registry['LAION_fMRI.IT-ood-ridge'] = lambda: LAIONfMRI('IT', 'ood')

benchmark_registry['LAION_fMRI.V1-tau-ridgecv'] = lambda: LAIONfMRI('V1', 'tau', metric_type='ridgecv')
benchmark_registry['LAION_fMRI.V1-ood-ridgecv'] = lambda: LAIONfMRI('V1', 'ood', metric_type='ridgecv')
benchmark_registry['LAION_fMRI.V2-tau-ridgecv'] = lambda: LAIONfMRI('V2', 'tau', metric_type='ridgecv')
benchmark_registry['LAION_fMRI.V2-ood-ridgecv'] = lambda: LAIONfMRI('V2', 'ood', metric_type='ridgecv')
benchmark_registry['LAION_fMRI.V4-tau-ridgecv'] = lambda: LAIONfMRI('V4', 'tau', metric_type='ridgecv')
benchmark_registry['LAION_fMRI.V4-ood-ridgecv'] = lambda: LAIONfMRI('V4', 'ood', metric_type='ridgecv')
benchmark_registry['LAION_fMRI.IT-tau-ridgecv'] = lambda: LAIONfMRI('IT', 'tau', metric_type='ridgecv')
benchmark_registry['LAION_fMRI.IT-ood-ridgecv'] = lambda: LAIONfMRI('IT', 'ood', metric_type='ridgecv')

# ── Shared pool RSA variants (representational alignment) ─────────────────
# Per-subject RDM × model-RDM Spearman r, averaged across 5 subjects.
# Only registered for the shared pool — Nili ceiling needs shared stimuli.
benchmark_registry['LAION_fMRI.V1-rdm-pearson'] = lambda: LAIONfMRIRSA('V1')
benchmark_registry['LAION_fMRI.V2-rdm-pearson'] = lambda: LAIONfMRIRSA('V2')
benchmark_registry['LAION_fMRI.V4-rdm-pearson'] = lambda: LAIONfMRIRSA('V4')
benchmark_registry['LAION_fMRI.IT-rdm-pearson'] = lambda: LAIONfMRIRSA('IT')
