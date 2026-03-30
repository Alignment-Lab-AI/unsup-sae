# unsupsae

Unsupervised discovery of good classifiers from Sparse Autoencoder features.

Given a pretrained model with an SAE, this identifies which SAE features are strong linear classifiers for structure in the data — **without labels or supervised training**. The method combines manifold geometry on decoder directions with graph-theoretic analysis of feature co-activation patterns.

## Problem

SAEs extract thousands of features from neural network activations. A small fraction of those features are useful classifiers for any given signal (e.g., topic in text, prosody in audio, style in images). Finding them normally requires labeled data or training a supervisor model. This is expensive when:

- You have many candidate signals to investigate
- Labeled data doesn't exist yet
- You can't afford to train a supervisor for each signal
- You want to audit what a model has learned before committing to labels

This tool finds those features using only the unlabeled activations and the SAE's decoder weight geometry.

## How it works

### Signals (what we measure per feature)

| Signal | Source | What it captures |
|--------|--------|-----------------|
| **kNN stability** | Decoder geometry | Whether a feature's neighborhood is preserved under random projection — real geometric structure vs basis artifact |
| **Procrustes invariance** | Decoder geometry + data | Cross-subsample SVD embedding alignment via orthogonal Procrustes (Q = U·Vᵀ from SVD of Sᵀ·T) — features whose structural role is stable across data splits |
| **Manifold selectivity** | Co-firing graph | Variance of manifold similarity with co-firing partners — features that relate differently to different neighbors are selective, not generic |
| **PageRank centrality** | Co-firing graph | Importance on the manifold-weighted co-firing graph — features on many high-weight paths (equivalent to composed FST paths) |
| **PC alignment** | Data | How much of the feature's variance falls along the top principal components — features aligned with data structure |
| **Activation selectivity** | Data | p(active) × (1 - p(active)) — features that fire on some inputs but not others |
| **Cross-split CV stability** | Data | Consistency of feature-PC correlations across random data splits — features whose relationship to data structure is reproducible |

### Scoring (how we combine them)

Two-stage pipeline:
1. **Geometry gate**: Compute geometry quality = (stability + 0.1) × (Procrustes + 0.1) × (msim_var + 0.05). Features below the 25th percentile are filtered out.
2. **Data-driven ranking**: Surviving features are ranked by data signals (selectivity × PC alignment × frequency × CV stability × graph centrality), with a dampened geometry multiplier (geometry^0.3) so that slightly lower geometry doesn't dominate the ranking.

### Perturbation ensemble

The pipeline runs 15 times across different configurations (PC dimensions ∈ {3, 4, 6, 8, 12} × 3 bootstrap resamples). Feature scores are accumulated with rank-weighted voting. This smooths over sensitivity to any single hyperparameter choice.

### FST connection

The co-firing graph mirrors finite-state transducer composition (OpenFst):
- Features = states, co-firing = arcs with manifold-similarity weights
- Composition chains relationships transitively: if A→B and B→C with high weight, then A→C is a composed path
- PageRank finds features on many high-weight composed paths — the equivalent of shortest-path queries on the composed FST

## Tested results

Evaluated on **GPT-2 small** with **SAE (gpt2-small-res-jb, blocks.8.hook_resid_pre, d_sae=24576)** on **AG News** (4-class text classification, 400 samples). Labels were **never used** in feature discovery — only in evaluation.

### Compositional accuracy (multi-feature logistic regression, 5-fold CV)

The primary benchmark: train a logistic regression classifier using only the top-K selected features, and measure how much of the full model's accuracy they recover.

| Features | Ours (unsup.) | Oracle: coef | Oracle: F-ratio | All active (6433) | Full model |
|----------|--------------|-------------|----------------|-------------------|------------|
| Top-5 | 0.5700 | 0.7250 | 0.6025 | — | — |
| Top-10 | 0.6950 | 0.7750 | 0.7075 | — | — |
| Top-20 | **0.7325** | 0.8100 | 0.7825 | 0.8375 | 0.8275 |

- **20 unsupervised features recover 88% of what all 6433 active features achieve** (0.7325 / 0.8375)
- **90% of the oracle's top-20 performance** (0.7325 / 0.8100)
- The sparse oracle (L1 logistic regression) needs only 16 nonzero features to fit the full problem — 8/20 of our picks overlap with its top-50
- 99.9% savings in manifold similarity computations (21K vs 20M all-pairs)

### Single-feature probe accuracy (for signal analysis)

| Method | Top-10 | Top-20 | Labels used? |
|--------|--------|--------|-------------|
| **Ours (unsupervised)** | **0.3725** | **0.3589** | No |
| Baseline: mean activation | 0.2722 | 0.2894 | No |
| Baseline: activation variance | 0.2795 | 0.3091 | No |
| Baseline: selectivity | 0.3225 | 0.3261 | No |
| Oracle: classifier coefficient | 0.3845 | 0.3614 | Yes |
| Oracle: F-ratio | 0.4055 | 0.3934 | Yes |
| Chance | 0.2500 | 0.2500 | — |

### What it finds vs misses

Top features found: 9127 (F=47.6, probe=0.41), 10925 (F=36.0, probe=0.43), 19315 (F=41.5, probe=0.40), 11328 (F=38.8, probe=0.38), 14253 (F=32.2, probe=0.40).

Main misses: Features 770 (F=168.9, probe=0.47) and 21204 (F=78.4, probe=0.41) — these have lower kNN stability (0.59, 0.52 vs >0.7 for our picks), meaning their decoder geometry is less rotation-invariant despite being highly discriminative in activation space. The gap is concentrated at small K (top-5: 0.57 vs oracle 0.73), suggesting our first few picks prioritize geometric quality over raw discriminative power.

## Usage

```bash
# Setup
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Run (generates figures/ if --save-figures)
python run.py --n-samples 400 --save-figures

# Key arguments
python run.py \
  --model gpt2 \
  --sae-release gpt2-small-res-jb \
  --sae-id blocks.8.hook_resid_pre \
  --n-samples 400 \
  --k 30 \
  --min-cofire 3 \
  --device cuda
```

## Files

```
run.py          — Main entry point: load, discover, evaluate, visualize
engine.py       — MetaconceptEngine: manifold geometry on decoder directions
graph.py        — FeatureGraph: co-firing graph, multi-signal scoring pipeline
visualize.py    — Benchmarking plots (method comparison, signal heatmap, etc.)
template.py     — Original monolithic prototype (kept for reference)
figures/        — Generated visualization outputs
```

## Visualizations

`--save-figures` generates:

- **compositional_accuracy.png** — Primary benchmark: multi-feature classifier accuracy at K=5,10,20
- **method_comparison.png** — Single-feature probe accuracy across all methods
- **signal_heatmap.png** — Per-feature signal breakdown for our top-10 vs oracle top-10
- **overlap_matrix.png** — Feature set overlap between all method pairs
- **score_distributions.png** — F-ratio distributions + per-rank probe accuracy curves
- **lift_curve.png** — Cumulative probe accuracy as features are added
- **signal_space_3d.png** — PCA of 8 scoring signals, showing our picks vs oracle in signal space
- **gap_analysis.png** — Per-signal percentile breakdown for oracle features (found vs missed)

## Known limitations

- The geometry gate filters features whose decoder directions are not rotation-invariant, which can exclude features that are discriminative purely through activation patterns (e.g., feature 770)
- Evaluated on one model/dataset/SAE combination — generalization to other architectures (audio encoders, vision models) is the intended use case but untested
- The manifold similarity computation is the bottleneck (~21K edge evaluations); further sparsification of the co-firing graph could reduce this
- Perturbation ensemble runs 15 scoring passes; wall time is dominated by the initial kNN + manifold weight computation which runs once
