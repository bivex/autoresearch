# Coupled HMM Research Program

Your goal is to optimize the hyper-parameters and structural constraints of a Hierarchical Coupled Hidden Markov Model (HMM) for musical harmony. This model is used to infer chords and keys from note data.

## The Model
- **States**: 12 pitch classes (roots) x 12 chord qualities (Maj, Min, Dim, etc.).
- **Observations**: Multi-hot vectors of 12 tones.
- **Emission Layer**: pnote matrix, representing which notes belong to which chord quality.
- **Transition Layer**: pchange tensor, representing transitions between chords.

## Task
Modify train.py to achieve the lowest val_bpb.
Since the training budget is fixed at 5 minutes, focus on:
1. Priors: The PRIOR_STRENGTH and how it is applied.
2. Structural Anchors: Hardcoding or biasing certain musical transitions.
3. Initialization: How pnote is initialized for different chord types.
4. Constraints: The MAX_SELF_LOOP and other anti-collapse mechanisms.

## Metric
- val_bpb: Negative log-likelihood per note. Lower is better.

Everything in train.py is fair game.
