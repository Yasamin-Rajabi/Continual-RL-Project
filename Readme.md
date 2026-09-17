# Continual RL: task-blind observations and optional policy-space composition

Updated from the uploaded project for HalfCheetah Vel/WindVel, Walker2D,
Hopper, and MetaWorld. This is a modified research implementation, not the
baseline paper authors' official code or a verified reproduction of its results.

Start with [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md),
[EVALUATION_GUIDE.md](EVALUATION_GUIDE.md), and
[VALIDATION_REPORT.md](VALIDATION_REPORT.md).

Default benchmark runs: baseline and combined, each in parameter and policy
composition space. Distil-only and weight-only remain opt-in ablations.
The original overview is retained as README_ORIGINAL.md for provenance.

Use a fresh output directory and retrain. Older task-conditioned checkpoints,
pretrained encoders and metric caches are not compatible.
