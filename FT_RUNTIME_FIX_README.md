# FT post-hoc runtime compatibility fix

This patch changes only checkpoint-validation logic used for post-hoc Forward Transfer (FT) metrics.

## Why the old evaluator rejected valid scratch curves

`metrics.py` asked `scratch_baselines.checkpoint_matches()` to compare each already-trained scratch checkpoint against the Python/package versions of the *current evaluation process*. That is too strict for FT because FT only reads learning curves that were recorded during training.

## New behavior

- Training/resume checkpoint reuse is unchanged and remains strict about the current runtime.
- FT metric validation ignores the runtime of the process recomputing metrics.
- FT instead compares each scratch run's saved `runtime_versions` against the saved `runtime_versions` of the corresponding continual-training run.
- Only first encounters of unseen tasks after sequence position 0 are validated, because those are the only positions used by FT.
- Training configuration/source/checkpoint provenance checks remain enabled.

If scratch and continual training really used different Python/package runtimes, FT is still rejected and the error now prints both saved runtime dictionaries.

## Recompute existing results

Use the existing evaluation-only launcher:

```bash
cd "$HOME/Cont/Continual-RL-Project/half-cheetah"
bash job_eval.sh
```

`job_eval.sh` contains `--skip-training`, so it will not retrain continual agents. Keep:

```text
--test-adapt-steps 5000
--frozen-eval-policy pool
```

for the alpha-only retention evaluation used for A_N / FG / BWT. FT remains computed from saved training curves and matched scratch curves.
