# FT / paper-metric recomputation

This patch makes checkpoint identity ignore pure `run_sac.py` writer/logging changes while still rejecting real training-source changes. Existing pre-CSV `run_manifest.json` files remain compatible if the only source difference is the SummaryWriter -> CsvSummaryWriter change.

For HalfCheetah, after the matching scratch jobs have completed, run:

```bash
cd "$HOME/Cont/Continual-RL-Project/half-cheetah"
bash job_eval.sh
```

`job_eval.sh` is a copy of `job.sh` with `--skip-training` added. It will never train missing/stale agents; it only accepts existing compatible checkpoints, reruns retention/survey evaluation, and recomputes FT from the saved learning curves.

The current HalfCheetah evaluation arguments retain:

```text
--test-adapt-steps 5000
--frozen-eval-policy pool
```

so A_N/FG/BWT use the finalized pool with alpha-only test adaptation. FT is computed from the periodic training-time `charts/test_success` and `charts/test_episodic_return` curves against matched scratch curves. `metrics.py` reads TensorBoard first and falls back to `scalars.csv` if needed.

Paper-facing per-seed outputs are under each method directory:

```text
plots/<suite>/survey_metrics.csv
plots/<suite>/summary_metrics.csv
plots/<suite>/retention_data/*.json
plots/<suite>/survey_metrics/*.json
```

To create one mean/std table across seeds:

```bash
python collect_paper_metrics.py \
  --experiment-root "$HOME/Cont/Continual-RL-Project/crl_experiments/ethos_student_halfcheetah_windvel_80k" \
  --suite halfcheetah_wind_vel \
  --eval-mode deterministic
```

Use deterministic and stochastic as separate protocols; do not average them together.
