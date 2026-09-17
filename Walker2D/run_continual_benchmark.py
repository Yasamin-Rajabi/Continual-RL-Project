"""Four-way continual benchmark for HalfCheetahVel and HalfCheetahWindVel.

Defaults run baseline and combined in parameter and policy composition spaces.
Legacy ablations remain selectable using --condition-index 0, 2, or 3.
Policy-space insertion includes an additional projection step documented in
IMPLEMENTATION_NOTES.md; it is not claimed to be the published CKA-RL baseline.

The legacy experimental cases are:

    baseline      = classic CKA vectors + arithmetic merge
    distil_only  = classic CKA vectors + KL distillation merge
    weight_only   = weight-delta vectors + alpha-mass + arithmetic merge
    combined      = weight-delta vectors + alpha-mass + KL distillation merge

Merge-pair selection follows the intended method for each condition:
- baseline and weight_only select the highest-cosine pair in stored parameter space;
- distil_only and combined select the lowest symmetric KL pair between full
  Gaussian policy outputs on balanced stored states.

This file only ORCHESTRATES: it defines the experiment config (CONDITIONS,
argparse), runs training (train_chain, one subprocess call to run_sac.py per
task), and calls into metrics.py / plots.py for everything else:

  - metrics.py computes every number: the full retention matrix (checkpoint x
    unique task, used by the heatmap-style diagnostic plots) AND the four
    survey metrics -- A_N, FG, BWT, FT (two variants) -- per the CRL survey's
    Eq. 7-10. See metrics.py's module docstring for exact formulas and which
    TensorBoard scalar backs p_i(t).
  - plots.py draws every PNG/CSV from whatever metrics.py computed. No
    training, no environment rollouts, no checkpoint loading happens there.

Forward transfer needs a from-scratch, single-task baseline per unique
task_id -- see scratch_baselines.py, which trains and caches those
separately (run it once before this script, or before calling this script's
survey-metrics step, with the SAME --total-timesteps).
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
from collections import OrderedDict

import torch

from tasks import DEFAULT_CONTINUAL_SEQUENCE, TASK_SUITES, get_task_name
import metrics
import plots
import scratch_baselines


CONDITIONS = OrderedDict([
    (
        "baseline",
        {"fusion_mode": "classic_cka", "distillation": False, "use_alpha_mass": False,
         "use_alpha_scale": True, "fix_alpha_scale": False},
    ),
    (
        "distil_only",
        {"fusion_mode": "classic_cka", "distillation": True, "use_alpha_mass": False,
         "use_alpha_scale": True, "fix_alpha_scale": False},
    ),
    (
        "weight_only",
        {"fusion_mode": "weight_delta", "distillation": False, "use_alpha_mass": True,
         "use_alpha_scale": False, "fix_alpha_scale": True},
    ),
    (
        "combined",
        {"fusion_mode": "weight_delta", "distillation": True, "use_alpha_mass": True,
         "use_alpha_scale": False, "fix_alpha_scale": True},
    ),
])


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--task-suites", nargs="+",
        default=["walker2d_dynamics"],
        choices=sorted(TASK_SUITES.keys()),
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--task-sequence", nargs="+", type=int, default=None)
    p.add_argument("--total-timesteps", type=int, default=300000)
    p.add_argument("--learning-starts", type=int, default=5_000)
    p.add_argument("--random-actions-end", type=int, default=5_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--policy-lr", type=float, default=3e-4)
    p.add_argument("--alpha-lr", type=float, default=5e-3)
    p.add_argument("--alpha-mass-lr", type=float, default=None,
                   help="Learning rate for the raw alpha-mass gate; default reuses --alpha-lr (legacy behavior).")
    p.add_argument("--alpha-mass-reg", type=float, default=0.05)
    p.add_argument("--alpha-warmup-steps", type=int, default=5_000)
    p.add_argument("--alpha-entropy-reg", type=float, default=0.01,
                   help="Knowledge-mixture entropy bonus during effective weight-delta warmup; 0 disables it.")
    p.add_argument("--drift-reg", type=float, default=1.0)
    p.add_argument("--distill-encoder-lr-mult", type=float, default=0.1,
                   help="Later-task encoder LR multiplier in distillation modes; 1 disables the slower-encoder optimization.")
    p.add_argument("--q-lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--pool-size", type=int, default=5)
    p.add_argument("--eval-every", type=int, default=10_000)
    p.add_argument("--num-evals", type=int, default=5)
    p.add_argument("--retention-eval-episodes", type=int, default=3)
    p.add_argument("--test-adapt-steps", type=int, default=0,
                   help="Test-time alpha-only adaptation steps for retention and FG/BWT checkpoint evaluation; 0 disables it.")
    p.add_argument("--test-adapt-lr", type=float, default=1e-2,
                   help="Learning rate for test-time alpha adaptation.")
    p.add_argument("--distill-observation-skip", action=argparse.BooleanOptionalAction, default=False,
                   help="Concatenate raw observations to encoder features before policy heads in distillation modes.")
    p.add_argument("--distill-extra-steps", "--distill-buffer-steps", dest="distill_extra_steps",
                   type=int, default=10_000, help="Frozen final B interactions INSIDE total-timesteps.")
    p.add_argument("--composition-spaces", nargs="+", choices=["parameter", "policy"],
                   default=["parameter", "policy"], help="Use parameter alone to disable policy-space runs.")
    p.add_argument("--policy-student-replay", action=argparse.BooleanOptionalAction, default=False,
                   help="Policy-space combined variant: execution mixture acts; novel expert learns from replay; then alpha/gate update separately.")
    p.add_argument("--projection-epochs", type=int, default=16)
    p.add_argument("--projection-max-samples", type=int, default=20_000)
    p.add_argument("--frozen-eval-policy", choices=["pool", "snapshot"], default="pool")
    p.add_argument("--eval-action-mode", choices=["deterministic", "stochastic"], default="deterministic")
    p.add_argument("--skip-forward-transfer", action="store_true",
                   help="Compute A_N/FG/BWT without scratch baselines; leave FT unreported.")
    p.add_argument("--max-distill-buffer", type=int, default=50_000)
    p.add_argument("--similarity-samples", type=int, default=2_048)
    p.add_argument("--balance-source-lineages", action=argparse.BooleanOptionalAction, default=False,
                   help="Balance behavioral-KL/distillation/merge-buffer sampling across original source_ids rather than immediate parents.")
    p.add_argument("--distill-max-samples", type=int, default=20_000)
    p.add_argument("--distill-epochs", type=int, default=16)
    p.add_argument("--distill-lr", type=float, default=5e-4)
    p.add_argument("--distill-batch-size", type=int, default=256)
    p.add_argument("--distill-test-frac", type=float, default=0.2)
    p.add_argument("--analysis-log-every", type=int, default=5_000)
    p.add_argument("--save-root", default="agents_walker2d")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--plots-root", default="plots_walker2d_continual")
    p.add_argument("--analysis-root", default="analysis_runs")
    p.add_argument("--skip-training", action="store_true")
    p.add_argument(
        "--skip-invalid-seeds", action="store_true",
        help="With --skip-training, skip seeds whose checkpoint chain is missing, stale, or unreadable instead of aborting aggregation.",
    )
    p.add_argument("--skip-retention", action="store_true")
    p.add_argument("--skip-survey-metrics", action="store_true")
    p.add_argument(
        "--scratch-seeds", nargs="+", type=int, default=scratch_baselines.DEFAULT_SCRATCH_SEEDS,
        help="Must match the seeds scratch_baselines.py was run with.",
    )
    p.add_argument(
        "--scratch-save-root", default=scratch_baselines.SCRATCH_SAVE_ROOT,
        help="Checkpoint root used by scratch_baselines.py.",
    )
    p.add_argument("--force-retrain", action="store_true")

    # Encoder/algorithm ablations.  Keep the historical frozen-root CLI default
    # for reproducible ablations; the friend's current notebook workflow passes
    # --train-shared explicitly and does not require a pretrained encoder.
    p.add_argument("--train-shared", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--freeze-root-encoder", action=argparse.BooleanOptionalAction, default=False,
                   help="Random-frozen root encoder ablation. Contradicts --train-shared.")
    p.add_argument("--encoder-from-base", action=argparse.BooleanOptionalAction, default=True,
                   help="When the encoder is frozen, reload the immutable root encoder on later tasks.")
    p.add_argument("--pretrained-encoder", default=None,
                   help="fc.pt from tdjepa_pretrain.py. Frozen from task 0 unless --train-shared.")
    p.add_argument("--encoder-linear-out", action=argparse.BooleanOptionalAction, default=False,
                   help="Must match the serialized encoder architecture; also changes the critic.")

    p.add_argument("--condition-alpha-scale", action=argparse.BooleanOptionalAction, default=False,
                   help="Use friend's condition-specific alpha-scale rule: learned for classic CKA, fixed at 5 for weight_delta.")
    p.add_argument("--use-alpha-scale", action=argparse.BooleanOptionalAction, default=False,
                   help="Global learned alpha-scale ablation used when --no-condition-alpha-scale.")
    p.add_argument("--fix-alpha-scale", action=argparse.BooleanOptionalAction, default=False,
                   help="Global fixed alpha-scale=5 ablation used when --no-condition-alpha-scale.")
    p.add_argument("--weight-use-alpha-mass", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable alpha-mass in weight_delta modes. Disable to isolate representation alone.")
    p.add_argument("--constrain-alpha-mass", action=argparse.BooleanOptionalAction, default=True,
                   help="Bounded sigmoid alpha-mass; disable for the legacy ablation.")
    p.add_argument("--distill-select-best-val", action=argparse.BooleanOptionalAction, default=True,
                   help="Restore the lowest held-out-KL distillation epoch; disable for last-epoch legacy behavior.")
    p.add_argument("--collect-cosine-buffers", action=argparse.BooleanOptionalAction, default=False,
                   help="Cosine modes do not need rollout buffers. Enable to retain otherwise-unused cosine-mode tail states.")

    p.add_argument("--alpha", type=float, default=0.2,
                   help="Fixed SAC entropy coefficient, or optional autotune initialization.")
    p.add_argument("--autotune", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--autotune-init-from-alpha", action=argparse.BooleanOptionalAction, default=False,
                   help="If enabled, entropy autotuning starts at --alpha instead of legacy 1.0.")
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--condition-index", nargs="+", type=int, default=[1, 4], choices=[0, 1, 2, 3, 4],
        help="0 = run all 4 CONDITIONS. Otherwise provide one or more of 1-4 "
             "(1=baseline, 2=distil_only, 3=weight_only, 4=combined), e.g. --condition-index 1 4.",
    )
    p.add_argument("--quick-test", action="store_true")
    args = p.parse_args()

    if args.quick_test:
        # Exercises at least one merge without committing to the full paper run.
        args.task_suites = ["walker2d_dynamics"]
        args.seeds = [1]
        args.task_sequence = [0, 1, 2, 3]
        args.total_timesteps = 20_000
        args.learning_starts = 1_000
        args.random_actions_end = 2_000
        args.pool_size = 2
        args.eval_every = 5_000
        args.num_evals = 1
        args.retention_eval_episodes = 1
        args.distill_extra_steps = 1_000
        args.max_distill_buffer = 4_000
        args.similarity_samples = 256
        args.distill_max_samples = 1_000
        args.distill_epochs = 2
        args.analysis_log_every = 2_000

    if args.train_shared and args.freeze_root_encoder:
        p.error("--train-shared and --freeze-root-encoder are contradictory")
    if args.autotune_init_from_alpha and args.alpha <= 0:
        p.error("--alpha must be > 0 with --autotune-init-from-alpha")
    if args.use_alpha_scale and args.fix_alpha_scale:
        p.error("--use-alpha-scale and --fix-alpha-scale are mutually exclusive")
    if args.test_adapt_steps < 0 or args.test_adapt_lr <= 0:
        p.error("--test-adapt-steps must be >=0 and --test-adapt-lr must be >0")
    if args.alpha_entropy_reg < 0:
        p.error("--alpha-entropy-reg must be >= 0")
    if args.distill_encoder_lr_mult <= 0:
        p.error("--distill-encoder-lr-mult must be > 0")
    if 0 in args.condition_index and len(args.condition_index) > 1:
        p.error("--condition-index 0 means all conditions and cannot be combined with other indices")
    if args.policy_student_replay:
        if args.composition_spaces != ["policy"]:
            p.error("--policy-student-replay must be run with --composition-spaces policy only")
        if args.condition_index != [4]:
            p.error("--policy-student-replay is defined for --condition-index 4 (combined) only")

    if not 0 <= args.distill_extra_steps < args.total_timesteps:
        p.error("Require 0 <= B < Delta")
    if args.test_adapt_steps and args.frozen_eval_policy != "pool":
        p.error("Test-time adaptation requires --frozen-eval-policy pool")
    return args


# ==========================================================================
# TRAINING: the only thing this file still does directly.
# ==========================================================================
def _effective_condition_config(args, cfg):
    cfg = dict(cfg)
    if cfg["fusion_mode"] == "weight_delta":
        cfg["use_alpha_mass"] = bool(cfg["use_alpha_mass"] and args.weight_use_alpha_mass)
    if not args.condition_alpha_scale:
        cfg["use_alpha_scale"] = bool(args.use_alpha_scale)
        cfg["fix_alpha_scale"] = bool(args.fix_alpha_scale)
    return cfg


def _expected_training_config(args, suite, task_id, seq_idx, seed, cfg):
    """Subset of run_sac training knobs used to validate resumable checkpoints."""
    return {
        "model_type": "cka-rl",
        "task_suite": suite,
        "task_id": int(task_id),
        "seq_idx": int(seq_idx),
        "seed": int(seed),
        "cuda": not bool(args.cpu),
        "fusion_mode": cfg["fusion_mode"],
        "composition_space": cfg.get("composition_space", "parameter"),
        "policy_student_replay": bool(args.policy_student_replay),
        "projection_epochs": int(args.projection_epochs),
        "projection_max_samples": int(args.projection_max_samples),
        "eval_action_mode": args.eval_action_mode,
        "total_timesteps": int(args.total_timesteps),
        "gamma": float(args.gamma),
        "tau": float(args.tau),
        "batch_size": int(args.batch_size),
        "learning_starts": int(args.learning_starts),
        "random_actions_end": int(args.random_actions_end),
        "policy_lr": float(args.policy_lr),
        "alpha_lr": float(args.alpha_lr),
        "alpha_mass_lr": float(args.alpha_lr if args.alpha_mass_lr is None else args.alpha_mass_lr),
        "alpha_warmup_steps": int(args.alpha_warmup_steps),
        "alpha_entropy_reg": float(args.alpha_entropy_reg),
        "distill_encoder_lr_mult": float(args.distill_encoder_lr_mult),
        "q_lr": float(args.q_lr),
        "alpha": float(args.alpha),
        "autotune": bool(args.autotune),
        "autotune_init_from_alpha": bool(args.autotune_init_from_alpha),
        "pool_size": int(args.pool_size),
        "eval_every": int(args.eval_every),
        "num_evals": int(args.num_evals),
        "encoder_from_base": bool(args.encoder_from_base),
        "freeze_root_encoder": bool(args.freeze_root_encoder),
        "distillation": bool(cfg["distillation"]),
        "use_alpha_mass": bool(cfg["use_alpha_mass"]),
        "use_alpha_scale": bool(cfg["use_alpha_scale"]),
        "fix_alpha_scale": bool(cfg["fix_alpha_scale"]),
        "alpha_mass_reg": float(args.alpha_mass_reg),
        "drift_reg": float(args.drift_reg),
        "constrain_alpha_mass": bool(args.constrain_alpha_mass),
        "train_shared": bool(args.train_shared),
        "encoder_linear_out": bool(args.encoder_linear_out),
        "distill_observation_skip": bool(args.distill_observation_skip),
        "distill_extra_steps": int(args.distill_extra_steps),
        "collect_cosine_buffers": bool(args.collect_cosine_buffers),
        "max_distill_buffer": int(args.max_distill_buffer),
        "similarity_samples": int(args.similarity_samples),
        "balance_source_lineages": bool(args.balance_source_lineages),
        "distill_max_samples": int(args.distill_max_samples),
        "distill_epochs": int(args.distill_epochs),
        "distill_lr": float(args.distill_lr),
        "distill_batch_size": int(args.distill_batch_size),
        "distill_test_frac": float(args.distill_test_frac),
        "distill_select_best_val": bool(args.distill_select_best_val),
    }

def train_chain(args, suite, condition, cfg, seed):
    cfg = _effective_condition_config(args, cfg)
    previous = []
    for seq_idx, task_id in enumerate(args.task_sequence):
        if task_id < 0 or task_id >= len(TASK_SUITES[suite]):
            raise ValueError(f"task_id {task_id} is invalid for {suite}")

        save_parent = metrics.checkpoint_dir(args.save_root, suite, condition, seed, seq_idx, task_id).parent
        run_dir = metrics.checkpoint_dir(args.save_root, suite, condition, seed, seq_idx, task_id)
        tb_dir = metrics.event_dir(args.runs_root, suite, condition, seed, seq_idx, task_id)
        analysis_dir = metrics.analysis_snapshot_path(
            args.analysis_root, suite, condition, seed, seq_idx, task_id
        ).parent
        prev_args = []
        if previous:
            prev_args = [previous[0]] if len(previous) == 1 else [previous[0], previous[-1]]
        expected_config = _expected_training_config(args, suite, task_id, seq_idx, seed, cfg)

        if args.force_retrain:
            for path in (run_dir, tb_dir, analysis_dir):
                if path.exists():
                    shutil.rmtree(path)

        if metrics.checkpoint_complete(run_dir):
            matches, reason = metrics.checkpoint_matches(
                run_dir, expected_config, parent_dirs=prev_args,
                pretrained_encoder=args.pretrained_encoder,
            )
            if matches:
                print(f"[{suite}/{condition}/seed={seed}] seq{seq_idx} already complete: {run_dir}")
                previous.append(run_dir)
                continue
            print(f"[{suite}/{condition}/seed={seed}] seq{seq_idx} stale checkpoint: {reason}; retraining")

        if args.skip_training:
            raise FileNotFoundError(
                f"Missing/stale checkpoint while --skip-training was set: {run_dir}"
            )

        # Remove partial outputs before a retry, otherwise TensorBoard can mix
        # stale and fresh event files from two different attempts.
        for path in (run_dir, tb_dir, analysis_dir):
            if path.exists():
                shutil.rmtree(path)

        tag = f"{suite}/{condition}/seed_{seed}/seq_{seq_idx}"
        cmd = [
            sys.executable, "run_sac.py",
            "--model-type=cka-rl",
            f"--task-suite={suite}",
            f"--task-id={task_id}",
            f"--seq-idx={seq_idx}",
            f"--seed={seed}",
            f"--tag={tag}",
            f"--save-dir={save_parent}",
            f"--runs-root={args.runs_root}",
            f"--analysis-root={args.analysis_root}",
            f"--total-timesteps={args.total_timesteps}",
            f"--learning-starts={args.learning_starts}",
            f"--random-actions-end={args.random_actions_end}",
            f"--batch-size={args.batch_size}",
            f"--policy-lr={args.policy_lr}",
            f"--alpha-lr={args.alpha_lr}",
            f"--alpha-mass-lr={args.alpha_lr if args.alpha_mass_lr is None else args.alpha_mass_lr}",
            f"--alpha-mass-reg={args.alpha_mass_reg}",
            f"--alpha-warmup-steps={args.alpha_warmup_steps}",
            f"--alpha-entropy-reg={args.alpha_entropy_reg}",
            f"--drift-reg={args.drift_reg}",
            f"--distill-encoder-lr-mult={args.distill_encoder_lr_mult}",
            f"--q-lr={args.q_lr}",
            f"--gamma={args.gamma}",
            f"--tau={args.tau}",
            f"--pool-size={args.pool_size}",
            f"--eval-every={args.eval_every}",
            f"--num-evals={args.num_evals}",
            "--distill-observation-skip" if args.distill_observation_skip else "--no-distill-observation-skip",
            f"--distill-extra-steps={args.distill_extra_steps}",
            f"--max-distill-buffer={args.max_distill_buffer}",
            f"--similarity-samples={args.similarity_samples}",
            "--balance-source-lineages" if args.balance_source_lineages else "--no-balance-source-lineages",
            f"--distill-max-samples={args.distill_max_samples}",
            f"--distill-epochs={args.distill_epochs}",
            f"--distill-lr={args.distill_lr}",
            f"--distill-batch-size={args.distill_batch_size}",
            f"--distill-test-frac={args.distill_test_frac}",
            f"--analysis-log-every={args.analysis_log_every}",
            f"--fusion-mode={cfg['fusion_mode']}",
            f"--composition-space={cfg.get('composition_space', 'parameter')}",
            "--policy-student-replay" if args.policy_student_replay else "--no-policy-student-replay",
            f"--projection-epochs={args.projection_epochs}",
            f"--projection-max-samples={args.projection_max_samples}",
            f"--eval-action-mode={args.eval_action_mode}",
            f"--alpha={args.alpha}",
            "--autotune" if args.autotune else "--no-autotune",
            "--autotune-init-from-alpha" if args.autotune_init_from_alpha else "--no-autotune-init-from-alpha",
            "--use-alpha-scale" if cfg["use_alpha_scale"] else "--no-use-alpha-scale",
            "--fix-alpha-scale" if cfg["fix_alpha_scale"] else "--no-fix-alpha-scale",
            "--distillation" if cfg["distillation"] else "--no-distillation",
            "--distill-select-best-val" if args.distill_select_best_val else "--no-distill-select-best-val",
            "--collect-cosine-buffers" if args.collect_cosine_buffers else "--no-collect-cosine-buffers",
            "--train-shared" if args.train_shared else "--no-train-shared",
            "--freeze-root-encoder" if args.freeze_root_encoder else "--no-freeze-root-encoder",
            "--encoder-from-base" if args.encoder_from_base else "--no-encoder-from-base",
            "--encoder-linear-out" if args.encoder_linear_out else "--no-encoder-linear-out",
            "--use-alpha-mass" if cfg["use_alpha_mass"] else "--no-use-alpha-mass",
            "--constrain-alpha-mass" if args.constrain_alpha_mass else "--no-constrain-alpha-mass",
        ]
        if args.pretrained_encoder:
            cmd.append(f"--pretrained-encoder={args.pretrained_encoder}")
        if args.cpu:
            cmd.append("--no-cuda")

        if prev_args:
            # run_sac only needs the immutable root and latest continual state.
            cmd.append("--prev-units")
            cmd.extend(str(path) for path in prev_args)

        print(
            f"\n>>> {suite} | {condition} | seed {seed} | seq{seq_idx} "
            f"task {task_id}: {get_task_name(task_id, suite)} <<<"
        )
        subprocess.run(cmd, check=True)
        if not metrics.checkpoint_complete(run_dir):
            raise RuntimeError(f"Training command finished but checkpoint is incomplete: {run_dir}")
        matches, reason = metrics.checkpoint_matches(
            run_dir, expected_config, parent_dirs=prev_args,
            pretrained_encoder=args.pretrained_encoder,
        )
        if not matches:
            raise RuntimeError(f"Training produced a checkpoint with unexpected identity: {reason}")
        previous.append(run_dir)
    return previous


# ==========================================================================
# MAIN
# ==========================================================================
def main():
    args = parse_args()
    pathlib.Path(args.plots_root).mkdir(parents=True, exist_ok=True)
    import json
    with open(pathlib.Path(args.plots_root) / "benchmark_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Evaluation device: {device}")
    print(f"Sequence: {args.task_sequence}")

    all_condition_names = list(CONDITIONS.keys())
    if 0 in args.condition_index:
        conditions = all_condition_names
    else:
        # Preserve CLI order while removing accidental duplicates.
        conditions = []
        for idx in args.condition_index:
            name = all_condition_names[idx - 1]
            if name not in conditions:
                conditions.append(name)
    # Cross selected methods with composition space without changing the legacy
    # ablation indices. The replay-trained standalone-expert variant gets its own
    # label so it never reuses/overwrites ordinary combined_policy checkpoints.
    selected_conditions = {}
    for name in conditions:
        for space in dict.fromkeys(args.composition_spaces):
            if space == "parameter":
                label = name
            elif args.policy_student_replay:
                label = name + "_policy_student"
            else:
                label = name + "_policy"
            selected_conditions[label] = {**CONDITIONS[name], "composition_space": space}
    conditions = list(selected_conditions)
    print(f"Conditions: {conditions}")

    requested_sequence = args.task_sequence
    requested_seeds = list(args.seeds)
    for suite in args.task_suites:
        from tasks import DEFAULT_CONTINUAL_SEQUENCE
        import tasks as task_definitions
        suite_default = (task_definitions.default_sequence(suite) if hasattr(task_definitions, "default_sequence")
                         else DEFAULT_CONTINUAL_SEQUENCE)
        args.task_sequence = list(requested_sequence if requested_sequence is not None else suite_default)
        args.seeds = list(requested_seeds)
        print(f"\n================ {suite} ================")

        # During evaluation-only aggregation, optionally validate each complete
        # continual seed independently.  A failed/partial seed is excluded from
        # the aggregate instead of aborting the good seeds.  Normal training and
        # strict --skip-training behavior are unchanged unless the explicit
        # --skip-invalid-seeds flag is present.
        if args.skip_training and args.skip_invalid_seeds:
            valid_seeds = []
            invalid_seed_reasons = {}
            for seed in requested_seeds:
                try:
                    for condition, cfg in selected_conditions.items():
                        train_chain(args, suite, condition, cfg, seed)
                except Exception as exc:
                    invalid_seed_reasons[seed] = f"{type(exc).__name__}: {exc}"
                    print(
                        f"[skip-invalid-seeds] skipping {suite} seed {seed}: "
                        f"{invalid_seed_reasons[seed]}",
                        file=sys.stderr,
                    )
                    continue
                valid_seeds.append(seed)

            if not valid_seeds:
                details = "; ".join(
                    f"seed {seed}: {reason}" for seed, reason in invalid_seed_reasons.items()
                )
                raise RuntimeError(
                    f"No valid seeds remain for {suite} after checkpoint validation. {details}"
                )
            args.seeds = valid_seeds
            print(f"[skip-invalid-seeds] checkpoint-valid seeds for {suite}: {args.seeds}")
        else:
            for condition, cfg in selected_conditions.items():
                for seed in args.seeds:
                    train_chain(args, suite, condition, cfg, seed)

        plots.plot_training_metrics(args, suite, conditions)
        plots.plot_sequence_diagnostics(args, suite, conditions)
        plots.plot_merge_lineage(args, suite, conditions)
        plots.plot_zero_shot(args, suite, conditions)

        if args.skip_training and args.skip_invalid_seeds:
            # Evaluate each seed transactionally: a seed contributes to the
            # aggregate only if every requested metric can be computed for every
            # selected condition.  This also catches unreadable/corrupt .pt files
            # that may pass the lightweight manifest/file-existence checks above.
            all_payloads = {condition: [] for condition in conditions}
            survey_payloads = {condition: [] for condition in conditions}
            metric_valid_seeds = []
            for seed in list(args.seeds):
                seed_retention = {}
                seed_survey = {}
                try:
                    for condition in conditions:
                        if not args.skip_retention:
                            seed_retention[condition] = metrics.build_retention_matrix(
                                args, suite, condition, seed, device
                            )
                        if not args.skip_survey_metrics:
                            seed_survey[condition] = metrics.compute_survey_metrics(
                                args, suite, condition, seed, device,
                                args.scratch_seeds, args.total_timesteps
                            )
                except Exception as exc:
                    print(
                        f"[skip-invalid-seeds] skipping {suite} seed {seed} during metric "
                        f"evaluation: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    continue

                metric_valid_seeds.append(seed)
                for condition, payload in seed_retention.items():
                    all_payloads[condition].append(payload)
                for condition, payload in seed_survey.items():
                    survey_payloads[condition].append(payload)

            if not metric_valid_seeds:
                raise RuntimeError(
                    f"No valid seeds remain for {suite} after metric evaluation."
                )
            args.seeds = metric_valid_seeds
            print(f"[skip-invalid-seeds] aggregate seeds for {suite}: {args.seeds}")

            if not args.skip_retention:
                plots.plot_retention(args, suite, conditions, all_payloads)
                plots.write_summary_csv(args, suite, conditions, all_payloads)
            if not args.skip_survey_metrics:
                plots.plot_survey_metrics(args, suite, conditions, survey_payloads)
                plots.write_survey_metrics_csv(args, suite, conditions, survey_payloads)
        else:
            if not args.skip_retention:
                all_payloads = {condition: [] for condition in conditions}
                for condition in conditions:
                    for seed in args.seeds:
                        all_payloads[condition].append(
                            metrics.build_retention_matrix(args, suite, condition, seed, device)
                        )
                plots.plot_retention(args, suite, conditions, all_payloads)
                plots.write_summary_csv(args, suite, conditions, all_payloads)

            if not args.skip_survey_metrics:
                survey_payloads = {condition: [] for condition in conditions}
                for condition in conditions:
                    for seed in args.seeds:
                        survey_payloads[condition].append(metrics.compute_survey_metrics(
                            args, suite, condition, seed, device,
                            args.scratch_seeds, args.total_timesteps))
                plots.plot_survey_metrics(args, suite, conditions, survey_payloads)
                plots.write_survey_metrics_csv(args, suite, conditions, survey_payloads)

    print(f"\nDone. Plots and cached metrics: {args.plots_root}")


if __name__ == "__main__":
    main()