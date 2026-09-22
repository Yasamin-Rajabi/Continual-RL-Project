"""Continual Atari benchmark for Freeway and Space Invaders modes.

The four legacy experimental conditions preserve the same two method axes used
by the HalfCheetah benchmark, and can now be crossed with parameter-space or
exact categorical policy-space composition:

    baseline     = classic CKA vectors + arithmetic merge
    distil_only  = classic CKA vectors + categorical-KL distillation merge
    weight_only  = weight-delta vectors + alpha-mass + arithmetic merge
    combined     = weight-delta vectors + alpha-mass + categorical-KL distillation merge

For the distillation conditions, merge-pair selection uses symmetric
categorical KL on balanced stored states. For non-distillation conditions,
pair selection remains parameter-space cosine similarity.

This file is orchestration only.  It launches run_ppo_continual.py once per
sequence position, validates resumable checkpoints, calls metrics.py for
retention/survey metrics, and calls plots.py for visualization.

The Atari PPO trainer now exposes the same task-boundary analysis lifecycle as
the HalfCheetah trainer (start/pre_finalize/post_finalize snapshots), so this
orchestrator forwards and cleans the matching analysis directory as well.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
from collections import OrderedDict

import torch

from atari_tasks import TASK_SUITES, get_continual_sequence, get_task_name
import metrics
import plots
import scratch_baselines


CONDITIONS = OrderedDict([
    (
        "baseline",
        {
            "fusion_mode": "classic_cka",
            "distillation": False,
            "use_alpha_mass": False,
            "use_alpha_scale": True,
            "fix_alpha_scale": False,
        },
    ),
    (
        "distil_only",
        {
            "fusion_mode": "classic_cka",
            "distillation": True,
            "use_alpha_mass": False,
            "use_alpha_scale": True,
            "fix_alpha_scale": False,
        },
    ),
    (
        "weight_only",
        {
            "fusion_mode": "weight_delta",
            "distillation": False,
            "use_alpha_mass": True,
            "use_alpha_scale": False,
            "fix_alpha_scale": True,
        },
    ),
    (
        "combined",
        {
            "fusion_mode": "weight_delta",
            "distillation": True,
            "use_alpha_mass": True,
            "use_alpha_scale": False,
            "fix_alpha_scale": True,
        },
    ),
])


def _load_thresholds(spec):
    if not spec:
        return {}
    path = pathlib.Path(spec)
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return json.loads(spec)


def _threshold(mapping, suite, task_id):
    row = (mapping or {}).get(suite, {})
    value = row.get(str(task_id), row.get(task_id))
    return None if value is None else float(value)


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # ------------------------------------------------------------------
    # Continual benchmark setup
    # ------------------------------------------------------------------
    p.add_argument(
        "--task-suites",
        nargs="+",
        default=["freeway", "space_invaders"],
        choices=sorted(TASK_SUITES.keys()),
    )
    p.add_argument("--seeds", nargs="+", type=int, default=[101, 102, 103])
    p.add_argument(
        "--task-sequence",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional sequence override used for every suite.  If omitted, "
            "each suite uses its own full mode sequence."
        ),
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeat each suite's default mode stream this many times.",
    )

    # ------------------------------------------------------------------
    # PPO settings -- every training-relevant run_ppo_continual.py knob
    # that belongs at benchmark level is represented here and forwarded.
    # ------------------------------------------------------------------
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--learning-rate", type=float, default=2.5e-4)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--anneal-lr", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--norm-adv", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--clip-vloss", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)

    p.add_argument("--target-kl", type=float, default=None)
    p.add_argument("--eval-every", type=int, default=50_000)
    p.add_argument("--num-evals", type=int, default=5)
    p.add_argument(
        "--torch-deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forwarded to run_ppo_continual.py for reproducibility.",
    )

    # ------------------------------------------------------------------
    # Knowledge pool / method settings
    # ------------------------------------------------------------------
    p.add_argument("--pool-size", type=int, default=5)
    p.add_argument("--alpha-init", choices=["Randn", "Major", "Uniform"], default="Randn")
    p.add_argument("--alpha-major", type=float, default=0.6)
    p.add_argument("--alpha-factor", type=float, default=1e-3)
    p.add_argument("--fix-alpha", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--alpha-learning-rate", type=float, default=2.5e-4)
    p.add_argument(
        "--alpha-mass-learning-rate", type=float, default=None,
        help="Learning rate for the raw alpha-mass gate; default reuses --alpha-learning-rate.",
    )
    p.add_argument("--alpha-warmup-steps", type=int, default=60_000)
    p.add_argument("--alpha-entropy-reg", type=float, default=0.0001)
    p.add_argument("--alpha-mass-reg", type=float, default=0.005)
    p.add_argument(
        "--constrain-alpha-mass",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--weight-use-alpha-mass",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--condition-alpha-scale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use condition-specific alpha-scale policy: learned for classic CKA, "
            "fixed at 5 for weight-delta."
        ),
    )
    p.add_argument(
        "--use-alpha-scale", action=argparse.BooleanOptionalAction, default=False
    )
    p.add_argument(
        "--fix-alpha-scale", action=argparse.BooleanOptionalAction, default=False
    )

    # Encoder policy.  Atari intentionally does NOT concatenate raw pixels to
    # encoder features before the policy head.
    p.add_argument(
        "--encoder-from-base", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument(
        "--train-shared", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument(
        "--freeze-root-encoder",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--pretrained-encoder", default=None)
    p.add_argument("--shared-dim", type=int, default=512)
    p.add_argument("--head-hidden-dim", type=int, default=512)
    p.add_argument("--distill-encoder-lr-mult", type=float, default=0.1)
    p.add_argument("--drift-reg", type=float, default=1.0)

    # Behavioral buffers / categorical distillation.
    p.add_argument(
        "--distill-extra-steps",
        type=int,
        default=20_000,
        help=(
            "Frozen final B Atari transitions INSIDE total-timesteps. "
            "PPO optimization receives Delta-B transitions."
        ),
    )
    p.add_argument(
        "--composition-spaces",
        nargs="+",
        choices=["parameter", "policy"],
        default=["parameter", "policy"],
        help="Use parameter alone to disable exact categorical policy-space runs.",
    )
    p.add_argument(
        "--policy-student-replay",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Combined policy-space variant: execution uses the exact mixture "
            "while a standalone novel categorical expert is trained for storage."
        ),
    )
    p.add_argument("--projection-epochs", type=int, default=16)
    p.add_argument("--projection-max-samples", type=int, default=20_000)
    p.add_argument(
        "--collect-cosine-buffers",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--max-distill-buffer", type=int, default=30_000)
    p.add_argument("--similarity-samples", type=int, default=512)
    p.add_argument(
        "--balance-source-lineages", action=argparse.BooleanOptionalAction, default=True,
        help="Balance behavioral-KL/distillation/merge-buffer sampling across original source_ids.",
    )
    p.add_argument("--distill-max-samples", type=int, default=10_000)
    p.add_argument("--distill-epochs", type=int, default=8)
    p.add_argument("--distill-lr", type=float, default=3e-4)
    p.add_argument("--distill-batch-size", type=int, default=256)
    p.add_argument("--distill-test-frac", type=float, default=0.2)
    p.add_argument(
        "--distill-select-best-val",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    # ------------------------------------------------------------------
    # Evaluation / survey metrics
    # ------------------------------------------------------------------
    p.add_argument("--retention-eval-episodes", type=int, default=10)
    p.add_argument("--test-adapt-steps", type=int, default=50_000)
    p.add_argument("--test-adapt-lr", type=float, default=1e-2)
    p.add_argument(
        "--frozen-eval-policy", choices=["pool", "snapshot"], default="pool"
    )
    p.add_argument(
        "--eval-action-mode", choices=["deterministic", "stochastic"],
        default="deterministic",
    )
    p.add_argument(
        "--skip-forward-transfer", action="store_true",
        help="Compute A_N/FG/BWT without scratch denominators; FT remains NaN.",
    )
    p.add_argument(
        "--success-thresholds-json",
        default=None,
        help=(
            "Path or JSON dictionary of fixed raw-score thresholds, e.g. "
            "{\"freeway\":{\"0\":10.0}}.  Without thresholds, reward metrics "
            "remain available and success metrics are NaN."
        ),
    )

    # Output / orchestration.
    p.add_argument("--save-root", default="agents_atari_continual")
    p.add_argument("--runs-root", default="runs_atari")
    p.add_argument("--plots-root", default="plots_atari_continual")
    p.add_argument("--analysis-root", default="analysis_runs_atari")
    p.add_argument("--analysis-log-every", type=int, default=40_000)
    p.add_argument(
        "--save-analysis-snapshots",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--scratch-save-root", default=scratch_baselines.SCRATCH_SAVE_ROOT
    )
    p.add_argument(
        "--scratch-seeds",
        nargs="+",
        type=int,
        default=scratch_baselines.DEFAULT_SCRATCH_SEEDS,
    )
    p.add_argument("--skip-training", action="store_true")
    p.add_argument("--skip-retention", action="store_true")
    p.add_argument("--skip-survey-metrics", action="store_true")
    p.add_argument("--force-retrain", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument(
        "--condition-index",
        nargs="+",
        type=int,
        default=[1, 4],
        choices=[0, 1, 2, 3, 4],
        help=(
            "0 = all four conditions; otherwise choose one or more of "
            "1=baseline, 2=distil_only, 3=weight_only, 4=combined."
        ),
    )
    p.add_argument("--quick-test", action="store_true")

    args = p.parse_args()
    args.success_thresholds = _load_thresholds(args.success_thresholds_json)
    args.task_sequence_override = (
        None if args.task_sequence is None else list(args.task_sequence)
    )

    if args.quick_test:
        args.task_suites = ["freeway"]
        args.seeds = [1]
        args.task_sequence_override = [0, 1, 2, 3]
        args.total_timesteps = 20_000
        args.num_envs = 4
        args.num_steps = 64
        args.pool_size = 2
        args.eval_every = 5_000
        args.num_evals = 1
        args.retention_eval_episodes = 1
        args.distill_extra_steps = 256
        args.max_distill_buffer = 512
        args.similarity_samples = 64
        args.distill_max_samples = 128
        args.distill_epochs = 2

    # Fail fast on contradictory / invalid benchmark-level settings.
    if args.repeats < 1:
        p.error("--repeats must be >= 1")
    if args.train_shared and args.freeze_root_encoder:
        p.error("--train-shared and --freeze-root-encoder are contradictory")
    if args.use_alpha_scale and args.fix_alpha_scale:
        p.error("--use-alpha-scale and --fix-alpha-scale are mutually exclusive")
    if 0 in args.condition_index and len(args.condition_index) > 1:
        p.error("--condition-index 0 means all conditions and cannot be combined")
    if args.total_timesteps < 1 or args.num_envs < 1 or args.num_steps < 1:
        p.error("total_timesteps, num_envs, and num_steps must be >= 1")
    if args.num_minibatches < 1 or args.update_epochs < 1:
        p.error("num_minibatches and update_epochs must be >= 1")
    if args.num_envs * args.num_steps < args.num_minibatches:
        p.error("num_minibatches cannot exceed num_envs * num_steps")
    if args.pool_size < 2:
        p.error("--pool-size must be >= 2")
    if args.alpha_learning_rate <= 0 or args.learning_rate <= 0:
        p.error("learning rates must be > 0")
    if not 0.0 < args.gamma <= 1.0 or not 0.0 <= args.gae_lambda <= 1.0:
        p.error("gamma must be in (0,1] and gae_lambda in [0,1]")
    if args.clip_coef <= 0 or args.max_grad_norm <= 0:
        p.error("clip_coef and max_grad_norm must be > 0")
    if args.ent_coef < 0 or args.vf_coef < 0:
        p.error("ent_coef and vf_coef must be >= 0")
    if args.target_kl is not None and args.target_kl <= 0:
        p.error("target_kl must be > 0 when provided")
    if args.alpha_entropy_reg < 0 or args.alpha_mass_reg < 0 or args.drift_reg < 0:
        p.error("drift/alpha regularization coefficients must be >= 0")
    if args.distill_encoder_lr_mult <= 0:
        p.error("--distill-encoder-lr-mult must be > 0")
    if args.max_distill_buffer < 2 or args.distill_max_samples < 2:
        p.error("distillation buffers/sample budgets must be >= 2")
    if args.similarity_samples < 2:
        p.error("--similarity-samples must be >= 2")
    if args.distill_epochs < 1 or args.distill_batch_size < 1:
        p.error("distill_epochs and distill_batch_size must be >= 1")
    if not 0.0 <= args.distill_test_frac < 1.0:
        p.error("--distill-test-frac must be in [0,1)")
    if args.test_adapt_steps < 0 or args.test_adapt_lr <= 0:
        p.error("test_adapt_steps must be >=0 and test_adapt_lr must be >0")
    if args.analysis_log_every < 0:
        p.error("--analysis-log-every must be >= 0")
    if args.alpha_mass_learning_rate is not None and args.alpha_mass_learning_rate <= 0:
        p.error("--alpha-mass-learning-rate must be > 0 when provided")
    if not 0 <= args.distill_extra_steps < args.total_timesteps:
        p.error("Require 0 <= B < Delta: distill-extra-steps is inside total-timesteps")
    if args.projection_epochs < 1 or args.projection_max_samples < 2:
        p.error("projection-epochs >= 1 and projection-max-samples >= 2 are required")
    if args.test_adapt_steps and args.frozen_eval_policy != "pool":
        p.error("Test-time adaptation requires --frozen-eval-policy pool")
    if args.policy_student_replay:
        if list(dict.fromkeys(args.composition_spaces)) != ["policy"]:
            p.error("--policy-student-replay requires --composition-spaces policy only")
        if args.condition_index != [4]:
            p.error("--policy-student-replay is defined for --condition-index 4 (combined) only")
        if not args.weight_use_alpha_mass:
            p.error("--policy-student-replay requires alpha-mass")
    batch_size = args.num_envs * args.num_steps
    if batch_size % args.num_minibatches != 0:
        p.error("num_envs * num_steps must be divisible by num_minibatches")
    if args.total_timesteps < batch_size:
        p.error("total_timesteps must be at least one PPO rollout")

    return args


def _effective_condition_config(args, cfg):
    cfg = dict(cfg)
    if cfg["fusion_mode"] == "weight_delta":
        cfg["use_alpha_mass"] = bool(
            cfg["use_alpha_mass"] and args.weight_use_alpha_mass
        )
    if not args.condition_alpha_scale:
        cfg["use_alpha_scale"] = bool(args.use_alpha_scale)
        cfg["fix_alpha_scale"] = bool(args.fix_alpha_scale)
    return cfg


def _expected_training_config(args, suite, task_id, seq_idx, seed, cfg):
    """Training signature expected from run_ppo_continual.py.

    Keep this synchronized with experiment_identity.TRAINING_KEYS and the Args
    dataclass in run_ppo_continual.py.  Extra keys are harmless if an older
    identity helper ignores them, but the identity helper should ultimately be
    updated to include every training-relevant key as well.
    """
    return {
        "model_type": "cka-rl",
        "task_suite": suite,
        "task_id": int(task_id),
        "seq_idx": int(seq_idx),
        "seed": int(seed),
        "torch_deterministic": bool(args.torch_deterministic),
        "cuda": not bool(args.cpu),
        "total_timesteps": int(args.total_timesteps),
        "learning_rate": float(args.learning_rate),
        "num_envs": int(args.num_envs),
        "num_steps": int(args.num_steps),
        "anneal_lr": bool(args.anneal_lr),
        "gamma": float(args.gamma),
        "gae_lambda": float(args.gae_lambda),
        "num_minibatches": int(args.num_minibatches),
        "update_epochs": int(args.update_epochs),
        "norm_adv": bool(args.norm_adv),
        "clip_coef": float(args.clip_coef),
        "clip_vloss": bool(args.clip_vloss),
        "ent_coef": float(args.ent_coef),
        "vf_coef": float(args.vf_coef),
        "max_grad_norm": float(args.max_grad_norm),
        "target_kl": None if args.target_kl is None else float(args.target_kl),
        "eval_every": int(args.eval_every),
        "num_evals": int(args.num_evals),
        "fusion_mode": cfg["fusion_mode"],
        "composition_space": cfg.get("composition_space", "parameter"),
        "policy_student_replay": bool(args.policy_student_replay),
        "projection_epochs": int(args.projection_epochs),
        "projection_max_samples": int(args.projection_max_samples),
        "eval_action_mode": args.eval_action_mode,
        "pool_size": int(args.pool_size),
        "alpha_init": args.alpha_init,
        "alpha_major": float(args.alpha_major),
        "alpha_factor": float(args.alpha_factor),
        "fix_alpha": bool(args.fix_alpha),
        "alpha_learning_rate": float(args.alpha_learning_rate),
        "alpha_mass_learning_rate": float(
            args.alpha_learning_rate
            if args.alpha_mass_learning_rate is None
            else args.alpha_mass_learning_rate
        ),
        "alpha_warmup_steps": int(args.alpha_warmup_steps),
        "alpha_entropy_reg": float(args.alpha_entropy_reg),
        "alpha_mass_reg": float(args.alpha_mass_reg),
        "use_alpha_scale": bool(cfg["use_alpha_scale"]),
        "fix_alpha_scale": bool(cfg["fix_alpha_scale"]),
        "use_alpha_mass": bool(cfg["use_alpha_mass"]),
        "constrain_alpha_mass": bool(args.constrain_alpha_mass),
        "encoder_from_base": bool(args.encoder_from_base),
        "train_shared": bool(args.train_shared),
        "freeze_root_encoder": bool(args.freeze_root_encoder),
        "shared_dim": int(args.shared_dim),
        "head_hidden_dim": int(args.head_hidden_dim),
        "distill_encoder_lr_mult": float(args.distill_encoder_lr_mult),
        "drift_reg": float(args.drift_reg),
        "distillation": bool(cfg["distillation"]),
        "collect_cosine_buffers": bool(args.collect_cosine_buffers),
        "distill_extra_steps": int(args.distill_extra_steps),
        "max_distill_buffer": int(args.max_distill_buffer),
        "similarity_samples": int(args.similarity_samples),
        "balance_source_lineages": bool(args.balance_source_lineages),
        "distill_max_samples": int(args.distill_max_samples),
        "distill_epochs": int(args.distill_epochs),
        "distill_lr": float(args.distill_lr),
        "distill_batch_size": int(args.distill_batch_size),
        "distill_test_frac": float(args.distill_test_frac),
        "distill_select_best_val": bool(args.distill_select_best_val),
        "success_threshold": _threshold(args.success_thresholds, suite, task_id),
    }


def train_chain(args, suite, condition, raw_cfg, seed):
    """Train/resume one continual chain for one suite/condition/seed."""
    cfg = _effective_condition_config(args, raw_cfg)
    previous = []

    for seq_idx, task_id in enumerate(args.task_sequence):
        if task_id < 0 or task_id >= len(TASK_SUITES[suite]):
            raise ValueError(f"task_id {task_id} is invalid for {suite}")

        run_dir = metrics.checkpoint_dir(
            args.save_root, suite, condition, seed, seq_idx, task_id
        )
        tb_dir = metrics.event_dir(
            args.runs_root, suite, condition, seed, seq_idx, task_id
        )
        prev_args = []
        if previous:
            prev_args = (
                [previous[0]]
                if len(previous) == 1
                else [previous[0], previous[-1]]
            )
        expected_config = _expected_training_config(
            args, suite, task_id, seq_idx, seed, cfg
        )

        tag = f"{suite}/{condition}/seed_{seed}/seq_{seq_idx}"
        analysis_dir = (
            pathlib.Path(args.analysis_root) / tag / metrics.run_name(suite, task_id, seed)
        )
        if args.force_retrain:
            for path in (run_dir, tb_dir, analysis_dir):
                if path.exists():
                    shutil.rmtree(path)

        if metrics.checkpoint_complete(run_dir):
            matches, reason = metrics.checkpoint_matches(
                run_dir,
                expected_config,
                parent_dirs=prev_args,
                pretrained_encoder=args.pretrained_encoder,
            )
            if matches:
                print(
                    f"[{suite}/{condition}/seed={seed}] seq{seq_idx} already "
                    f"complete: {run_dir}"
                )
                previous.append(run_dir)
                continue
            print(
                f"[{suite}/{condition}/seed={seed}] seq{seq_idx} stale "
                f"checkpoint: {reason}; retraining"
            )

        if args.skip_training:
            raise FileNotFoundError(
                f"Missing/stale checkpoint while --skip-training was set: {run_dir}"
            )

        # Clear partial/stale outputs before retry so EventAccumulator never
        # merges old and new PPO learning curves.
        for path in (run_dir, tb_dir, analysis_dir):
            if path.exists():
                shutil.rmtree(path)

        cmd = [
            sys.executable,
            "run_ppo_continual.py",
            "--model-type=cka-rl",
            f"--task-suite={suite}",
            f"--task-id={task_id}",
            f"--seq-idx={seq_idx}",
            f"--seed={seed}",
            f"--save-dir={run_dir}",
            f"--runs-root={args.runs_root}",
            f"--tag={tag}",
            f"--total-timesteps={args.total_timesteps}",
            f"--learning-rate={args.learning_rate}",
            f"--num-envs={args.num_envs}",
            f"--num-steps={args.num_steps}",
            f"--gamma={args.gamma}",
            f"--gae-lambda={args.gae_lambda}",
            f"--num-minibatches={args.num_minibatches}",
            f"--update-epochs={args.update_epochs}",
            f"--clip-coef={args.clip_coef}",
            f"--ent-coef={args.ent_coef}",
            f"--vf-coef={args.vf_coef}",
            f"--max-grad-norm={args.max_grad_norm}",
            f"--eval-every={args.eval_every}",
            f"--num-evals={args.num_evals}",
            f"--fusion-mode={cfg['fusion_mode']}",
            f"--composition-space={cfg.get('composition_space', 'parameter')}",
            "--policy-student-replay" if args.policy_student_replay else "--no-policy-student-replay",
            f"--projection-epochs={args.projection_epochs}",
            f"--projection-max-samples={args.projection_max_samples}",
            f"--eval-action-mode={args.eval_action_mode}",
            f"--pool-size={args.pool_size}",
            f"--alpha-init={args.alpha_init}",
            f"--alpha-major={args.alpha_major}",
            f"--alpha-factor={args.alpha_factor}",
            f"--alpha-learning-rate={args.alpha_learning_rate}",
            f"--alpha-mass-learning-rate={args.alpha_learning_rate if args.alpha_mass_learning_rate is None else args.alpha_mass_learning_rate}",
            f"--alpha-warmup-steps={args.alpha_warmup_steps}",
            f"--alpha-entropy-reg={args.alpha_entropy_reg}",
            f"--alpha-mass-reg={args.alpha_mass_reg}",
            f"--shared-dim={args.shared_dim}",
            f"--head-hidden-dim={args.head_hidden_dim}",
            f"--distill-encoder-lr-mult={args.distill_encoder_lr_mult}",
            f"--drift-reg={args.drift_reg}",
            f"--analysis-root={args.analysis_root}",
            f"--analysis-log-every={args.analysis_log_every}",
            f"--distill-extra-steps={args.distill_extra_steps}",
            f"--max-distill-buffer={args.max_distill_buffer}",
            f"--similarity-samples={args.similarity_samples}",
            "--balance-source-lineages" if args.balance_source_lineages else "--no-balance-source-lineages",
            f"--distill-max-samples={args.distill_max_samples}",
            f"--distill-epochs={args.distill_epochs}",
            f"--distill-lr={args.distill_lr}",
            f"--distill-batch-size={args.distill_batch_size}",
            f"--distill-test-frac={args.distill_test_frac}",
            "--torch-deterministic"
            if args.torch_deterministic
            else "--no-torch-deterministic",
            "--anneal-lr" if args.anneal_lr else "--no-anneal-lr",
            "--norm-adv" if args.norm_adv else "--no-norm-adv",
            "--clip-vloss" if args.clip_vloss else "--no-clip-vloss",
            "--fix-alpha" if args.fix_alpha else "--no-fix-alpha",
            "--use-alpha-scale"
            if cfg["use_alpha_scale"]
            else "--no-use-alpha-scale",
            "--fix-alpha-scale"
            if cfg["fix_alpha_scale"]
            else "--no-fix-alpha-scale",
            "--use-alpha-mass"
            if cfg["use_alpha_mass"]
            else "--no-use-alpha-mass",
            "--constrain-alpha-mass"
            if args.constrain_alpha_mass
            else "--no-constrain-alpha-mass",
            "--encoder-from-base"
            if args.encoder_from_base
            else "--no-encoder-from-base",
            "--train-shared" if args.train_shared else "--no-train-shared",
            "--freeze-root-encoder"
            if args.freeze_root_encoder
            else "--no-freeze-root-encoder",
            "--distillation"
            if cfg["distillation"]
            else "--no-distillation",
            "--collect-cosine-buffers"
            if args.collect_cosine_buffers
            else "--no-collect-cosine-buffers",
            "--distill-select-best-val"
            if args.distill_select_best_val
            else "--no-distill-select-best-val",
            "--save-analysis-snapshots"
            if args.save_analysis_snapshots
            else "--no-save-analysis-snapshots",
            "--cuda" if not args.cpu else "--no-cuda",
        ]

        if args.target_kl is not None:
            cmd.append(f"--target-kl={args.target_kl}")
        if args.pretrained_encoder:
            cmd.append(f"--pretrained-encoder={args.pretrained_encoder}")

        threshold = _threshold(args.success_thresholds, suite, task_id)
        if threshold is not None:
            cmd.append(f"--success-threshold={threshold}")

        if prev_args:
            # Same lineage convention as HalfCheetah: immutable root + latest.
            cmd.append("--prev-units")
            cmd.extend(str(path) for path in prev_args)

        print(
            f"\n>>> {suite} | {condition} | seed {seed} | seq{seq_idx} "
            f"task {task_id}: {get_task_name(task_id, suite)} <<<"
        )
        subprocess.run(cmd, check=True)

        if not metrics.checkpoint_complete(run_dir):
            raise RuntimeError(
                f"Training command finished but checkpoint is incomplete: {run_dir}"
            )

        # This post-run identity validation existed in the HalfCheetah file and
        # was missing from the previous Atari orchestrator.  Keep it: a process
        # can finish successfully yet still write an unexpected configuration.
        matches, reason = metrics.checkpoint_matches(
            run_dir,
            expected_config,
            parent_dirs=prev_args,
            pretrained_encoder=args.pretrained_encoder,
        )
        if not matches:
            raise RuntimeError(
                f"Training produced a checkpoint with unexpected identity: {reason}"
            )

        previous.append(run_dir)

    return previous


def _call_optional_diagnostic_plots(args, suite, conditions):
    """Call the HalfCheetah-style diagnostics when Atari plots.py implements them.

    The first Atari port only implemented plot_training_metrics/plot_retention/
    survey plots.  Calling nonexistent functions would crash the benchmark, so
    keep this orchestration future-compatible and make the missing plot support
    explicit instead of silently pretending it exists.
    """
    for name in ("plot_sequence_diagnostics", "plot_merge_lineage", "plot_zero_shot"):
        fn = getattr(plots, name, None)
        if fn is None:
            print(
                f"[plot warning] plots.{name} is not implemented in the current "
                "Atari plots.py; skipping it for now."
            )
            continue
        fn(args, suite, conditions)



def main():
    args = parse_args()
    pathlib.Path(args.plots_root).mkdir(parents=True, exist_ok=True)

    # Resolve suite-specific streams once.  This also makes benchmark_config.json
    # reproducible: the previous Atari file wrote task_sequence=None before the
    # per-suite defaults were resolved.
    sequence_by_suite = {
        suite: (
            list(args.task_sequence_override)
            if args.task_sequence_override is not None
            else list(get_continual_sequence(suite, repeats=args.repeats))
        )
        for suite in args.task_suites
    }

    config_to_save = dict(vars(args))
    config_to_save["resolved_sequences"] = sequence_by_suite
    # task_sequence is suite-specific at execution time; avoid writing a stale
    # value from whichever suite happened to run last.
    config_to_save["task_sequence"] = args.task_sequence_override
    with (pathlib.Path(args.plots_root) / "benchmark_config.json").open("w") as f:
        json.dump(config_to_save, f, indent=2)

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )
    print(f"Evaluation device: {device}")

    all_condition_names = list(CONDITIONS.keys())
    if 0 in args.condition_index:
        conditions = all_condition_names
    else:
        conditions = []
        for idx in args.condition_index:
            name = all_condition_names[idx - 1]
            if name not in conditions:
                conditions.append(name)
    selected_conditions = {}
    for name in conditions:
        for space in dict.fromkeys(args.composition_spaces):
            if space == "parameter":
                label = name
            elif args.policy_student_replay:
                label = name + "_policy_student"
            else:
                label = name + "_policy"
            selected_conditions[label] = {
                **CONDITIONS[name],
                "composition_space": space,
            }
    conditions = list(selected_conditions)
    print(f"Conditions: {conditions}")

    for suite in args.task_suites:
        args.task_sequence = list(sequence_by_suite[suite])
        print(f"\n================ {suite} ================")
        print(f"Sequence: {args.task_sequence}")

        # --------------------------------------------------------------
        # Training / resume
        # --------------------------------------------------------------
        for condition, cfg in selected_conditions.items():
            for seed in args.seeds:
                train_chain(args, suite, condition, cfg, seed)

        # --------------------------------------------------------------
        # Training and sequence diagnostics.
        # --------------------------------------------------------------
        plots.plot_training_metrics(args, suite, conditions)
        plots.plot_sequence_diagnostics(args, suite, conditions)
        plots.plot_merge_lineage(args, suite, conditions)
        plots.plot_zero_shot(args, suite, conditions)

        # --------------------------------------------------------------
        # Full retention matrix + retention plots/tables.
        # --------------------------------------------------------------
        if not args.skip_retention:
            all_payloads = {condition: [] for condition in conditions}
            for condition in conditions:
                for seed in args.seeds:
                    all_payloads[condition].append(
                        metrics.build_retention_matrix(
                            args, suite, condition, seed, device
                        )
                    )
            plots.plot_retention(args, suite, conditions, all_payloads)
            plots.write_summary_csv(args, suite, conditions, all_payloads)

        # --------------------------------------------------------------
        # Survey metrics. A_N/FG/BWT do not depend on scratch runs. FT reads
        # scratch learning curves only; metrics.py leaves FT as NaN when they
        # are unavailable or --skip-forward-transfer is set.
        # --------------------------------------------------------------
        if not args.skip_survey_metrics:
            scratch_seeds = [] if args.skip_forward_transfer else [
                int(x) for x in args.scratch_seeds
            ]
            survey_payloads = {condition: [] for condition in conditions}
            for condition in conditions:
                for seed in args.seeds:
                    survey_payloads[condition].append(
                        metrics.compute_survey_metrics(
                            args, suite, condition, seed, device,
                            scratch_seeds, args.total_timesteps,
                        )
                    )

            out_path = (
                pathlib.Path(args.plots_root)
                / suite
                / "survey_metrics"
                / "all_conditions.json"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w") as f:
                json.dump(survey_payloads, f, indent=2)

            plots.plot_survey_metrics(args, suite, conditions, survey_payloads)
            plots.write_survey_metrics_csv(
                args, suite, conditions, survey_payloads
            )

    print(f"\nDone. Plots and cached metrics: {args.plots_root}")


if __name__ == "__main__":
    main()
