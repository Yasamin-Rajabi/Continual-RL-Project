"""From-scratch single-task PPO baselines for Atari forward-transfer denominators.

Forward transfer requires a single-task learning curve for every task under the
same PPO/encoder training setup and the same training-step budget as the
continual run.  Atari has only one scratch architecture here: raw observations
are NOT concatenated to the CNN feature vector, so the distillation and
non-distillation continual conditions share the same base CNN + categorical
policy-head architecture.

Important invariants
--------------------
* ``--total-timesteps`` MUST match the continual benchmark.
* PPO hyperparameters and encoder settings MUST match the continual benchmark.
* Scratch seeds should be disjoint from continual-run seeds.
* ``success_threshold`` affects FT_success because it defines the logged
  ``charts/test_success`` curve; therefore the same fixed threshold must be used
  for continual and scratch runs.
* Pool/merge/distillation settings are disabled for scratch runs because a
  scratch run is a single root task and has no historical pool to merge.  This
  does not change its actor/critic architecture in the Atari implementation.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys
from typing import Any, Mapping

from atari_tasks import TASK_SUITES, get_task_name
from experiment_identity import (
    MANIFEST_NAME,
    checkpoint_matches as identity_checkpoint_matches,
    load_manifest,
)


SCRATCH_SAVE_ROOT = "scratch_models_atari"
# A multi-seed denominator is substantially less noisy for FT than one scratch
# run.  Keep these disjoint from the continual defaults [1,2,3].
DEFAULT_SCRATCH_SEEDS = [101, 102, 103]


def scratch_run_name(suite: str, task_id: int, seed: int) -> str:
    return f"{suite}__task_{task_id}__cka-rl__run_ppo__{seed}"


def scratch_tag(suite: str, task_id: int, total_timesteps: int, seed: int) -> str:
    return f"scratch/{suite}/task_{task_id}/steps_{total_timesteps}/seed_{seed}"


def scratch_checkpoint_dir(
    save_root, suite: str, task_id: int, total_timesteps: int, seed: int
) -> pathlib.Path:
    return (
        pathlib.Path(save_root)
        / suite
        / f"task_{task_id}"
        / f"steps_{total_timesteps}"
        / f"seed_{seed}"
        / scratch_run_name(suite, task_id, seed)
    )


def scratch_event_dir(
    runs_root, suite: str, task_id: int, total_timesteps: int, seed: int
) -> pathlib.Path:
    return (
        pathlib.Path(runs_root)
        / scratch_tag(suite, task_id, total_timesteps, seed)
        / scratch_run_name(suite, task_id, seed)
    )


def checkpoint_complete(path) -> bool:
    path = pathlib.Path(path)
    required = ["policy_snapshot.pt", "fc.pt", "policy_pool.pt", MANIFEST_NAME]
    return (
        path.exists()
        and all((path / name).exists() for name in required)
        and load_manifest(path) is not None
    )


def _load_thresholds(spec):
    if not spec:
        return {}
    path = pathlib.Path(spec)
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return json.loads(spec)


def _threshold(mapping, suite: str, task_id: int):
    if not mapping:
        return None
    row = mapping.get(suite, {})
    value = row.get(str(task_id), row.get(task_id))
    return None if value is None else float(value)


def _arg(args, name: str, default=None):
    """Read a setting from either this script's args or benchmark args."""
    return getattr(args, name, default)


def _expected_training_config(
    suite: str,
    task_id: int,
    total_timesteps: int,
    seed: int,
    args,
) -> dict:
    """Exact run_ppo_continual.py settings used by a scratch run.

    We intentionally include PPO keys even if an older experiment_identity.py
    does not yet fingerprint all of them.  ``checkpoint_matches`` performs an
    additional direct manifest-args comparison below, so changing a PPO setting
    cannot silently reuse a stale FT denominator.
    """
    success_thresholds = _arg(args, "success_thresholds", {})
    return {
        # identity / environment
        "model_type": "cka-rl",
        "task_suite": suite,
        "task_id": int(task_id),
        "seq_idx": 0,
        "seed": int(seed),
        "torch_deterministic": bool(_arg(args, "torch_deterministic", True)),
        "cuda": not bool(_arg(args, "cpu", False)),
        "total_timesteps": int(total_timesteps),

        # PPO -- must match continual training
        "learning_rate": float(_arg(args, "learning_rate", 2.5e-4)),
        "num_envs": int(_arg(args, "num_envs", 8)),
        "num_steps": int(_arg(args, "num_steps", 128)),
        "anneal_lr": bool(_arg(args, "anneal_lr", True)),
        "gamma": float(_arg(args, "gamma", 0.99)),
        "gae_lambda": float(_arg(args, "gae_lambda", 0.95)),
        "num_minibatches": int(_arg(args, "num_minibatches", 4)),
        "update_epochs": int(_arg(args, "update_epochs", 4)),
        "norm_adv": bool(_arg(args, "norm_adv", True)),
        "clip_coef": float(_arg(args, "clip_coef", 0.1)),
        "clip_vloss": bool(_arg(args, "clip_vloss", True)),
        "ent_coef": float(_arg(args, "ent_coef", 0.01)),
        "vf_coef": float(_arg(args, "vf_coef", 0.5)),
        "max_grad_norm": float(_arg(args, "max_grad_norm", 0.5)),
        "target_kl": _arg(args, "target_kl", None),
        "eval_every": int(_arg(args, "eval_every", 50_000)),
        "num_evals": int(_arg(args, "num_evals", 5)),
        "success_threshold": _threshold(success_thresholds, suite, task_id),

        # Root CKA policy.  Alpha/pool settings do not affect a root with no
        # history, but we make them explicit for reproducibility.
        "fusion_mode": "classic_cka",
        "pool_size": int(_arg(args, "pool_size", 5)),
        "alpha_init": str(_arg(args, "alpha_init", "Randn")),
        "alpha_major": float(_arg(args, "alpha_major", 0.6)),
        "alpha_factor": float(_arg(args, "alpha_factor", 1e-3)),
        "fix_alpha": bool(_arg(args, "fix_alpha", False)),
        "alpha_learning_rate": float(_arg(args, "alpha_learning_rate", 2.5e-4)),
        "alpha_warmup_steps": int(_arg(args, "alpha_warmup_steps", 5_000)),
        "alpha_entropy_reg": float(_arg(args, "alpha_entropy_reg", 0.01)),
        "alpha_mass_reg": float(_arg(args, "alpha_mass_reg", 0.05)),
        "use_alpha_scale": False,
        "fix_alpha_scale": False,
        "use_alpha_mass": False,
        "constrain_alpha_mass": bool(_arg(args, "constrain_alpha_mass", True)),

        # Encoder -- these can materially change root training and MUST match.
        "encoder_from_base": bool(_arg(args, "encoder_from_base", True)),
        "train_shared": bool(_arg(args, "train_shared", False)),
        "freeze_root_encoder": bool(_arg(args, "freeze_root_encoder", False)),
        "pretrained_encoder": _arg(args, "pretrained_encoder", None),
        "shared_dim": int(_arg(args, "shared_dim", 512)),
        "head_hidden_dim": int(_arg(args, "head_hidden_dim", 128)),
        "distill_encoder_lr_mult": float(_arg(args, "distill_encoder_lr_mult", 0.1)),
        "drift_reg": float(_arg(args, "drift_reg", 1.0)),

        # No historical pair exists in a scratch root, so these are disabled.
        "distillation": False,
        "collect_cosine_buffers": False,
        "distill_extra_steps": 0,
        "max_distill_buffer": int(_arg(args, "max_distill_buffer", 5_000)),
        "similarity_samples": int(_arg(args, "similarity_samples", 512)),
        "distill_max_samples": int(_arg(args, "distill_max_samples", 2_000)),
        "distill_epochs": int(_arg(args, "distill_epochs", 8)),
        "distill_lr": float(_arg(args, "distill_lr", 3e-4)),
        "distill_batch_size": int(_arg(args, "distill_batch_size", 256)),
        "distill_test_frac": float(_arg(args, "distill_test_frac", 0.2)),
        "distill_select_best_val": bool(_arg(args, "distill_select_best_val", True)),
    }


def _same_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, pathlib.Path):
        expected = str(expected)
    if isinstance(actual, pathlib.Path):
        actual = str(actual)
    # Manifest JSON turns tuples into lists, but none of the expected scratch
    # keys here should require special sequence handling.
    return actual == expected


def _strict_manifest_args_match(path, expected: Mapping[str, Any]):
    """Guard PPO keys until experiment_identity.py is fully Atari-aware."""
    manifest = load_manifest(path)
    if manifest is None:
        return False, "missing/invalid run_manifest.json"
    saved_args = manifest.get("args", {})
    for key, value in expected.items():
        # pretrained_encoder is validated by content hash through
        # identity_checkpoint_matches; its path string may legitimately differ.
        if key == "pretrained_encoder":
            continue
        if key not in saved_args:
            return False, f"saved manifest args missing {key}"
        if not _same_value(saved_args[key], value):
            return (
                False,
                f"training config mismatch for {key}: "
                f"saved={saved_args[key]!r}, expected={value!r}",
            )
    return True, "match"


def checkpoint_matches(
    path,
    suite: str,
    task_id: int,
    total_timesteps: int,
    seed: int,
    args,
):
    expected = _expected_training_config(
        suite, task_id, total_timesteps, seed, args
    )
    if not checkpoint_complete(path):
        return False, "checkpoint files or valid manifest are missing"

    pretrained_encoder = _arg(args, "pretrained_encoder", None)
    matches, reason = identity_checkpoint_matches(
        path,
        expected,
        pretrained_encoder=pretrained_encoder,
    )
    if not matches:
        return False, reason

    return _strict_manifest_args_match(path, expected)


def _append_bool(cmd: list[str], name: str, value: bool):
    cmd.append(f"--{name}" if value else f"--no-{name}")


def train_one_baseline(
    suite: str,
    task_id: int,
    total_timesteps: int,
    seed: int,
    args,
):
    if suite not in TASK_SUITES:
        raise ValueError(f"unknown Atari suite {suite!r}")
    if task_id < 0 or task_id >= len(TASK_SUITES[suite]):
        raise ValueError(f"task_id={task_id} is invalid for {suite}")

    run_dir = scratch_checkpoint_dir(
        args.save_root, suite, task_id, total_timesteps, seed
    )
    event_dir = scratch_event_dir(
        args.runs_root, suite, task_id, total_timesteps, seed
    )
    analysis_dir = (
        pathlib.Path(_arg(args, "analysis_root", "analysis_runs_atari_scratch"))
        / scratch_tag(suite, task_id, total_timesteps, seed)
        / scratch_run_name(suite, task_id, seed)
    )
    expected = _expected_training_config(
        suite, task_id, total_timesteps, seed, args
    )

    if checkpoint_complete(run_dir) and not args.force_retrain:
        matches, reason = checkpoint_matches(
            run_dir, suite, task_id, total_timesteps, seed, args
        )
        if matches:
            print(
                f"[scratch] {suite}/task_{task_id}/seed_{seed} already complete: "
                f"{run_dir}"
            )
            return run_dir
        print(f"[scratch] stale checkpoint ({reason}); retraining: {run_dir}")

    # Never let a retry mix old and new TensorBoard points; FT is an AUC metric.
    for path in (run_dir, event_dir, analysis_dir):
        if path.exists():
            shutil.rmtree(path)

    cmd = [
        sys.executable,
        "run_ppo_continual.py",
        "--model-type=cka-rl",
        f"--task-suite={suite}",
        f"--task-id={task_id}",
        "--seq-idx=0",
        f"--seed={seed}",
        f"--save-dir={run_dir}",
        f"--runs-root={args.runs_root}",
        f"--tag={scratch_tag(suite, task_id, total_timesteps, seed)}",
        f"--total-timesteps={total_timesteps}",

        # PPO parity with continual training
        f"--learning-rate={expected['learning_rate']}",
        f"--num-envs={expected['num_envs']}",
        f"--num-steps={expected['num_steps']}",
        f"--gamma={expected['gamma']}",
        f"--gae-lambda={expected['gae_lambda']}",
        f"--num-minibatches={expected['num_minibatches']}",
        f"--update-epochs={expected['update_epochs']}",
        f"--clip-coef={expected['clip_coef']}",
        f"--ent-coef={expected['ent_coef']}",
        f"--vf-coef={expected['vf_coef']}",
        f"--max-grad-norm={expected['max_grad_norm']}",
        f"--eval-every={expected['eval_every']}",
        f"--num-evals={expected['num_evals']}",

        # Root CKA setup.  No historical pool exists, but we pass all knobs so
        # the run manifest is explicit and reproducible.
        "--fusion-mode=classic_cka",
        f"--pool-size={expected['pool_size']}",
        f"--alpha-init={expected['alpha_init']}",
        f"--alpha-major={expected['alpha_major']}",
        f"--alpha-factor={expected['alpha_factor']}",
        f"--alpha-learning-rate={expected['alpha_learning_rate']}",
        f"--alpha-warmup-steps={expected['alpha_warmup_steps']}",
        f"--alpha-entropy-reg={expected['alpha_entropy_reg']}",
        f"--alpha-mass-reg={expected['alpha_mass_reg']}",
        "--no-use-alpha-scale",
        "--no-fix-alpha-scale",
        "--no-use-alpha-mass",

        # Encoder parity
        f"--shared-dim={expected['shared_dim']}",
        f"--head-hidden-dim={expected['head_hidden_dim']}",
        f"--distill-encoder-lr-mult={expected['distill_encoder_lr_mult']}",
        f"--drift-reg={expected['drift_reg']}",
        f"--analysis-root={_arg(args, 'analysis_root', 'analysis_runs_atari_scratch')}",
        f"--analysis-log-every={int(_arg(args, 'analysis_log_every', 0))}",

        # Scratch has no historical merge.  Disable post-training behavioral
        # buffer collection while keeping all distillation metadata explicit.
        "--no-distillation",
        "--no-collect-cosine-buffers",
        "--distill-extra-steps=0",
        f"--max-distill-buffer={expected['max_distill_buffer']}",
        f"--similarity-samples={expected['similarity_samples']}",
        f"--distill-max-samples={expected['distill_max_samples']}",
        f"--distill-epochs={expected['distill_epochs']}",
        f"--distill-lr={expected['distill_lr']}",
        f"--distill-batch-size={expected['distill_batch_size']}",
        f"--distill-test-frac={expected['distill_test_frac']}",
    ]

    _append_bool(cmd, "torch-deterministic", expected["torch_deterministic"])
    _append_bool(cmd, "anneal-lr", expected["anneal_lr"])
    _append_bool(cmd, "norm-adv", expected["norm_adv"])
    _append_bool(cmd, "clip-vloss", expected["clip_vloss"])
    _append_bool(cmd, "fix-alpha", expected["fix_alpha"])
    _append_bool(cmd, "constrain-alpha-mass", expected["constrain_alpha_mass"])
    _append_bool(cmd, "encoder-from-base", expected["encoder_from_base"])
    _append_bool(cmd, "train-shared", expected["train_shared"])
    _append_bool(cmd, "freeze-root-encoder", expected["freeze_root_encoder"])
    _append_bool(cmd, "distill-select-best-val", expected["distill_select_best_val"])
    _append_bool(cmd, "save-analysis-snapshots", bool(_arg(args, "save_analysis_snapshots", False)))
    _append_bool(cmd, "cuda", expected["cuda"])

    if expected["target_kl"] is not None:
        cmd.append(f"--target-kl={expected['target_kl']}")
    if expected["success_threshold"] is not None:
        cmd.append(f"--success-threshold={expected['success_threshold']}")

    pretrained_encoder = expected["pretrained_encoder"]
    if pretrained_encoder:
        cmd.append(f"--pretrained-encoder={pretrained_encoder}")

    print(
        f"\n>>> scratch | {suite} | task {task_id} "
        f"({get_task_name(task_id, suite)}) | seed {seed} | "
        f"{total_timesteps} steps <<<"
    )
    subprocess.run(cmd, check=True)

    if not checkpoint_complete(run_dir):
        raise RuntimeError(
            f"scratch baseline finished but checkpoint is incomplete: {run_dir}"
        )
    matches, reason = checkpoint_matches(
        run_dir, suite, task_id, total_timesteps, seed, args
    )
    if not matches:
        raise RuntimeError(f"scratch baseline has unexpected identity: {reason}")
    return run_dir


def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "--task-suites",
        nargs="+",
        default=["freeway", "space_invaders"],
        choices=sorted(TASK_SUITES),
    )
    p.add_argument(
        "--seeds", nargs="+", type=int, default=DEFAULT_SCRATCH_SEEDS
    )

    # PPO defaults deliberately mirror run_continual_benchmark.py.
    # Do NOT tune scratch separately: doing so invalidates the FT denominator.
    p.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="MUST equal the continual run's total-timesteps for FT.",
    )
    p.add_argument("--learning-rate", type=float, default=2.5e-4)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument(
        "--anneal-lr", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument(
        "--norm-adv", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--clip-coef", type=float, default=0.1)
    p.add_argument(
        "--clip-vloss", action=argparse.BooleanOptionalAction, default=True
    )
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
    )

    # Root/pool knobs are retained so the CLI can mirror benchmark settings.
    p.add_argument("--pool-size", type=int, default=5)
    p.add_argument(
        "--alpha-init", choices=["Randn", "Major", "Uniform"], default="Randn"
    )
    p.add_argument("--alpha-major", type=float, default=0.6)
    p.add_argument("--alpha-factor", type=float, default=1e-3)
    p.add_argument(
        "--fix-alpha", action=argparse.BooleanOptionalAction, default=False
    )
    p.add_argument("--alpha-learning-rate", type=float, default=2.5e-4)
    p.add_argument("--alpha-warmup-steps", type=int, default=5_000)
    p.add_argument("--alpha-entropy-reg", type=float, default=0.01)
    p.add_argument("--alpha-mass-reg", type=float, default=0.05)
    p.add_argument(
        "--constrain-alpha-mass",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument(
        "--encoder-from-base", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument(
        "--train-shared", action=argparse.BooleanOptionalAction, default=False
    )
    p.add_argument(
        "--freeze-root-encoder",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--pretrained-encoder", default=None)
    p.add_argument("--shared-dim", type=int, default=512)
    p.add_argument("--head-hidden-dim", type=int, default=128)
    p.add_argument("--distill-encoder-lr-mult", type=float, default=0.1)
    p.add_argument("--drift-reg", type=float, default=1.0)

    # These do not affect scratch root training, but keeping the same defaults
    # makes manifests and imported benchmark calls unambiguous.
    p.add_argument("--max-distill-buffer", type=int, default=5_000)
    p.add_argument("--similarity-samples", type=int, default=512)
    p.add_argument("--distill-max-samples", type=int, default=2_000)
    p.add_argument("--distill-epochs", type=int, default=8)
    p.add_argument("--distill-lr", type=float, default=3e-4)
    p.add_argument("--distill-batch-size", type=int, default=256)
    p.add_argument("--distill-test-frac", type=float, default=0.2)
    p.add_argument(
        "--distill-select-best-val",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    p.add_argument("--success-thresholds-json", default=None)
    p.add_argument("--save-root", default=SCRATCH_SAVE_ROOT)
    p.add_argument("--runs-root", default="runs_atari")
    p.add_argument("--analysis-root", default="analysis_runs_atari_scratch")
    p.add_argument("--analysis-log-every", type=int, default=0)
    p.add_argument(
        "--save-analysis-snapshots",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    p.add_argument("--force-retrain", action="store_true")
    p.add_argument("--cpu", action="store_true")

    args = p.parse_args()
    args.success_thresholds = _load_thresholds(args.success_thresholds_json)

    if args.total_timesteps < 1 or args.num_envs < 1 or args.num_steps < 1:
        p.error("total_timesteps, num_envs and num_steps must be >= 1")
    if args.learning_rate <= 0:
        p.error("--learning-rate must be > 0")
    if not 0.0 < args.gamma <= 1.0 or not 0.0 <= args.gae_lambda <= 1.0:
        p.error("gamma must be in (0,1] and gae_lambda in [0,1]")
    if args.clip_coef <= 0 or args.max_grad_norm <= 0:
        p.error("clip_coef and max_grad_norm must be > 0")
    if args.ent_coef < 0 or args.vf_coef < 0:
        p.error("ent_coef and vf_coef must be >= 0")
    if args.target_kl is not None and args.target_kl <= 0:
        p.error("target_kl must be > 0 when provided")
    if args.num_minibatches < 1 or args.update_epochs < 1:
        p.error("num_minibatches and update_epochs must be >= 1")
    batch_size = args.num_envs * args.num_steps
    if batch_size < args.num_minibatches:
        p.error("num_minibatches cannot exceed num_envs * num_steps")
    if args.total_timesteps < batch_size:
        p.error("total_timesteps must be at least one PPO rollout")
    if batch_size % args.num_minibatches != 0:
        p.error("num_envs * num_steps must be divisible by num_minibatches")
    if args.eval_every <= 0 or args.num_evals < 1:
        p.error("eval_every and num_evals must be positive")
    if args.pool_size < 2:
        p.error("--pool-size must be >= 2")
    if args.train_shared and args.freeze_root_encoder:
        p.error("--train-shared and --freeze-root-encoder are contradictory")
    if args.distill_encoder_lr_mult <= 0:
        p.error("--distill-encoder-lr-mult must be > 0")
    if args.drift_reg < 0:
        p.error("--drift-reg must be >= 0")
    if args.analysis_log_every < 0:
        p.error("--analysis-log-every must be >= 0")
    if args.max_distill_buffer < 2 or args.distill_max_samples < 2:
        p.error("distillation buffer/sample budgets must be >= 2")
    if args.similarity_samples < 2:
        p.error("--similarity-samples must be >= 2")
    if args.distill_epochs < 1 or args.distill_batch_size < 1:
        p.error("distillation epochs/batch size must be >= 1")
    if not 0.0 <= args.distill_test_frac < 1.0:
        p.error("--distill-test-frac must be in [0,1)")

    return args


def main():
    args = parse_args()
    for suite in args.task_suites:
        for task_id in range(len(TASK_SUITES[suite])):
            for seed in args.seeds:
                train_one_baseline(
                    suite,
                    task_id,
                    args.total_timesteps,
                    seed,
                    args,
                )
    print(f"\nDone. Scratch baselines cached under: {args.save_root}")
    print("Use the same --total-timesteps/PPO/encoder settings in the continual benchmark.")


if __name__ == "__main__":
    main()
