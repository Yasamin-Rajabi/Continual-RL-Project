# Validation report

Date: 8 September 2026. Results concern this patched archive, not the draft's
reported learning performance.

| Check | Result |
|---|---|
| Python syntax in requested runners and new tests | 112 files parsed successfully |
| Shell syntax / LF endings | 16 scripts passed `bash -n`; no CRLF remaining |
| New CPU regression suite | 12 tests passed in each of 5 runner directories: 60 passes |
| Original pool sanity suite | Passed in HalfCheetah and Walker2D; logs included |
| Synthetic SAC chains | 4 conditions x 3 task occurrences x 6 suite/entry paths = 72 task runs |
| Synthetic learning budget | Each task used exactly Delta=40 interactions: 32 optimization-phase + 8 frozen-tail |
| Synthetic evaluation | Both pool/snapshot and deterministic/stochastic modes passed; optional adaptation counts checked |
| Retention/FT metric checks | Retention matrices, zero adaptation counts, unavailable FT, and analytic FT examples passed |
| Original pair-selection/distillation functions | AST-identical in HalfCheetah, Walker2D, and modern MetaWorld |
| Real MuJoCo / MetaWorld tasks | **Not run** |
| Full learning/convergence/performance reproduction | **Not run** |

## What the regression tests cover

Mixture densities against PyTorch's reference distribution; stable tanh Jacobian;
zero mixture weights; whole-action rather than actuator-wise component sampling;
actor mixing-weight gradients against finite differences; root/continuation
lifecycles; history immutability; sigmoid bounds; singleton warmup; exact ensemble
snapshot restoration; storage projection and subsequent overflow merges; bounded
buffers; aligned state/metadata subsampling; task-concatenation identity behavior;
and finalized-pool loading.

## What the synthetic chains mean

`tests/synthetic_smoke.py` executes the real `run_continual_benchmark.train_chain`
and `run_sac.py` update/saving code using deliberately substituted minimal
Gym/replay/logging adapters and a six-dimensional synthetic state. This detects
Python integration, optimizer, serialization, budget and metric bugs. It **does
not** test real SB3 replay behavior, Tyro parsing, Gymnasium auto-reset details,
MuJoCo physical model modifications, MetaWorld API compatibility, or convergence.
Its logs must not be described as environment experiment results.

The six entry paths are HalfCheetah-Vel, HalfCheetah-WindVel, Walker2D, Hopper,
modern MetaWorld and the legacy MetaWorld directory. Each ran baseline/combined
in parameter/policy composition space, including an overflow merge.

## Missing validation and next checks

The environment had Torch/NumPy/Pytest but lacked simulator/training dependencies.
A dependency-installation attempt failed because package-host name resolution
was unavailable. Therefore real simulator `tasks.py --check`, `walker2d_smoke.py`,
Tyro CLI parsing and a real `--quick-test` still need to run in the user's
installed environment. No return improvements or published-number reproduction
are asserted.

The new Hopper suite and randomized MetaWorld paper-task option particularly
require those real checks before large runs. Old task-conditioned checkpoints
must not be loaded as revised results.

Machine-readable details and individual logs are in `validation_logs/`.
