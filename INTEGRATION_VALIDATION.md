# Validation report

## Executed in the development container

Python 3.13.5, PyTorch 2.10.0+cpu, NumPy 2.3.5. The user's target is a different
runtime (Python 3.11 / PyTorch 2.8 / MuJoCo 2.3.7); see the limitations below.

| Test group | Result |
|---|---:|
| Baseline models, gradients, protected parameters, checkpoint reload, Ant dimensions, compression modes, identity, configuration, collector and orchestration | 30 passed |
| Real main `run_sac.py` tiny three-occurrence lifecycle, including finalization/logging, for baseline, KL merge, random merge and random discard | 4 passed |
| New-baseline tiny three-occurrence training, reload, CSV, retention, FT and completed-cell reuse (all seven methods) | 7 passed |
| Existing policy-space regression tests, HalfCheetah | 12 passed |
| Existing policy-space regression tests, Walker2D | 12 passed |
| Existing policy-space regression tests, AntDir engine | 12 passed |

Total: **77 tests passed in isolated test groups**, not 77 real environment
training experiments. Baseline tests use real PyTorch tensors, differentiation,
Adam, lifecycle methods and serialization. In the pipeline fixtures only, the
simulator, replay container, TensorBoard and argument-parsing dependencies are
replaced with test doubles. The latest pipeline fixtures omit retention-figure
rendering; numerical output/CSV/cell caching remain exercised. The collector
has its own plot/CSV tests. These doubles are never imported by production jobs.

### Default-code equivalence check

The uploaded original and patched HalfCheetah engines were each run on the same
deterministic test environment for three occurrences, including an overflow.
For **baseline** and **combined_policy with default KL merging**, every tensor
in the final actor/model state matched exactly at all three boundaries
(rtol=0, atol=0; NaNs compared equal). This checks that the opt-in ablations did
not alter the exercised default learning/finalization path. It does not prove
bitwise equality across different CUDA hardware or software versions.

All **30 original shell scripts** are byte-for-byte unchanged. No original
project file is removed. Changed/new Python sources passed compilation and all
shell files passed `bash -n`. SHA-256 hashes and the exact file list are in
`INTEGRATION_CHANGES.json`; the unified diff of modified original text files is
`INTEGRATION_DIFF.patch`.

### Runner limitations observed

The instrumented development Python environment intermittently stalled during
or after repeated multi-method pytest batches, despite successful test reports
from the isolated groups. Disabling plugin autoload and running groups/methods
in fresh processes gave the reported results, but did not eliminate every
intermittent runner stall. No production fix has been inferred from that
behavior, and no forced process-exit workaround is part of the package.

Recommended isolated test commands (from repository root):

```bash
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q tests_paper/test_models.py tests_paper/test_ablation.py tests_paper/test_identity_and_config.py tests_paper/test_orchestration.py tests_paper/test_collection.py
python -m pytest -q tests_paper/test_main_sac_pipeline.py
# Run the seven parameterized pipeline cases separately when necessary.
python -m pytest -q tests_paper/test_training_pipeline.py
CRL_FAMILY=half-cheetah python -m pytest -q tests/test_policy_space.py
CRL_FAMILY=Walker2D python -m pytest -q tests/test_policy_space.py
CRL_FAMILY=AntDir python -m pytest -q tests/test_policy_space.py
```

## Not executed / not established

* No real Gymnasium/MuJoCo/SB3 environment runs: those dependencies were not
  installed in this sandbox and package retrieval was unavailable.
* No execution in the user's Apptainer image, on an H100, or on Slurm/Kuma.
* No long-horizon convergence tests, benchmark scores or published-result
  reproduction. AntDir's preset is a starting configuration, not a tuned one.
* No successful live submission/cancellation/queue test. Launcher validation
  covers dry-run behavior, argument extraction, file identity and reference
  reservation; actual scheduling depends on Kuma.
* No complete validation of every historical project revision. Only explicitly
  approved uploaded-source transitions are accepted by the checkpoint guard.

Before allocating a large batch, run `paper_runs/smoke_real.py` in the user's
container. First check all environment constructors, then use `--train
--evaluate` in a GPU allocation. This is the remaining acceptance test for the
actual simulator and dependency stack. See INTEGRATION_README.md for commands.
