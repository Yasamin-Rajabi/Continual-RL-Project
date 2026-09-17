# Hopper extension

This new directory was added because the uploaded project did not contain a
Hopper runner. It uses native Hopper-v5 observations with six fixed-objective
dynamics tasks, each repeated twice. It is not a reproduction of a supplied
Hopper experiment and has not been validated on a real MuJoCo installation here.

See ../IMPLEMENTATION_NOTES.md and ../EVALUATION_GUIDE.md.

```bash
python -m pip install -r requirements.txt
python tasks.py --check
python run_continual_benchmark.py --quick-test --skip-forward-transfer
bash run_comparison.sh --skip-forward-transfer
```
