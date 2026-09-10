mkdir -p "$HOME/Cont/Continual-RL-Project/crl_experiments/recovery_logs"

sbatch \
  --job-name=causal \
  --partition=h100 \
  --qos=normal \
  --gres=gpu:1 \
  --nodelist=kh036 \
  --time=00:10:00 \
  --cpus-per-task=4 \
  --mem=8G \
  --output="$HOME/Cont/Continual-RL-Project/crl_experiments/recovery_logs/recover_%j.out" \
  --error="$HOME/Cont/Continual-RL-Project/crl_experiments/recovery_logs/recover_%j.err" \
  --wrap='
SRC=/scratch/najar/Cont/Continual-RL-Project/crl_experiments/ethos_student_halfcheetah_windvel_80k
DST=$HOME/Cont/Continual-RL-Project/crl_experiments/ethos_student_halfcheetah_windvel_80k

echo "Running on $(hostname)"
echo "Checking $SRC"

if [ -d "$SRC" ]; then
    mkdir -p "$DST"
    cp -a "$SRC/." "$DST/"
    echo "RECOVERY SUCCESS"
    find "$DST" -maxdepth 4 -type d | head -100
else
    echo "RECOVERY FAILED: source directory no longer exists"
    exit 2
fi
'