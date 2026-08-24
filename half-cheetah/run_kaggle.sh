#!/usr/bin/env bash
# run_kaggle.sh — یک دستور برای کل پایپ‌لاین، مناسب سلول‌های نوت‌بوک Kaggle.
#
# چرا یک فایل واحد: Kaggle GPU quota هفتگی محدود است (۳۰ ساعت) و session ها
# قطع می‌شوند. run_continual_benchmark.py و scratch_baselines.py هر دو از قبل
# checkpoint_complete() را چک می‌کنند، یعنی اجرای دوباره‌ی همین اسکریپت فقط
# قسمت‌های ناتمام را ادامه می‌دهد — کافی است در سلول بعدی دوباره صدایش بزنید.
#
# استفاده در یک سلول نوت‌بوک:
#   !bash run_kaggle.sh setup
#   !bash run_kaggle.sh sanity
#   !bash run_kaggle.sh pretrain hc
#   !bash run_kaggle.sh baselines hc
#   !bash run_kaggle.sh continual hc
#   !bash run_kaggle.sh all hc          # چهار مرحله‌ی بالا پشت‌سرهم
#
# برای Ant: بجای hc بنویسید ant  (بعد از انجام کالیبراسیون سرعت، بخش ۵ README)
#
# هر مرحله lock فایل خودش را دارد؛ اگر session قطع شد همان دستور را دوباره بزنید.

set -euo pipefail
cd "$(dirname "$0")"

OUT="${KAGGLE_WORKING:-/kaggle/working}"
mkdir -p "$OUT/pretrained_encoders" "$OUT/agents" "$OUT/plots" "$OUT/logs"

# --------------------------------------------------------------------------
# پیکربندی — همینجا مقادیر را عوض کنید، نه در خط فرمان
# --------------------------------------------------------------------------
SEEDS="1 2 3"
TOTAL_TIMESTEPS=50000          # به‌جای 300000 پیش‌فرض؛ بخش ۶ README را ببینید
PRETRAIN_STEPS_PER_TASK=100000
PRETRAIN_EPOCHS=30

HC_SUITE="halfcheetah_vel"
ANT_SUITE="ant_vel"

# --------------------------------------------------------------------------
suite_of() { [ "$1" = "ant" ] && echo "$ANT_SUITE" || echo "$HC_SUITE"; }
enc_dir()  { echo "$OUT/pretrained_encoders/$1"; }

step_setup() {
    echo ">>> setup: نصب وابستگی‌ها"
    pip install -q -r requirements.txt --break-system-packages 2>/dev/null || \
    pip install -q -r requirements.txt
    echo "OK"
}

step_sanity() {
    echo ">>> sanity: تأیید ساختاری pool و tasks (بدون GPU)"
    python3 sanity_check_pool.py
    python3 tasks.py
    python3 tasks.py --check
    echo "OK"
}

step_pretrain() {
    local fam="$1" suite; suite="$(suite_of "$fam")"
    local out; out="$(enc_dir "$fam")"
    if [ -f "$out/fc.pt" ]; then
        echo ">>> pretrain[$fam]: fc.pt از قبل موجود است، رد می‌شود ($out)"
        return 0
    fi
    echo ">>> pretrain[$fam]: TD-JEPA روی $suite"
    if [ "$fam" = "ant" ]; then
        # ⚠️ این سه سرعت موقتی‌اند — قبل از اجرای واقعی کالیبره کنید (بخش ۵ README)
        python3 tdjepa_pretrain.py \
            --task-suite "$suite" \
            --pretrain-velocities 0.4 0.9 1.4 \
            --heldout-velocities 0.25 1.0 2.0 \
            --steps-per-task "$PRETRAIN_STEPS_PER_TASK" --epochs "$PRETRAIN_EPOCHS" \
            --out "$out" 2>&1 | tee "$OUT/logs/pretrain_${fam}.log"
    else
        python3 tdjepa_pretrain.py \
            --task-suite "$suite" \
            --pretrain-velocities 0.75 1.75 2.75 \
            --heldout-velocities 0.5 2.0 3.0 \
            --steps-per-task "$PRETRAIN_STEPS_PER_TASK" --epochs "$PRETRAIN_EPOCHS" \
            --out "$out" 2>&1 | tee "$OUT/logs/pretrain_${fam}.log"
    fi
    echo "OK -> $out/fc.pt"
}

step_baselines() {
    local fam="$1" suite; suite="$(suite_of "$fam")"
    local enc; enc="$(enc_dir "$fam")/fc.pt"
    [ -f "$enc" ] || { echo "ERROR: ابتدا 'pretrain $fam' را اجرا کنید"; exit 1; }
    echo ">>> baselines[$fam]: بیس‌لاین‌های scratch (همان انکودر ران‌های continual)"
    python3 scratch_baselines.py \
        --task-suites "$suite" \
        --total-timesteps "$TOTAL_TIMESTEPS" \
        --pretrained-encoder "$enc" --encoder-linear-out \
        --analysis-root "$OUT/logs/scratch_${fam}" \
        2>&1 | tee -a "$OUT/logs/baselines_${fam}.log"
    echo "OK"
}

step_continual() {
    local fam="$1" suite; suite="$(suite_of "$fam")"
    local enc; enc="$(enc_dir "$fam")/fc.pt"
    [ -f "$enc" ] || { echo "ERROR: ابتدا 'pretrain $fam' را اجرا کنید"; exit 1; }
    echo ">>> continual[$fam]: ران کامل benchmark (S0 خط پایه + S4 TD-JEPA)"

    # S0 — خط پایه، بدون انکودر پیش‌آموزش‌دیده
    python3 run_continual_benchmark.py \
        --task-suites "$suite" --seeds $SEEDS \
        --total-timesteps "$TOTAL_TIMESTEPS" \
        --save-root "$OUT/agents/${fam}_S0" \
        --plots-root "$OUT/plots/${fam}_S0" \
        2>&1 | tee -a "$OUT/logs/continual_${fam}_S0.log"

    # S4 — TD-JEPA
    python3 run_continual_benchmark.py \
        --task-suites "$suite" --seeds $SEEDS \
        --total-timesteps "$TOTAL_TIMESTEPS" \
        --pretrained-encoder "$enc" --encoder-linear-out \
        --save-root "$OUT/agents/${fam}_S4" \
        --plots-root "$OUT/plots/${fam}_S4" \
        2>&1 | tee -a "$OUT/logs/continual_${fam}_S4.log"
    echo "OK"
}

step_all() {
    local fam="$1"
    step_pretrain "$fam"
    step_baselines "$fam"
    step_continual "$fam"
}

case "${1:-}" in
    setup)      step_setup ;;
    sanity)     step_sanity ;;
    pretrain)   step_pretrain "${2:?hc یا ant}" ;;
    baselines)  step_baselines "${2:?hc یا ant}" ;;
    continual)  step_continual "${2:?hc یا ant}" ;;
    all)        step_all "${2:?hc یا ant}" ;;
    *)
        echo "Usage: bash run_kaggle.sh {setup|sanity|pretrain|baselines|continual|all} [hc|ant]"
        exit 1
        ;;
esac
