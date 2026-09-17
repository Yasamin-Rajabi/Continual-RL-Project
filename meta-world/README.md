> **Updated implementation:** read `../IMPLEMENTATION_NOTES.md` and
> `../EVALUATION_GUIDE.md` first. The material below describes the uploaded
> project's older presets; its old defaults and performance claims do not
> validate this revised task-blind/policy-space implementation. Use
> `run_comparison.sh` for the new defaults and train fresh checkpoints.

# CKA-RL روی Meta-World

پورت کدبیس HalfCheetah شما (نسخه‌ی فعلی، با `experiment_identity`, `--test-adapt-steps`, `--freeze-root-encoder`, `--condition-alpha-scale`) به Meta-World، با انتخاب تسک و چینشی که برای بودجه‌ی **۱۵۰k گام** طراحی شده.

---

## ۰. اجرا در Kaggle

**Accelerator = `GPU T4 x2`** (نه P100). هر شرط یک پروسه‌ی مستقل است؛ با دو GPU می‌شود دو شرط را همزمان روی دو اکانت جلو برد و مدل کوچک است پس گلوگاه فیزیک MuJoCo روی CPU است، نه توان تک‌هسته‌ای GPU.

`kaggle_runner.ipynb` را باز کنید، `STAGE` را ست کنید، و `Save Version → Save & Run All (Commit)`. سقف‌ها: **۱۲ ساعت** هر session و **۲۰ گیگابایت** `/kaggle/working` که خودکار ذخیره می‌شود.

### هر ایده یک stage جدا

این خواسته‌ی اصلی شما بود: هیچ stage‌ای مجبور نیست کل برنامه را داخل ۱۲ ساعت جا کند.

| stage | زمان تقریبی | چه چیزی را جواب می‌دهد |
|---|---|---|
| `smoke` | ~۱۰ دقیقه | آیا پایپ‌لاین اصلاً سرتاسر اجرا می‌شود |
| `pilot` | ~۱ ساعت | آیا این تسک‌ها در ۱۵۰k گام چیزی یاد می‌گیرند |
| `baselines s0` | ~۱.۵ ساعت | مخرج FT |
| `cond s0 1` | ~۴ تا ۷ ساعت | `baseline` — classic_cka + کسینوس |
| `cond s0 2` | ~۴ تا ۷ ساعت | `distil_only` — classic_cka + KL رفتاری + distillation |
| `cond s0 3` | ~۴ تا ۷ ساعت | `weight_only` — weight_delta + کسینوس |
| `cond s0 4` | ~۴ تا ۷ ساعت | `combined` — weight_delta + KL + distillation |
| `pretrain` | ~۰.۵ ساعت | انکودر TD-JEPA (ایده‌ی لایه‌ی مشترک) |
| `baselines s4` + `cond s4 N` | مثل بالا | همان چهار شرط با انکودر پیش‌آموزش‌دیده |
| `report s0` | چند دقیقه | متریک و پلات روی هرچه موجود است |

هر شرط با ۳ seed حدود **۳.۸۴ میلیون گام** است ⇒ ۴.۳ تا ۷.۱ ساعت. یعنی **یک شرط = یک session** و راحت زیر سقف جا می‌شود.

### تقسیم پیشنهادی بین دو اکانت

| روز | اکانت A | اکانت B |
|---|---|---|
| ۱ | `smoke` → `pilot` → `baselines s0` | `baselines s0` |
| ۱ | `cond s0 1` | `cond s0 3` |
| ۲ | `cond s0 2` | `cond s0 4` |
| ۲ (اگر وقت ماند) | `pretrain` → `baselines s4` → `cond s4 1` | `cond s4 3` |

بعد خروجی هر دو را در یک نوت‌بوک merge کنید و `report s0` بزنید. `--skip-training` دارد، پس روی هرچه موجود باشد گزارش می‌سازد.

---

## ۱. نصب Meta-World

از ایده‌ی خودتان استفاده کردم و کمی محکم‌ترش کردم:

```bash
python3 -m pip install -q "mujoco>=3.0,<4"
python3 -m pip install -q --no-deps \
    "git+https://github.com/Farama-Foundation/Metaworld.git@c822f28f582ba1ad49eb5dcf61016566f28003ba"
grep -v -E "^\s*(metaworld|mujoco)" requirements.txt > /tmp/reqs_clean.txt
python3 -m pip install -q -r /tmp/reqs_clean.txt
```

سه دلیل که چرا این ترتیب:

۱. **commit پین‌شده** به‌جای `@master`. تلاش قبلی با `@master` شکست خورد (`git checkout -q master did not run successfully`) — مسیر داخلی pip برای shallow clone روی Kaggle ناپایدار است. ضمناً master فعلی به API نسخه‌ی v3 رفته و `ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE` که نام‌های تسک v2 ما به آن تکیه دارند دیگر همان معنا را ندارد.

۲. **`--no-deps`** یعنی pip هرگز pin های کهنه‌ی خود Meta-World را resolve نمی‌کند و torch/numpy/gymnasium را زیر پای شما downgrade نمی‌کند. برای همین mujoco باید *قبلش* نصب شود و بقیه از `requirements.txt` بیاید.

۳. **فیلتر کردن دو خط** از requirements تا با pin نجنگند. `requirements.txt` عمداً `metaworld` و `mujoco` ندارد و همین دلیل در خودش کامنت شده.

`metaworld_envs.py` هم یک resolver دارد که سه نسل API را به ترتیب امتحان می‌کند (dict نسخه‌ی v2 ← `metaworld.MT1` ← ثبت gymnasium)، پس اگر pin را عوض کردید کد بی‌صدا نمی‌شکند. مرحله‌ی `setup` چاپ می‌کند کدام API فعال شده.

---

## ۲. انتخاب تسک — چرا این چهار تا

دو قید منتشرشده تعیین‌کننده بودند.

**قید اول: بودجه.** Continual World صریحاً می‌گوید CW10 را تسک‌هایی انتخاب کرده که «در بودجه‌ی فرضی ۱ میلیون گام نه خیلی آسان باشند نه خیلی سخت». بودجه‌ی شما ۱۵۰k است — ۷ برابر کمتر. در همان CW10، `stick-pull` و `shelf-place` در رده‌ی **خیلی سخت** پارتیشن سختی MT50 هستند و در ۱۵۰k هرگز از صفر بالا نمی‌آیند؛ فقط نویز به FG، BWT و FT اضافه می‌کنند. (لیست قبلی `meta-world/tasks.py` شما هر دو را داشت.)

پس اشتراک «در CW10» و «در رده‌ی آسان MT50» را گرفتم. دقیقاً چهار تسک هر دو شرط را دارند:

| id | تسک | نقش |
|---|---|---|
| 0 | `window-close-v2` | مبدأ triplet سوم CW |
| 1 | `faucet-close-v2` | طبق تحلیل زیرفضای CSP، هم‌ناحیه با `peg-unplug-side` |
| 2 | `handle-press-side-v2` | **مزاحم** طراحی‌شده‌ی triplet |
| 3 | `peg-unplug-side-v2` | مقصد triplet؛ از ۰ و ۱ انتقال می‌گیرد |

**قید دوم: ساختار انتقال.** CW هشت triplet سه‌تسکی دارد که عمداً طوری ساخته شده‌اند که «از تسک ۱ به تسک ۳ انتقال مثبت باشد ولی تسک ۲ به‌عنوان مزاحم تداخل کند». از آن هشت‌تا **دقیقاً یکی هر سه تسکش در رده‌ی آسان است**:

```
window-close → handle-press-side → peg-unplug-side
```

عیناً در موقعیت‌های ۰ تا ۲ دنباله نشسته. تمیزترین آزمون ممکن برای ادعای مرکزی شماست: `weight_delta` از ترکیب یادگرفته‌ای از **سیاست‌های کامل قبلی** شروع می‌کند، پس باید بتواند وزن `window-close` را بالا و وزن مزاحم را نزدیک صفر ببرد؛ `classic_cka` باید همان را از دلتاهای باقی‌مانده بازسازی کند.

---

## ۳. چینش

```
0, 2, 3, 1, 0, 3, 2, 1
```

- موقعیت ۰-۲ = triplet سوم CW عیناً.
- هر تسک دقیقاً دو بار، با فاصله‌های **۳، ۴، ۴، ۵** (assert شده). فاصله‌های متنوع یعنی ماتریس retention می‌تواند درباره‌ی فراموشی **به‌عنوان تابعی از فاصله** حرف بزند، نه یک عدد.
- موقعیت ۵، `peg-unplug-side` را یک قدم بعد از `window-close` می‌گذارد — همان جفت انتقالی، این بار بدون مزاحم. تفاوت موقعیت ۲ و ۵ خوانش مستقیم هزینه‌ی مزاحم برای هر متد است.
- دومین ظهور هر تسک جایی است که `weight_delta` باید قاطع ببرد: سیاست کاملِ همان تسک در pool هست و وزن‌ها فقط باید پیدایش کنند.
- هیچ تسکی پشت‌سرهم تکرار نمی‌شود.

**`pool_size = 4`:** با ۸ موقعیت و ۴ تسک، فشرده‌سازی *درست* یعنی ادغام هر جفت هم‌تسک. پس merge یک **ground truth قابل‌راستی‌آزمایی** پیدا می‌کند و «آیا KL رفتاری بیشتر از کسینوس جفت‌های هم‌تسک را پیدا می‌کند؟» یک عدد قابل‌اندازه‌گیری می‌شود — نه یک حرف کیفی. ۴ رویداد merge در هر ران (assert شده).

---

## ۴. stage `smoke` — چرا اضافه شد

خواسته‌ی شما بود و کاملاً درست است: قبل از خرج کردن ساعت GPU باید بدانید پایپ‌لاین سرتاسر کار می‌کند.

`smoke` از suite `mw_smoke2` (دو تسک) با دنباله‌ی `0 1 0`، ۳۰۰۰ گام هر تسک و `pool_size=2` استفاده می‌کند. عدد ۲ عمدی است: با ۳ موقعیت، pool به ۳ می‌رسد و **یک merge واقعاً fire می‌کند** — همان بخشی که بیشترین احتمال شکست بی‌صدا را دارد. کل مسیر (chain، merge، retention، survey metrics، scratch baseline) در حدود ۱۰ دقیقه تست می‌شود.

---

## ۵. 🔴 دو چیزی که در پورت اصلاح شد

**۱) `FT_return` روی Meta-World غلط بود.** نسخه‌ی HalfCheetah از میان‌بر جبری `FT_i = 1 − R_i/R_i^b` استفاده می‌کرد که فقط با `r_max = 0` معتبر است (آنجا پاداش `−|velocity_error| − ctrl_cost` همیشه ≤ ۰ بود). پاداش Meta-World **مثبت** و در بازه‌ی تقریبی `[0, 10]` هر گام است، پس آن میان‌بر **علامت متریک را وارونه می‌کرد**. حالا بازده با `MAX_REWARD_PER_STEP × HORIZON` نرمال می‌شود و همان فرمول تحت‌اللفظی `FT_success` اعمال می‌شود.

> `FT_success` را متریک اصلی گزارش کنید — `success` معیار خود بنچمارک است، بازده فقط پروکسی shaped.

**۲) `MetaWorldInfoWrapper` باید زیرکلاس `gymnasium.Env` باشد.** نسخه‌ی اول را یک کلاس ساده با `__getattr__` نوشته بودم و `gymnasium.wrappers.TimeLimit` با `AssertionError: Expected env to be a gymnasium.Env` رد کرد. حالا زیرکلاس `Env` است و دستی delegate می‌کند — نه `Wrapper`، چون `Wrapper.__init__` همان assert را روی env داخلی هم می‌زند و بسته به نسخه‌ی Meta-World ممکن است آن env یک `gym.Env` قدیمی باشد. ضمناً خروجی ۴-تایی gym قدیمی و `reset` بدون info را هم به فرم gymnasium نرمال می‌کند.

---

## ۶. پارامترها

| پارامتر | مقدار | دلیل |
|---|---|---|
| `total_timesteps` | `150_000` | خواسته‌ی شما؛ اعتبارش با `pilot` سنجیده می‌شود |
| `pool_size` | `4` | بخش ۳ |
| افق اپیزود | `200` | قرارداد Continual World، نه پیش‌فرض ۵۰۰. در ۱۵۰k، افق ۵۰۰ فقط ۳۰۰ اپیزود می‌دهد؛ ۲۰۰ می‌دهد ۷۵۰. `success` اپیزودی است پس اپیزود بیشتر = سیگنال بیشتر |
| `_freeze_rand_vec` | `True` | پروتکل MT10/MT50 خود Meta-World. نسخه‌ی goal-observable تصادفی در TD-MPC صریحاً **سخت‌تر** توصیف شده |
| seed محیط | `12345 + task_id` | ⚠️ با هدف فریزشده، **seed خودِ تسک است**. کد قبلی `np.random.randint(0,1024)` داشت که با فریز، محیط train و eval را به دو تسک متفاوت می‌کرد |
| `success` | latch‌شده در اپیزود | معیار CW «موفقیت در هر نقطه‌ای از اپیزود» است |
| `task_error` | `obj_to_target` | جای `velocity_error` را می‌گیرد، پس `metrics.py` و `plots.py` بدون تغییر کار می‌کنند |
| `MW_COMMIT` | `c822f28…` | بخش ۱ |

---

## ۷. 🔴 قبل از هر چیز

**هیچ منبعی منحنی SAC تک‌تسکی روی Meta-World در مقیاس ۱۵۰k گزارش نکرده** — قرارداد جامعه ۱ میلیون گام است (Continual World، TD-MPC). انتخاب بالا یک **برون‌یابی مستدل از پارتیشن سختی MT50 است، نه واقعیت اندازه‌گیری‌شده.**

```bash
bash run_kaggle.sh smoke     # ~۱۰ دقیقه
bash run_kaggle.sh pilot     # ~۱ ساعت
```

خواندن نتیجه‌ی pilot:

- **همه ≥ ۰.۶** → طبق برنامه پیش بروید.
- **همه ≥ ۰.۶ ولی اکثراً زیر ۳۰k رسیده‌اند** → زودتر از موعد اشباع؛ AUC را plateau تسخیر می‌کند و متدها یکسان به نظر می‌رسند. `TOTAL_TIMESTEPS` را کم کنید یا به `mw_easy6` بروید.
- **هر تسکی نزدیک صفر ماند** → عوضش کنید. جایگزین‌های رده‌ی آسان: `door-close-v2`, `drawer-close-v2`, `button-press-topdown-v2`, `plate-slide-v2`.

ران‌های from-scratch این مرحله **همان بیس‌لاین‌های FT** هستند (همان `--runs-root`)، پس هدر نمی‌رود.

---

## ۸. فایل‌ها

| فایل | وضعیت |
|---|---|
| `tasks.py` | 🆕 انتخاب تسک، سه دنباله (easy4 / easy6 / smoke2) |
| `metaworld_envs.py` | 🆕 ساخت محیط، افق، latch موفقیت، `task_error`، resolver سه‌نسلی API |
| `pilot_check.py` | 🆕 بررسی امکان‌سنجی ۱۵۰k |
| `run_kaggle.sh` | 🆕 هر ایده یک stage |
| `kaggle_runner.ipynb` | 🆕 نوت‌بوک با clone از ریپوی public |
| `requirements.txt` | ✏️ بدون metaworld/mujoco (بخش ۱) |
| `metrics.py` | ✏️ رفع `FT_return`، تغییر نام `task_error` |
| `run_sac.py` | ✏️ suite، پیش‌فرض ۱۵۰k، `pool_size=4`، تغییر نام |
| `plots.py`, `run_continual_benchmark.py`, `scratch_baselines.py`, `run_eval_custom.py`, `estimate_timing.py` | ✏️ پیش‌فرض‌ها و برچسب‌ها |
| `tdjepa_pretrain.py` | ✏️ جمع‌آوری داده روی تسک‌های held-out Meta-World |
| `cka_rl.py`, `knowledge_pools.py`, `shared_arch.py`, `policy_utils.py`, `td_jepa.py`, `analysis_logging.py`, `sanity_check_pool.py`, `experiment_identity.py` | ✅ **دست‌نخورده** — متد اصلی شما تغییری نکرده |

---

## ۹. وضعیت تست

- ✅ همه‌ی فایل‌ها syntax-check شده‌اند، `run_kaggle.sh` با `bash -n` تأیید شد
- ✅ ویژگی‌های دنباله assert شد: فاصله‌های `{3,4,4,5}`، بدون تکرار پشت‌سرهم، triplet در موقعیت ۰-۲، ۴ merge با `pool_size=4`، ۱ merge در smoke
- ✅ نرمال‌سازی `FT_return` عددی تست شد
- ✅ همه‌ی فلگ‌های استفاده‌شده در `run_kaggle.sh` با CLI واقعی تطبیق داده شدند
- ❌ **اجرا نشده** — در محیط تحلیل نه torch بود نه metaworld. `smoke` اولین چیزی است که باید بزنید.

---

## مراجع

- Wołczyk و همکاران، **Continual World**, NeurIPS 2021 (arXiv:2105.10919)
- Yu و همکاران، **Meta-World**, CoRL 2019 (arXiv:1910.10897)
- Gaya و همکاران، **CSP**, ICLR 2023 (arXiv:2211.10445)
- Hansen و همکاران، **TD-MPC**, ICML 2022
- Hu و همکاران، **CKA-RL**, NeurIPS 2025 (arXiv:2510.19314)
- Bagatella و همکاران، **TD-JEPA**, arXiv:2510.00739
