# Runbook：泛化实验（g 系列）

本分支在 [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced) 之上做 held-out 泛化实验。实验设计、指标定义和结果怎么读，见 [docs/GENERALIZATION.md](docs/GENERALIZATION.md)；这份文件只写**按什么顺序跑、每步跑完检查什么**。

## 要回答的问题

主问题：**可微世界模型知不知道策略在哪里脆弱？**SuperMimic 的策略只从世界模型的梯度学习，同一个模型也能给出"最坏的扰动方向"和"它认为策略会在多大扰动下失败"。这里在仿真器里检验这两点，并比较训练动作和整类留出的新动作。

1. **模型的最坏方向是不是真的最坏**：沿模型梯度给出的方向扰动（`--kind adv`），仿真器里的失败幅度是否明显小于随机方向（`--kind state`）？
2. **模型预测的裕度准不准**：模型自己想象出的失败幅度，与仿真器在同一时长内的失败幅度是否一致？漏报（仿真会失败、模型认为不会）有多少？
3. **在新动作上是否变差**：在 `heldout_cat` 上，上面两点是否退化？是策略更脆（裕度变小），还是模型误差变大（`--kind wm` 的 λ 变小）？

背景实验（不是主贡献）：三个集合上的回报和世界模型误差，即训练集抽样（`train_eval`）、同类别 held-out（`heldout_in`）和整类留出（`heldout_cat`）。数据量与 held-out 泛化的关系已经有人做过（[sl221/supermimic-scaling](https://github.com/sl221/supermimic-scaling)，按动作名和表演者留出），这里按类别留出，结果可以和它对照。观测用 `--env.anchor-obs ori`（6 维相对朝向，IMU 可得）。

## 本分支相对上游加了什么

| 改动 | 文件 |
|---|---|
| `--env.anchor-obs` 改为三档：`none` / `ori`（6 维）/ `pos_ori`（9 维，需要状态估计）；旧 checkpoint 的 True/False 仍可读 | `src/supermimic/anchor_obs.py` 等 |
| benchmark 增加去漂移 MPJPE（按锚点 xy + yaw 对齐后的刚体位置误差） | `src/supermimic/eval/metrics.py`、`benchmark.py` |
| 类别划分：整类留出 + 同类别留出，镜像对不拆开，可固定训练时长 | `scripts/data/make_split.py` |
| 世界模型在策略实际访问状态上的 k 步误差，按类别和 clip 汇总 | `scripts/eval_wm_heldout.py` |
| 三个集合的结果汇成一张表 | `scripts/report_split.py` |
| 反事实鲁棒裕度：沿动作在各时刻把状态（或动作）改一点，找出导致失败的最小幅度，并与世界模型自身的预测误差比较 | `scripts/eval_margins.py`、`src/supermimic/eval/margins.py`、`perturb.py` |
| 模型最坏方向裕度（`--kind adv`）：想象回报对起始状态扰动的梯度方向；同时记录模型自己预测的裕度，与仿真器同时长的裕度比较 | `src/supermimic/eval/adversarial.py`、`scripts/eval_margins.py` |
| 随机方向与模型最坏方向的逐时刻对比、模型裕度的校准 | `scripts/compare_margins.py` |

**注意**：这些代码是在没有 GPU、没有 mjlab 的机器上写的。`tests/test_metrics.py`、`test_make_split.py`、`test_anchor_obs.py`、`test_perturb.py`、`test_margins.py`（分叉逻辑，用替身环境）已在本地跑通；仿真端的部分（MPJPE 接线、`eval_wm_heldout.py`、`eval_margins.py` 在真实仿真器上的分叉、`ori` 模式的仿真端与想象端对拍）**还没在真实环境里跑过**，所以第 1、2 步是必须先过的关卡。

## 硬件与时间

- 单卡 H200（141 GB）。与 H100 同代、算力相近，论文的 H100 耗时可直接当参照；显存足够，主训练用论文默认的 batch 4096。
- 总计约 8–9 小时，BONES-SEED 下载占大头。

## 一键脚本

`scripts/run_experiments.sh` 把下面的步骤串起来，每一阶段结束都有自动关卡，失败就停下并给出原因。建议在 tmux 里跑。已完成的输出会被跳过，所以中断后重跑同一条命令就能接着跑；训练中断会自动续训。

```bash
bash scripts/run_experiments.sh validate               # 第 1–2 步：测试、LAFAN1 短训练、关卡检查（约 2 小时）
bash scripts/run_experiments.sh prepare                # 第 3 步：BONES-SEED 下载、转换、过滤，最后打印元数据（约 1 小时）
# 第 4 步手动：选类别列和留出类别，运行 make_split.py --out data/splits/g1
bash scripts/run_experiments.sh main data/splits/g1    # 第 5–6 步：训练、评测、汇总、裕度（约 7 小时）
```

可调的环境变量：`RUN`（默认 g1）、`LAFAN_HOURS`（1.5）、`MAIN_HOURS`（4）、`NUM_ENVS`（4096）、`BATCH`（4096）、`WORKERS`（20）。日志和结果都在 `logs/<run>/`；关卡由 `scripts/check_run.py` 判定，阈值见下面各步的"检查"。

脚本的串联逻辑已在本机用假的 `uv` 演练过（文件名、跳过、续训、关卡失败停下）；真正的命令要到服务器上才第一次运行。

### 只有 L4 时：主问题改在 LAFAN1 上做

没有 H200 时，主问题在 LAFAN1 上用整类留出来做：留出 dance、fight、fightAndSports（13 条，0.76 h），训练其余 25 条（1.57 h）。BONES-SEED 放不下（单是压缩包就 23.5 GB），主训练也要几十小时。L4 的设置：

- 批 2048。批 4096 要约 24 GB 显存，放不下；批 2048 用 12.4 GB，稳态 1.55 it/s（5090 是 4.3 it/s）。
- 训练 5 小时，约 2.8 万次迭代。
- 这台机器没有可用的 EGL，运行前先 `source` 一个软件 EGL（Mesa），例如 Uni-SuperMimic 的 `tools/softgl.sh`。

```bash
uv run python scripts/data/make_split.py --manifest data/npz/lafan1/manifest.csv \
  --heldout-categories dance fight fightAndSports --out data/splits/lafan_cat
LAFAN_HOURS=5 BATCH=2048 WORKERS=8 bash scripts/run_experiments.sh validate
RUN=lafan_cat NPZ=data/npz/lafan1 MAIN_HOURS=5 BATCH=2048 WORKERS=8 bash scripts/run_experiments.sh main data/splits/lafan_cat
```

`heldout_in` 只有 2 条，只能看个大概。

## 待办

### 0. 环境

- [ ] clone 本仓库并切到本分支，`uv sync --group dev`

```bash
uv sync --group dev
```

### 1. LAFAN1 + 测试（约 15 分钟）

- [ ] 下载 LAFAN1 的 G1 重定向数据（不需要申请）
- [ ] 转成 50 Hz npz
- [ ] 跑全部测试

```bash
uv run hf download lvhaidong/LAFAN1_Retargeting_Dataset --repo-type dataset --include "g1/*" --local-dir data/raw/lafan1
uv run python scripts/data/csv_to_npz_batch.py --input-dir data/raw/lafan1/g1 --output-dir data/npz/lafan1 --input-fps 30
env -u PYTHONPATH uv run pytest tests -q
```

较老的 `huggingface-hub` 用 `huggingface-cli download` 代替 `hf download`。

**检查**：全部通过，尤其是 `test_anchor_relative_obs_matches_env[ori]`（新观测在仿真端和想象端完全一致）。失败就停下排查，不要进入下一步。

### 2. LAFAN1 短训练，端到端验证新代码（约 1.5 小时）

可以同时开始第 3 步的下载。

- [ ] 用 `ori` 观测训练 1.5 小时
- [ ] benchmark
- [ ] 漂移诊断
- [ ] 世界模型误差脚本
- [ ] 裕度脚本（只跑 4 个 clip，检查确定性和 0 幅度对照）
- [ ] 模型最坏方向裕度 `--kind adv`（同样 4 个 clip，检查代码能跑通、两项对照，并看 `compare_margins.py` 的输出）

```bash
env -u PYTHONPATH uv run sm-train --run-name lafan_ori --data.npz-dir data/npz/lafan1 --env.anchor-obs ori --schedule.max-hours 1.5
env -u PYTHONPATH uv run python scripts/eval_benchmark.py --ckpt logs/lafan_ori/ckpt_latest.pt --npz-dir data/npz/lafan1
env -u PYTHONPATH uv run python scripts/diag_drift.py --ckpt logs/lafan_ori/ckpt_latest.pt --npz-dir data/npz/lafan1
env -u PYTHONPATH uv run python scripts/eval_wm_heldout.py --ckpt logs/lafan_ori/ckpt_latest.pt --npz-dir data/npz/lafan1
env -u PYTHONPATH uv run python scripts/eval_margins.py --ckpt logs/lafan_ori/ckpt_latest.pt --npz-dir data/npz/lafan1 --max-clips 4 --max-times 8
```

**检查**：

| 指标 | 预期 | 不对时说明什么 |
|---|---|---|
| 归一化回报 | 约 0.8（论文 H100 1.2 h 到 0.82；supermimic_vibed 同类观测 0.829） | 约 0.56 说明朝向观测没进到策略里 |
| 去漂移 MPJPE | 几厘米 | 几十厘米说明指标接错了 |
| 平均 yaw 误差（diag_drift） | 几度 | 无锚点观测时是 82° |
| 世界模型 8 步误差 | 接近 TensorBoard 里的 `wm/online_nse_8step` | 差很多说明新脚本有 bug |
| 裕度脚本的 `drift_frac_over_tol` / `drift_event_rate` | 一步漂移 > 0.01 的环境·步 ≤ 1%；> 0.25（最小幅度）的 ≤ 1e-4 | 每步复制后仍然分得很开，分叉之间的差异就不只来自扰动，裕度不可信。`max_determinism_drift` 只作参考：GPU 上少数病态接触时刻（约每万步一次）会让相同状态和动作解出不同结果，最大值总会到 1–10 |
| `retest_margins.py`（打乱分支到环境的分配后重跑） | 同一时刻的裕度相差不超过一档的比例 ≥ 90%；adv 与随机方向的结论相同，中位 log2 比值差 ≤ 0.5 | 仿真噪声在改变结论 |
| 裕度脚本的 `zero_scale_failure_rate` | 约 0 | 只是把原状态写回仿真器就会导致失败，说明写入本身在扰动仿真 |

### 3. 准备 BONES-SEED（约 1 小时，主要是下载）

数据需要在 Hugging Face 上接受 [BONES-SEED 许可](https://huggingface.co/datasets/bones-studio/seed)（学术/非商业用途）。

- [ ] `uv run hf auth login` 登录
- [ ] 下载 `g1.tar.gz`（23.5 GB）和元数据
- [ ] 解压、转换（120 fps，格式自动识别）
- [ ] 名字过滤 + 可行性过滤

```bash
uv run hf download bones-studio/seed g1.tar.gz metadata/seed_metadata_v004.csv --repo-type dataset --local-dir data/raw/bones_seed
mkdir -p data/raw/bones_seed/g1_csv && tar -xzf data/raw/bones_seed/g1.tar.gz -C data/raw/bones_seed/g1_csv
uv run python scripts/data/csv_to_npz_batch.py --input-dir data/raw/bones_seed/g1_csv --output-dir data/npz/bones_seed --input-fps 120 --workers 20
uv run python scripts/data/filter_bones_seed.py --npz-dir data/npz/bones_seed --workers 20
```

解压到单独的 `g1_csv` 目录：转换脚本会递归扫描 `--input-dir` 下所有 CSV，若直接扫 `data/raw/bones_seed`，元数据 CSV 也会被当成动作文件。

**检查**：过滤后约 129,491 条（与上游 e8 一致）。

### 4. 选留出类别、划分数据

- [ ] 看元数据的列和类别大小
- [ ] 选类别列和约 5 个留出类别：难度要拉开，各选一个中等大小的走路类、舞蹈类、倒地起身类、跳跃类、格斗或运动类
- [ ] 划分，训练集固定 30 小时

```bash
uv run python scripts/data/make_split.py --manifest data/npz/bones_seed/manifest_filtered.csv \
  --metadata data/raw/bones_seed/metadata/seed_metadata_v004.csv --inspect-only
uv run python scripts/data/make_split.py --manifest data/npz/bones_seed/manifest_filtered.csv \
  --metadata data/raw/bones_seed/metadata/seed_metadata_v004.csv --category-col <列名> \
  --heldout-categories <类别1> <类别2> ... --train-hours 30 --out data/splits/g1
```

**检查**：`data/splits/g1/split.json` 里 `heldout_in_category_seen_in_train` 接近 1.0，训练集接近 30 小时。

### 5. 主训练、评测、汇总（训练约 4 小时，评测 1 小时以内）

- [ ] 打包训练库并训练
- [ ] 三个集合的 benchmark
- [ ] 三个集合的世界模型误差
- [ ] 汇总成一张表

命令见 [docs/GENERALIZATION.md](docs/GENERALIZATION.md) 的第 3–6 步。H200 显存足够，三个集合的评测可以同时跑。checkpoint 只按 `train_eval` 选，不看 held-out。

**检查**：对照 GENERALIZATION.md 里"结果怎么读"的四种情况，判断属于哪一种，再决定下一步（类别多样性 vs 数据量，或只给世界模型补数据）。

### 6. 反事实鲁棒裕度（主实验，每组约 20–30 分钟，可并行）

- [ ] `train_eval` 和 `heldout_cat` 上各跑 `--kind state`（随机方向）、`--kind wm`（按世界模型自身误差）和 `--kind adv`（世界模型的最坏方向）
- [ ] 用 `compare_margins.py` 对比随机方向和模型最坏方向

```bash
for s in train_eval heldout_cat; do for k in state wm adv; do
  env -u PYTHONPATH uv run python scripts/eval_margins.py --ckpt logs/g1/ckpt_latest.pt --npz-dir data/npz/bones_seed \
    --clip-list data/splits/g1/$s.txt --categories data/splits/g1/clip_categories.csv --kind $k --out logs/g1/margins_${s}_$k.json
done; done
for s in train_eval heldout_cat; do
  env -u PYTHONPATH uv run python scripts/compare_margins.py --random logs/g1/margins_${s}_state.json --adv logs/g1/margins_${s}_adv.json --out logs/g1/compare_margins_$s.json
done
```

**检查**：先看 `check_run.py margins` 的漂移比例、`zero_scale_failure_rate`（≤ 0.01）和 `retest_margins.txt`（heldout_cat 4 条 clip 的重测），否则结果不可信。然后看三件事：

1. `compare_margins` 的 `median_log2_adv_over_random`：负数表示模型的方向用更小的扰动就让策略失败，−1 是一半大小。
2. adv 文件的 `model_vs_sim`：`agree`（模型裕度与仿真在 2 倍以内，或都不失败）、`missed`（仿真失败、模型认为不会）。
3. 两个集合对比：`heldout_cat` 上上面两项是否变差，`--kind state` 的 `median_margin_min` 是否变小，`--kind wm` 的 λ 是否变小。怎么读见 GENERALIZATION.md 的"反事实鲁棒裕度"一节。

## 结果记录

| 日期 | run | train_eval 失败率 / 回报 | heldout_in | heldout_cat | 世界模型 8 步误差（三集） | 随机方向裕度中位数（train_eval / heldout_cat） | adv / 随机 log2 中位数 | 模型裕度 agree / missed | 结论 |
|---|---|---|---|---|---|---|---|---|---|
| 2026-10-28 | lafan_ori（第 2 步） | 8/40，0.741，MPJPE 5.4 cm | – | – | 1 步 0.011，8 步 0.028 | 16（旧网格上限，漂移未修，无效） | | | 局部跟踪与 CuboiLeo e3 同时长一致，差距来自不观测根部位置 |
| | g1 | | | | | | | | |
