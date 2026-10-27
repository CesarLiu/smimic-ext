# Runbook：泛化实验（g 系列）

本分支在 [CuboiLeo/SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced) 之上做 held-out 泛化实验。实验设计、指标定义和结果怎么读，见 [docs/GENERALIZATION.md](docs/GENERALIZATION.md)；这份文件只写**按什么顺序跑、每步跑完检查什么**。

## 要回答的问题

论文的基准只测训练动作的覆盖率（Appendix F.2）。这里要回答：

1. 在物理可行的动作上，泛化受限于**数据量**还是**见过的动作类别**？
2. held-out 上的失败来自**世界模型**还是**策略**？

做法：在 BONES-SEED 上训练一次，评测三个集合：训练集抽样（`train_eval`）、同类别 held-out（`heldout_in`）、整类留出（`heldout_cat`）。观测用 `--env.anchor-obs ori`（6 维相对朝向，IMU 可得）。

## 本分支相对上游加了什么

| 改动 | 文件 |
|---|---|
| `--env.anchor-obs` 改为三档：`none` / `ori`（6 维）/ `pos_ori`（9 维，需要状态估计）；旧 checkpoint 的 True/False 仍可读 | `src/supermimic/anchor_obs.py` 等 |
| benchmark 增加去漂移 MPJPE（按锚点 xy + yaw 对齐后的刚体位置误差） | `src/supermimic/eval/metrics.py`、`benchmark.py` |
| 类别划分：整类留出 + 同类别留出，镜像对不拆开，可固定训练时长 | `scripts/data/make_split.py` |
| 世界模型在策略实际访问状态上的 k 步误差，按类别和 clip 汇总 | `scripts/eval_wm_heldout.py` |
| 三个集合的结果汇成一张表 | `scripts/report_split.py` |
| 反事实鲁棒裕度：沿动作在各时刻把状态（或动作）改一点，找出导致失败的最小幅度，并与世界模型自身的预测误差比较 | `scripts/eval_margins.py`、`src/supermimic/eval/margins.py`、`perturb.py` |

**注意**：这些代码是在没有 GPU、没有 mjlab 的机器上写的。`tests/test_metrics.py`、`test_make_split.py`、`test_anchor_obs.py`、`test_perturb.py`、`test_margins.py`（分叉逻辑，用替身环境）已在本地跑通；仿真端的部分（MPJPE 接线、`eval_wm_heldout.py`、`eval_margins.py` 在真实仿真器上的分叉、`ori` 模式的仿真端与想象端对拍）**还没在真实环境里跑过**，所以第 1、2 步是必须先过的关卡。

## 硬件与时间

- 单卡 H200（141 GB）。与 H100 同代、算力相近，论文的 H100 耗时可直接当参照；显存足够，主训练用论文默认的 batch 4096。
- 总计约 8–9 小时，BONES-SEED 下载占大头。

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
| 裕度脚本的 `max_determinism_drift` | 约 0（≤ 1e-4） | 不同环境同样输入结果不同，分叉之间的差异就不只来自扰动，裕度不可信 |
| 裕度脚本的 `zero_scale_failure_rate` | 约 0 | 只是把原状态写回仿真器就会导致失败，说明写入本身在扰动仿真 |

### 3. 准备 BONES-SEED（约 1 小时，主要是下载）

数据需要在 Hugging Face 上接受 [BONES-SEED 许可](https://huggingface.co/datasets/bones-studio/seed)（学术/非商业用途）。

- [ ] `uv run hf auth login` 登录
- [ ] 下载 `g1.tar.gz`（23.5 GB）和元数据
- [ ] 解压、转换（120 fps，格式自动识别）
- [ ] 名字过滤 + 可行性过滤

```bash
uv run hf download bones-studio/seed g1.tar.gz metadata/seed_metadata_v004.csv --repo-type dataset --local-dir data/raw/bones_seed
tar -xzf data/raw/bones_seed/g1.tar.gz -C data/raw/bones_seed
uv run python scripts/data/csv_to_npz_batch.py --input-dir data/raw/bones_seed --output-dir data/npz/bones_seed --input-fps 120 --workers 20
uv run python scripts/data/filter_bones_seed.py --npz-dir data/npz/bones_seed --workers 20
```

压缩包解压出的目录名未确认；`--input-dir` 找不到文件时，改成实际放 CSV 的目录。

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

### 6. 反事实鲁棒裕度（每组约 20–30 分钟，可并行）

- [ ] `train_eval` 和 `heldout_cat` 上各跑 `--kind state`（随机方向）和 `--kind wm`（按世界模型自身误差）

```bash
for s in train_eval heldout_cat; do for k in state wm; do
  env -u PYTHONPATH uv run python scripts/eval_margins.py --ckpt logs/g1/ckpt_latest.pt --npz-dir data/npz/bones_seed \
    --clip-list data/splits/g1/$s.txt --categories data/splits/g1/clip_categories.csv --kind $k --out logs/g1/margins_${s}_$k.json
done; done
```

**检查**：先看 `max_determinism_drift` 和 `zero_scale_failure_rate` 都约等于 0，否则结果不可信。然后对比两个集合的 `median_margin_min`（裕度是否变小）和 `frac_model_error_exceeds_margin`（世界模型误差是否足以让策略失败）。怎么读见 GENERALIZATION.md 的"反事实鲁棒裕度"一节。

## 结果记录

| 日期 | run | train_eval 失败率 / 回报 | heldout_in | heldout_cat | 世界模型 8 步误差（三集） | 裕度中位数 / 模型误差≥裕度比例（train_eval vs heldout_cat） | 结论 |
|---|---|---|---|---|---|---|---|
| | lafan_ori（第 2 步） | | – | – | | – | |
| | g1 | | | | | | |
