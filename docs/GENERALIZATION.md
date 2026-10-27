# 泛化实验（g 系列）

## 问题

论文的基准只测训练动作的覆盖率（Appendix F.2 原文明确说明）。g 系列要回答的是：

1. 在物理可行的参考动作上，泛化受限于见过的**数据量**，还是见过的**动作类别**？
2. held-out 上的失败来自**世界模型**（没见过这些状态），还是来自**策略**（目标编码器没泛化）？

## 设计：一次训练，三个评测集

| 集合 | 内容 | 用途 |
|---|---|---|
| `train_eval` | 训练集里随机 2000 条 | 训练覆盖率基线 |
| `heldout_in` | 剩余类别里随机 5%，镜像对（`_M`）不拆开 | 同类别泛化 |
| `heldout_cat` | 整个留出的若干类别 | 跨类别泛化 |

- 数据：BONES-SEED，先做名字过滤和可行性过滤（`filter_bones_seed.py`），held-out 集因此不含物理上不可行的参考动作。SuperMimicKit 的 LAFAN1 val37 就是反例：37 条里有 23 条是障碍物和被推类动作，混合了"类别外"和"平地上不可行"两种失败。
- 观测：`--env.anchor-obs ori`（6 维相对朝向，IMU 可得，不需要状态估计）。
- 训练集下采样到固定时长（`--train-hours`），作为后续 scaling 曲线的第一个点。
- checkpoint 只按 `train_eval` 选，不看 held-out。

## 步骤

需要先在 Hugging Face 上接受 BONES-SEED 许可，并下载 `metadata/seed_metadata_v004.csv`。

```bash
# 1. 看元数据有哪些列、类别有多大，选一个类别列和要留出的类别
uv run python scripts/data/make_split.py --manifest data/npz/bones_seed/manifest_filtered.csv \
  --metadata data/raw/bones_seed/metadata/seed_metadata_v004.csv --inspect-only

# 2. 划分（不给 --heldout-categories 时按 --seed 随机抽 --num-heldout-categories 个中等大小的类别）
uv run python scripts/data/make_split.py --manifest data/npz/bones_seed/manifest_filtered.csv \
  --metadata data/raw/bones_seed/metadata/seed_metadata_v004.csv --category-col <列名> \
  --heldout-categories <类别1> <类别2> ... --train-hours 30 --out data/splits/g1

# 3. 训练库与训练
uv run python scripts/data/build_library.py --npz-dir data/npz/bones_seed \
  --manifest data/splits/g1/train_manifest.csv --out data/library/g1_train.pt
env -u PYTHONPATH uv run sm-train --run-name g1 --data.library data/library/g1_train.pt \
  --env.anchor-obs ori --env.num-envs 4096 --policy-trainer.batch-size 4096 --schedule.max-hours 4
# batch 4096 = paper default (fits an 80 GB+ card); e5/e8 used 2048 only because of the 5090's 32 GB

# 4. 三个集合的 benchmark（含去漂移 MPJPE）
for s in train_eval heldout_in heldout_cat; do
  env -u PYTHONPATH uv run python scripts/eval_benchmark.py --ckpt logs/g1/ckpt_latest.pt \
    --npz-dir data/npz/bones_seed --clip-list data/splits/g1/$s.txt --out logs/g1/benchmark_$s.json
done

# 5. 世界模型在三个集合上的 8 步误差（训练分布协议）
for s in train_eval heldout_in heldout_cat; do
  env -u PYTHONPATH uv run python scripts/eval_wm_heldout.py --ckpt logs/g1/ckpt_latest.pt \
    --npz-dir data/npz/bones_seed --clip-list data/splits/g1/$s.txt \
    --categories data/splits/g1/clip_categories.csv --out logs/g1/wm_$s.json
done

# 6. 汇总
uv run python scripts/report_split.py --categories data/splits/g1/clip_categories.csv \
  --bench train_eval=logs/g1/benchmark_train_eval.json heldout_in=logs/g1/benchmark_heldout_in.json heldout_cat=logs/g1/benchmark_heldout_cat.json \
  --wm train_eval=logs/g1/wm_train_eval.json heldout_in=logs/g1/wm_heldout_in.json heldout_cat=logs/g1/wm_heldout_cat.json \
  --out logs/g1/report.json
```

## 指标

- 失败率与归一化回报：论文 Appendix F 协议，不变。
- `mpjpe_aligned`：把参考动作按锚点（torso）的 xy 和 yaw 对齐到机器人后，14 个跟踪刚体的平均位置误差，单位米。对齐方式与刚体位置奖励相同（`update_relative_body_poses`），高度误差计入，全局漂移不计入。终止那一步不计（状态已被重置）。
- 世界模型误差：与训练日志 `wm/online_nse_8step` 同一个归一化指标，在策略于该集合上实际访问的状态上计算。三个集合必须用同一个 `--protocol`。

## 结果怎么读

| 结果 | 含义 | 下一步 |
|---|---|---|
| `heldout_in` ≈ `train_eval`，`heldout_cat` 偏低，世界模型误差不变 | 瓶颈在策略/目标编码器，类别覆盖是关键 | 固定时长，比较类别多样性和数据量 |
| `heldout_in` ≈ `train_eval`，`heldout_cat` 偏低，世界模型误差在 `heldout_cat` 上升 | 世界模型没见过这些状态 | 只给世界模型补其他类别的数据 |
| `heldout_in` 明显低于 `train_eval` | 同类别也在记忆 | 先查数据量和过拟合 |
| `heldout_cat` ≈ `train_eval` | 泛化强 | 直接用更少数据画 scaling 曲线 |

## 反事实鲁棒裕度

问题：此刻的状态或动作稍有不同，整段动作还能跟住吗？世界模型的预测稍有偏差，会不会让策略失败？

做法（`scripts/eval_margins.py`）：同一批环境从同一个 clip 的首帧同步播放（确定性协议），每个环境只在分配给它的时刻 t 被扰动一次，之后同一策略在仿真器里继续跑；`--window`（默认 100 步 = 2 s）内触发训练终止条件即为失败。扰动幅度用世界模型的归一化误差尺度（`StateSpec.error_vector`）来度量，所以扰动大小和模型误差大小可以直接比较。

- **裕度 r(t)**：在幅度网格上导致失败的最小幅度；多个方向时取最小值（最坏方向）和中位数。
- **模型误差 e(t)**：世界模型从 t−k 出发、用实际动作开环预测到 t 的误差范数（k 默认 8，与训练日志的指标一致）。
- `--kind state`：随机方向。`e(t) ≥ r(t)` 的时刻比例 = 模型的典型误差大小足以让策略失败的时刻。
- `--kind wm`：方向和大小直接取模型在该状态的真实误差，按倍数 λ 缩放；λ < 1 时就失败，说明模型自己的误差就会让策略失败。
- 默认只扰动速度（`--components vel`），因为位置扰动会把脚压进地面，接触求解会额外给一个冲击；`--components all` 包括位置和姿态。
- 扰动在第 t 步开始时写入仿真器，策略从 t+1 起才观测到（相当于一次推力），不改动观测历史。

两项对照必须约等于 0，否则结果不可信：`max_determinism_drift`（未分叉的环境与对照环境 env 0 的最大状态差）和 `zero_scale_failure_rate`（幅度 0 的分支只把原状态写回）。`nonmonotonic_rate` 是"小幅度失败、大幅度反而成功"的比例，来自接触的非光滑性，比例高时裕度应看成粗略估计。

怎么读：

| 现象 | 含义 |
|---|---|
| heldout_cat 的裕度明显小于 train_eval | 策略在新动作上"活着但很脆"，只看失败率看不出来 |
| heldout_cat 上 e(t) ≥ r(t) 的比例明显高于 train_eval | 世界模型误差在新动作上足以让策略失败；再分清是 e 变大（模型没见过）还是 r 变小（策略更脆） |
| 失败 clip 在失败前有一段裕度很低的区间 | 失败可以定位到具体时刻（单脚支撑、落地等） |

## 未验证

- 这些工具是在没有 GPU 和 mjlab 的机器上写的：`test_metrics.py`、`test_make_split.py`、`test_anchor_obs.py`、`test_perturb.py`、`test_margins.py` 在本地用四元数替身和替身环境跑通；`eval_wm_heldout.py`、`eval_margins.py` 在真实仿真器上的分叉（依赖不同环境同样输入结果完全相同）、benchmark 里的 MPJPE 接线、`ori` 模式的仿真端与想象端对拍（`test_anchor_relative_obs_matches_env[ori]`）都还没在真实环境里跑过。第一次在 GPU 机器上先跑 `pytest tests`。
- BONES-SEED 元数据的列名未知，`make_split.py` 的文件名推导类别（`high_jump_R_001__A450` → `high_jump`）只在两个样例名上核对过。
