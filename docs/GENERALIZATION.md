# 泛化实验（g 系列）

## 问题

主问题（第"反事实鲁棒裕度"一节）：**可微世界模型知不知道策略在哪里脆弱？**SuperMimic 的策略只通过世界模型的梯度学习，所以同一个模型可以给出最坏扰动方向和它预测的失败幅度。用仿真器检验：

1. 模型梯度给出的方向，在仿真器里是否比随机方向更容易让策略失败？
2. 模型预测的失败幅度与仿真器是否一致，漏报多少？
3. 在整类留出的新动作上，这两点是否退化？退化来自策略更脆，还是模型误差变大？

背景问题（本节以下的设计和"结果怎么读"）：论文的基准只测训练动作的覆盖率（Appendix F.2 原文明确说明）。

1. 在物理可行的参考动作上，泛化受限于见过的**数据量**，还是见过的**动作类别**？
2. held-out 上的失败来自**世界模型**（没见过这些状态），还是来自**策略**（目标编码器没泛化）？

相关工作：[sl221/supermimic-scaling](https://github.com/sl221/supermimic-scaling) 已经做了 1–254 小时 BONES-SEED 的数据量与 held-out 泛化（按动作名、按表演者留出，外加 LAFAN1），包括世界模型在 held-out 上的误差。这里按类别留出，可以与它对照；扰动下的鲁棒裕度和世界模型能否预测失败，目前没有人做过。相邻的论文：[WorldFailBench](https://arxiv.org/abs/2610.33565)（视频世界模型能否复现策略的失败边界）、[DreamTest](https://arxiv.org/abs/2610.04494)（用世界模型想象来搜索强化学习策略的失败工况）。

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

做法（`scripts/eval_margins.py`）：同一批环境从同一个 clip 的首帧同步播放（确定性协议）；分叉之前，每一步都把对照环境 env 0 的状态（qpos、qvel、求解器热启动）复制给尚未分叉的环境，所以分叉从名义状态加扰动开始。每个环境只在分配给它的时刻 t 被扰动一次，之后同一策略在仿真器里继续跑；`--window`（默认 100 步 = 2 s）内触发训练终止条件即为失败。扰动幅度用世界模型的归一化误差尺度（`StateSpec.error_vector`）来度量，所以扰动大小和模型误差大小可以直接比较。

- **裕度 r(t)**：在幅度网格上导致失败的最小幅度；多个方向时取最小值（最坏方向）和中位数。
- **模型误差 e(t)**：世界模型从 t−k 出发、用实际动作开环预测到 t 的误差范数（k 默认 8，与训练日志的指标一致）。
- `--kind state`：随机方向。`e(t) ≥ r(t)` 的时刻比例 = 模型的典型误差大小足以让策略失败的时刻。
- `--kind wm`：方向和大小直接取模型在该状态的真实误差，按倍数 λ 缩放；λ < 1 时就失败，说明模型自己的误差就会让策略失败。
- 默认只扰动速度（`--components vel`），因为位置扰动会把脚压进地面，接触求解会额外给一个冲击；`--components all` 包括位置和姿态。
- 扰动在第 t 步开始时写入仿真器，策略从 t+1 起才观测到（相当于一次推力），不改动观测历史。
- `--kind adv`：模型的最坏方向。从名义状态和名义观测历史出发，在世界模型里想象 `--adv-horizon` 步（默认取训练的 H=32），目标是训练用的回报（不含 critic），对起始状态扰动求梯度，取下降最快的单位方向（限制在 `--components` 子空间）。仿真器里沿这个方向按幅度网格分叉，得到仿真裕度。同时记录两个模型侧的量：梯度范数，以及模型自己预测的裕度（沿同一方向、想象 rollout 结束时存活概率低于 0.5 的最小幅度）。为了公平比较，仿真裕度也按模型的时长重新截取一份（`sim_margin_model_horizon`）。想象 rollout 和仿真分叉一样，策略在下一步才看到扰动。
- `scripts/compare_margins.py`：同一批 clip、同一批时刻上，`adv` 的仿真裕度与随机方向中最坏的那个比较。

两项对照必须约等于 0，否则结果不可信：`max_determinism_drift`（复制之前，未分叉环境一步内与 env 0 的最大偏差，归一化误差范数，和扰动幅度同一单位，应远小于最小幅度 0.25，门限 0.01）和 `zero_scale_failure_rate`（幅度 0 的分支只把原状态写回）。

为什么要每步复制：H200 上第一次跑（LAFAN1，240 s 的 clip）时，没有复制，同样输入的并行环境在几千步后相差 52（原始状态单位）。浮点舍入随环境原点不同，长时间跟踪对这种微小差异是混沌的：同一个策略、同一个 clip，`run2_subject1` 在 `diag_drift.py` 里跑完了，在裕度脚本的 env 0 里却在第 11066 步失败。所以长 clip 上"会不会失败"本身对 1e-7 级的差异敏感，只有短窗口（`--window`）内的失败才有意义。`nonmonotonic_rate` 是"小幅度失败、大幅度反而成功"的比例，来自接触的非光滑性，比例高时裕度应看成粗略估计。

怎么读：

| 现象 | 含义 |
|---|---|
| heldout_cat 的裕度明显小于 train_eval | 策略在新动作上"活着但很脆"，只看失败率看不出来 |
| heldout_cat 上 e(t) ≥ r(t) 的比例明显高于 train_eval | 世界模型误差在新动作上足以让策略失败；再分清是 e 变大（模型没见过）还是 r 变小（策略更脆） |
| 失败 clip 在失败前有一段裕度很低的区间 | 失败可以定位到具体时刻（单脚支撑、落地等） |
| adv 的裕度明显小于随机方向（log2 比值中位数 ≤ −1） | 模型知道策略的弱点方向：模型可以用来做压力测试，也可以拿来在训练中针对性加扰动 |
| adv 与随机方向差不多 | 模型的一阶梯度在这些状态上没有信息；先看是否集中在接触时刻（落地、离地），那里模型的梯度最可能不准 |
| 模型裕度 `missed` 高 | 模型低估失败：想象里能救回来、仿真里会倒。用模型做安全评估会过于乐观 |
| `heldout_cat` 上 `missed` 或 adv/随机比值变差 | 模型的"自知之明"不能推广到新动作，这是新动作上需要真实仿真验证的直接理由 |

LAFAN1 上的第一次结果（随机速度方向）：随机方向的裕度中位数顶到当时网格上限 16，远大于模型 8 步误差的范数（约 1.4）。所以随机方向下"模型误差 ≥ 裕度"几乎不会出现，有区分度的是模型方向和最坏方向。

## 未验证

- 这些工具是在没有 GPU 和 mjlab 的机器上写的：`test_metrics.py`、`test_make_split.py`、`test_anchor_obs.py`、`test_perturb.py`、`test_margins.py` 在本地用四元数替身和替身环境跑通；`eval_wm_heldout.py`、`eval_margins.py` 在真实仿真器上的分叉（依赖不同环境同样输入结果完全相同）、benchmark 里的 MPJPE 接线、`ori` 模式的仿真端与想象端对拍（`test_anchor_relative_obs_matches_env[ori]`）都还没在真实环境里跑过。第一次在 GPU 机器上先跑 `pytest tests`。
- BONES-SEED 元数据的列名未知，`make_split.py` 的文件名推导类别（`high_jump_R_001__A450` → `high_jump`）只在两个样例名上核对过。
