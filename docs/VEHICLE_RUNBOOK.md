# Runbook：车辆方向（SuperMimic for Vehicles，E0–E1 与第 2 步）

设计和全部结论见 [`VEHICLE_PROPOSAL.md`](VEHICLE_PROPOSAL.md)：§3.1 仿真器与对拍，§3.8 端到端 demo，§3.9 E1 雅可比检查，§3.10 极限附近的策略学习（第 2 步），§6 实验计划。这份文件只写**按什么顺序跑、每步跑完检查什么、结果怎么读**。

代码位置：
- `src/supermimic/vehicles/`：仿真器、环境、学习端接口、雅可比工具；
- `scripts/vehicle/`：对拍参考生成、吞吐测试、demo、E1 检查、极限研究；
- `docs/results/`：已有运行的 JSON 报告，可用来对照。

## 当前状态

| 阶段 | 内容 | 状态 | 依据 |
|---|---|---|---|
| E0 | 与 CommonRoad 多体模型对拍，误差 < 5% | ✅ 通过，最差 4.08% | §3.1 |
| E0 | 单卡吞吐 ≥ 10⁶ 环境步/秒 | ⏳ 未测，需要 GPU | 第 3 步 |
| demo | 整条 SuperMimic 循环在车辆上学会跟踪 | ✅ 通过：留出参考回报 0.10 → 0.36（加入原始动作约束后；之前为 0.62，见 §3.8） | §3.8，`docs/results/vehicle_demo_e2e.json` |
| E1 | 饱和区雅可比检查（修订后的关卡） | ✅ 大容量世界模型通过（饱和区 8 步梯度余弦 0.94，误差 0.11）；默认规模未通过 | §3.9，`docs/results/vehicle_jacobian_check*.json` |
| 第 2 步 | 策略在附着极限附近学跟踪，大小世界模型对比（无关卡，只测量） | ✅ 已测：三档摩擦都学会（回报约 0.1 → 0.4–0.6）；大模型近极限仍在参考上 57 / 96 / 49%，小模型 47 / 49 / 12%；终止率仍有 33–50% | §3.10，`docs/results/vehicle_limit_study.json` |

## 环境

与仓库其余部分相同（见 README 的 Setup）：Python 3.12，`uv sync --group dev`。车辆部分只依赖 torch，**全部可以在 CPU 上跑**；只有吞吐关卡需要 GPU。

- 下文命令都写成 `env -u PYTHONPATH uv run ...`；不用 uv 时，等价于 `PYTHONPATH=src python ...`。
- 没有显示环境的机器上，如果导入 mujoco 时 EGL 报错，在命令前加 `MUJOCO_GL=disable`（车辆部分不渲染）。
- 只有重新生成 CommonRoad 参考（第 2 步）时才需要 `uv pip install commonroad-vehicle-models`。仓库已带参考数据，平时不用装。

下面的耗时都是在 4 核 CPU 上实测的。

## 待办

### 1. 测试（约 2–4 分钟）

- [ ] 全套测试

```bash
env -u PYTHONPATH uv run pytest tests -q
```

**检查**：CPU 上应为 129 passed，19 skipped（跳过的需要 GPU 或 LAFAN1 数据）。车辆相关的五个文件：

| 文件 | 测什么 |
|---|---|
| `test_vehicle_sim.py` | 轮胎公式与 CommonRoad 逐点相等；14 个工况对拍 ≤ 5%；子步数收敛；ABS/ESC 有效；摩擦缩放；批内各行互不影响 |
| `test_vehicle_env.py` | 回放参考指令能复现参考；终止相对参考判定；指令延迟；ξ 采样；回放缓冲区；重置采样器；按摩擦生成参考；原始动作截断与越界惩罚；仿真器非有限值时终止 |
| `test_vehicle_e2e.py` | 学习端对每条转移的打分与环境一致；状态表示的接口约定；小规模跑通 demo 和极限研究 |
| `test_vehicle_jacobian.py` | `logged_step` 与环境逐位相等；两种雅可比估计在已知系统上精确；小规模跑通检查脚本 |
| `test_*_interface.py`、`test_policy_io.py` | G1 golden 回归：接口化重构没有改变 G1 的数值 |

### 2. E0 对拍（约 10 秒）

- [ ] 与 CommonRoad 多体模型的响应对拍

```bash
env -u PYTHONPATH uv run python -m supermimic.vehicles.validation            # float32，10 子步（GPU 上的设置）
env -u PYTHONPATH uv run python -m supermimic.vehicles.validation --float64  # 精度对照
```

**检查**：最后一行 `worst metric 4.08% (gate 5%)`。逐工况的指标见 §3.1 的表。最大的一项是 `sine_v20_a0.070` 的侧向加速度 NRMSE。

- [ ] （可选）重新生成参考数据：只有改了工况定义（`vehicles/maneuvers.py`）或参数时才需要

```bash
uv pip install commonroad-vehicle-models
env -u PYTHONPATH uv run python scripts/vehicle/make_commonroad_reference.py --out tests/data/commonroad_mb_reference.pt
```

约 15 秒。正弦幅值超过 0.07 rad 时，CommonRoad 多体模型自己会失稳发散，脚本会报 `integration failed`。

### 3. E0 吞吐（GPU，约 5 分钟）

- [ ] 在 GPU 上测吞吐

```bash
env -u PYTHONPATH uv run python scripts/vehicle/bench_sim.py --device cuda --num-envs 4096 16384 65536
env -u PYTHONPATH uv run python scripts/vehicle/bench_sim.py --device cuda --num-envs 4096 16384 65536 --compile
```

**检查**：`env` 一列（完整环境步，含执行器、奖励、回放记录、重置）至少有一个规模达到 1,000,000 steps/s，即为通过。把结果填进 §3.1 末尾和 §6 的 E0 行。CPU 参照值：16384 个环境时 eager 约 1.2×10⁵，compile 约 1.7×10⁵。

**不过关时**：先看 `dynamics` 一列。如果动力学本身就不够快，瓶颈在逐元素算子的启动开销：先确认 `--compile` 是否生效，再考虑把子步循环写成 CUDA graph 或 Warp kernel。如果动力学够快而环境慢，瓶颈在环境的记账逻辑（回放写入、重置）。

### 4. 端到端 demo（CPU 约 8 分钟）

- [ ] 跑 demo

```bash
env -u PYTHONPATH uv run python scripts/vehicle/demo_e2e.py --out logs/vehicle_demo/report.json
# GPU：--device cuda，可加大 --num-envs
```

**检查**：
- 最后打印 `PASSED`，退出码 0；
- 种子 0 下数字应与 `docs/results/vehicle_demo_e2e.json` 相同：留出参考上的归一化回报 0.104 → 0.356，平均坚持步数 86 → 254，世界模型 8 步误差 0.25；
- 成功判据写在脚本的 `check()` 里：回报至少翻倍且至少高 0.15，坚持步数至少翻倍，世界模型误差 < 3。

**怎么读日志**：
- `wm 8-step`：第 100 次迭代起训练世界模型。策略开始学习后（第 300 次迭代），状态分布在变，这个值会先升到 1–3，再降到 0.3 以下，这是正常的；
- `critic`：应保持在 1 以下。如果一路涨到 10² 以上，说明 critic 在发散，检查 `value_clip` 是否生效（§3.8）；
- `imag return`：16 步想象回报，应从约 0.67 升到 0.9 左右。

### 5. E1 雅可比检查

- [ ] 默认规模（约 4 分钟）

```bash
env -u PYTHONPATH uv run python scripts/vehicle/jacobian_check.py --out logs/e1/default.json
```

- [ ] 大容量（约 17 分钟）

```bash
env -u PYTHONPATH uv run python scripts/vehicle/jacobian_check.py --hidden 512 512 --updates 12000 --collect 1500 --out logs/e1/large.json
```

**检查**：
- 关卡（2026-11-04 修订，`jacobian.e1_gate`）：u > 0.9 时 8 步动作梯度余弦中位数 > 0.8，且 8 步误差平均值 < 0.2；
- **大容量配置应打印 `PASSED`**（退出码 0；0.943 / 0.115）；
- **默认配置打印 `FAILED`**（退出码 1；误差 0.551 超标），这是已知结果，说明需要大容量世界模型；
- 种子 0 下数字应与 `docs/results/vehicle_jacobian_check.json`、`vehicle_jacobian_check_large.json` 一致。

可选参数：
- `--save run.pt`：保存世界模型和比较用的雅可比，便于离线分析；
- `--per-bin`：每档的样本数；
- `--noise`：行为策略的噪声；
- `--k`：多步梯度的步数。

**怎么读输出**：

| 输出块 | 含义 | 当前结论 |
|---|---|---|
| `8-step normalised error per utilisation bin` | 世界模型开环误差，按附着利用率 u 分档 | 饱和区误差是线性区的 5–11 倍，但大容量模型饱和区的绝对误差（0.09–0.16）低于默认模型线性区的误差 |
| `one-step Jacobian` 的 `cos`、`cos_*` | 单步雅可比与仿真器有限差分的余弦，整体和按分量分组 | 各档约 0.5–0.6；速度组在饱和区只有 0.17–0.29 |
| `one-step Jacobian` 的 `*_sign_ok`、`*_n` | 关键灵敏度的符号一致率和有效样本数（只统计仿真器值 ≥ 1e-3 的样本） | 转向 → 侧向加速度：线性区 100%，饱和区约 70%；样本少 |
| `8-step gradient w.r.t. the first action` | 8 步后的状态对第一步动作的梯度，即 BPTT 用的量 | 大容量：线性区 0.99，接近极限 0.95，越过极限 0.92；默认规模越过极限只有 0.71 |
| `excluded_nonsmooth` | 两种差分步长结果不一致、被排除的样本数 | 每档 0–4 个 |

**判定**：脚本按修订后的关卡判定。原关卡（饱和区单步余弦 > 0.8、误差比 ≤ 3 倍）两种配置都不过；修订的理由见 §3.9。单步雅可比仍然输出，作为诊断。

### 6. 第 2 步：极限附近的策略学习（CPU 约 23 分钟）

- [ ] 跑极限研究（依次训练小、大两种世界模型）

```bash
env -u PYTHONPATH uv run python scripts/vehicle/limit_study.py --out logs/limit/report.json
# 只跑一种：--models large；改摩擦档：--mu-levels 0.5 0.8 1.0
```

**检查**：
- 退出码 0，最后打印一张表（每档摩擦：回报 / 近极限仍在参考上的比例 / 近极限横向误差）；
- 种子 0 下数字应与 `docs/results/vehicle_limit_study.json` 相同。训练后：小 0.516 / 0.47、0.381 / 0.49、0.405 / 0.12；大 0.590 / 0.57、0.530 / 0.95、0.489 / 0.49；
- 没有通过/不通过的判定，这是测量。

**怎么读日志**：
- `wm 8-step` 在策略开始学习后（第 300 次迭代）会先升到 1–3，再降到 0.1–0.3，与 demo 相同；
- **如果它在后期突然升到 10 以上、`critic` 跟着上千**：这是修正前的发散模式（§3.10）。先检查回放里待执行指令那几维（状态第 18–23 维）是否有界（应 ≤ 3），再检查奖励里 `action_bound` 一项是否生效。

**结果怎么读**：
- “近极限”指参考自身利用率 > 0.8 的步，分母对所有策略相同（`near_limit_steps`）；
- “仍在参考上的比例”是策略在这些步上尚未终止的比例，横向 / 航向 / 侧偏误差只在这些步上平均，所以撑得更久的策略会在更难的段上被计入误差（幸存者偏差）；
- 每档只有 6 条留出参考，终止率的分辨率是 1/6。

### 7. 下一步（E3/E4）

- [x] E1 的关卡按 §3.9 修订（2026-11-04）
- [x] 第 2 步：策略能否在附着极限附近学好（§3.10）。能；大世界模型明显更好，此后默认用大模型
- [ ] 加长训练（GPU），看终止率能否降到 E4 的量级（< 1%）、大小模型差距是否保持；多种子、更多留出参考
- [ ] 统计终止原因，确认极限附近的失效以横向跑偏为主
- [ ] 世界模型损失对饱和区样本加权或过采样，速度组加大权重，用 `jacobian_check.py` 复查
- [ ] 小集成，用方差做不确定性惩罚
- [ ] E3（视野与频率）和 E4（全库训练），在策略训练里比较大小两种世界模型在极限附近的表现

## 常见问题

- **导入 mujoco 时报 EGL 错误**：加 `MUJOCO_GL=disable`。
- **demo 或检查脚本比上表慢很多**：多个脚本或测试同时跑会抢 CPU。CPU 上请串行运行。
- **数字与 `docs/results/` 不一致**：先确认种子和参数都是默认值。CPU 上的结果是确定的；换机器或换 BLAS 后，浮点舍入可能不同，数字会有小差异，但结论不应改变。
- **开启 ABS/ESC 做 E1 或第 2 步**：两个脚本都固定关闭它们（`env_cfg()` 里）。门限逻辑附近的有限差分没有意义；如果要研究开启时的梯度质量，需要另行设计（例如只在远离门限的样本上比较）。
