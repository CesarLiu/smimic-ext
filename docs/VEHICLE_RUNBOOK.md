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
| E0 | 单卡吞吐 ≥ 10⁶ 环境步/秒 | ✅ 通过（H200，65536 个环境：完整环境 eager 1.9×10⁶，compile 1.3×10⁷） | §3.1，第 3 步 |
| demo | 整条 SuperMimic 循环在车辆上学会跟踪 | ✅ 通过：留出参考回报 0.10 → 0.36（加入原始动作约束后；之前为 0.62，见 §3.8） | §3.8，`docs/results/vehicle_demo_e2e.json` |
| E1 | 饱和区雅可比检查（修订后的关卡） | ✅ 大容量世界模型通过（饱和区 8 步梯度余弦 0.94，误差 0.11）；默认规模未通过 | §3.9，`docs/results/vehicle_jacobian_check*.json` |
| 第 2 步 | 策略在附着极限附近学跟踪，大小世界模型对比（无关卡，只测量） | ✅ 已测两次（两台机器）：三档摩擦都学会（回报约 0.1 → 0.38–0.76）；大模型在 12 个对比项里 11 项更好、1 项持平，但单次运行的数字波动很大，需要多种子；终止率仍有 33–50% | §3.10，`docs/results/vehicle_limit_study.json` |
| E6（部分） | S 弯 demo：SuperMimic 对比 PPO（同一训练库，S 弯为未见过的测试） | ✅ CPU 上跑过一次（各 8 分钟，单种子）：相同数据量下 SuperMimic 样本效率高约 3–4 倍；相同时间下 PPO 更好（0.96 对 0.81）；两者都跑完 S 弯。待 H200、多种子。NMPC 基线已实现，只用策略拿到的车载信息时：各项回报 0.99、横向误差 0.01–0.03 m，噪声、延迟、参数失配下无终止；每步求解约 9 ms，策略约 0.1 ms。这个设定下 NMPC 明显优于两个学习方法 | §3.11、§3.12，第 7 步，`docs/results/vehicle_compare_ppo_cpu*`、`vehicle_nmpc*` |
| demo | 麋鹿测试（ISO 3888-2 避障变道，按锥桶接触评分） | ✅ CPU 上跑过（赛道尺寸已按二手资料更正，第一版结果作废）：车载信息版 NMPC 通过到干路 60、湿滑路 45 km/h（参考线的上限）；两个策略在所有速度都碰到 1–5 个桶 | §3.13，第 8 步，`docs/results/vehicle_elk*` |

## 环境

与仓库其余部分相同（见 README 的 Setup）：Python 3.12，`uv sync --group dev`。车辆部分只依赖 torch，**全部可以在 CPU 上跑**；只有吞吐关卡需要 GPU。

- 下文命令都写成 `env -u PYTHONPATH uv run ...`；不用 uv 时，等价于 `PYTHONPATH=src python ...`。
- 没有显示环境的机器上，如果导入 mujoco 时 EGL 报错，在命令前加 `MUJOCO_GL=disable`（车辆部分不渲染）。
- NMPC 基线需要 `casadi`：`uv pip install casadi`（`uv run` 不会把它删掉）。不装也行，`compare_ppo.py` 加 `--methods supermimic ppo` 即可；脚本在训练开始前检查。
- 只有重新生成 CommonRoad 参考（第 2 步）时才需要 `uv pip install commonroad-vehicle-models`。仓库已带参考数据，平时不用装。

下面的耗时都是在 4 核 CPU 上实测的。

**CPU 线程数由脚本自己限制**：所有车辆脚本、验证模块和测试（`tests/conftest.py`）启动时都把 PyTorch 限制在 `min(8, 可用核数)` 个线程；设置了 `OMP_NUM_THREADS` 时用它；也可以用 `--threads N` 指定。启动时会打印 `cpu threads: N (of M usable cores)`。

这样做的原因：这些脚本训练的都是小网络，PyTorch 默认每个核开一个线程，线程同步的开销远大于计算本身。早期版本的脚本不设限，在 192 核机器上会占满整台机器、几乎卡死，E1 默认规模跑了 1949 s，4 核机器上只要 229 s（实现见 `src/supermimic/vehicles/threads.py`）。在 4 核机器上，默认值就是 4，与之前的结果逐位相同。

**哪些脚本用 GPU**：`bench_sim.py`、`demo_e2e.py`、`limit_study.py` 和 `compare_ppo.py` 支持 `--device cuda`；`jacobian_check.py` 和验证脚本只在 CPU 上跑。

## H200 服务器上的完整命令（192 核 + GPU）

在仓库根目录执行。脚本默认各用 8 个 CPU 线程，不需要额外前缀。并行跑时，用 `taskset` 把每个任务绑在不重叠的核上，避免互相抢核。GPU 任务用 `CUDA_VISIBLE_DEVICES` 选卡（例子里用 3 号卡，换成空闲的卡）。各步的检查标准见后面的第 1–6 步。

```bash
git pull
mkdir -p logs/e1 logs/limit logs/vehicle_demo logs/bench
```

**① 测试和对拍（几分钟，先跑，确认环境正常）**

```bash
env -u PYTHONPATH uv run pytest tests -q
env -u PYTHONPATH uv run python -m supermimic.vehicles.validation
```

预期：`136 passed, 19 skipped`；`worst metric 4.08% (gate 5%)`。

**② GPU：吞吐和 demo（各几分钟）**

```bash
CUDA_VISIBLE_DEVICES=3 env -u PYTHONPATH uv run python scripts/vehicle/bench_sim.py --device cuda --num-envs 4096 16384 65536 --compile | tee logs/bench/h200.txt
CUDA_VISIBLE_DEVICES=3 env -u PYTHONPATH uv run python scripts/vehicle/demo_e2e.py --device cuda --out logs/vehicle_demo/report_gpu.json | tee logs/vehicle_demo/gpu.log
```

预期：吞吐 `env` 一列在 65536 个环境时约 1.3×10⁷；demo 打印 `PASSED`。GPU 上的数字不会与 CPU 报告逐位相同。

极限研究也可以放在 GPU 上跑（`--device cuda`；参考仍在 CPU 上生成，所以与 CPU 版完全相同）：

```bash
CUDA_VISIBLE_DEVICES=3 env -u PYTHONPATH uv run python scripts/vehicle/limit_study.py --device cuda --out logs/limit/report_gpu.json | tee logs/limit/gpu.log
```

默认规模（256 个环境、batch 256）时，GPU 不一定比 8 线程 CPU 快，因为每个算子都很小。GPU 的价值在加大规模（见第 6 步）。

**③ CPU 上的长任务：三个并行，各绑 8 个核（约 25 分钟，由最长的极限研究决定）**

```bash
taskset -c 8-15  env -u PYTHONPATH uv run python scripts/vehicle/jacobian_check.py --out logs/e1/default.json > logs/e1/default.log 2>&1 &
taskset -c 16-23 env -u PYTHONPATH uv run python scripts/vehicle/jacobian_check.py --hidden 512 512 --updates 12000 --collect 1500 --out logs/e1/large.json > logs/e1/large.log 2>&1 &
taskset -c 24-31 env -u PYTHONPATH uv run python scripts/vehicle/limit_study.py --out logs/limit/report.json > logs/limit/report.log 2>&1 &
wait
tail -n 3 logs/e1/default.log logs/e1/large.log; tail -n 8 logs/limit/report.log
```

预期：
- 每个日志第一行是 `cpu threads: 8 (of 8 usable cores)`；
- E1 默认规模打印 `FAILED`（误差 0.551），这是已知结果；
- E1 大容量打印 `PASSED`（0.943 / 0.115）；
- 极限研究打印一张表。“训练前”一行应与报告相同；“训练后”的数字在不同机器上会明显不同（见第 6 步），这台服务器上的一次结果记录在 §3.10 的运行 B。

**④ 可选：CPU 上跑 demo（约 8 分钟，可以和 ③ 一起跑）**

```bash
taskset -c 32-39 env -u PYTHONPATH uv run python scripts/vehicle/demo_e2e.py --out logs/vehicle_demo/report_cpu.json > logs/vehicle_demo/cpu.log 2>&1 &
```

预期：打印 `PASSED`。训练前的 0.104 应与报告相同；训练后的数字在不同机器上可能与报告（0.356）明显不同，原因同第 6 步。

**说明**
- **线程数**：8 是默认上限。想试别的值，用 `--threads N`，或用常见问题里的小测试找出最快的线程数。用 `taskset` 绑核时，核数要与线程数相等。
- **核范围**：如果机器上有别人的任务，用 `htop` 找空闲的核，换掉 `taskset -c` 的范围。
- **进度**：并行任务的进度看各自的 `.log`；`jobs` 看哪些还在跑。

## T4 服务器：多种子极限研究（下一步要跑的）

§3.10 的问题是单次运行波动太大，大小世界模型的差距到底多大还不知道。这一步用多个种子回答它。

- 极限研究在 CPU 上跑：默认规模下每个算子都很小，T4 不会比 8 线程 CPU 快；
- 每个种子是一个独立进程，各绑 8 个核，按机器核数尽量并行；
- 种子同时改变参考轨迹和训练过程，所以得到的波动包含这两部分。

**① 准备（1 分钟）**

```bash
git pull
nproc; nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p logs/limit logs/bench logs/vehicle_demo
env -u PYTHONPATH uv run pytest tests -q -x     # 预期 136 passed, 19 skipped
```

**② 可选：T4 上的 GPU 检查（几分钟）**

```bash
env -u PYTHONPATH uv run python scripts/vehicle/bench_sim.py --device cuda --num-envs 4096 16384 65536 --compile | tee logs/bench/t4.txt
env -u PYTHONPATH uv run python scripts/vehicle/demo_e2e.py --device cuda --out logs/vehicle_demo/report_t4.json | tee logs/vehicle_demo/t4.log
```

T4 的吞吐预计比 H200 低一个数量级左右，未实测。demo 应打印 `PASSED`。

**③ 多种子极限研究（主任务）**

```bash
SEEDS="0 1 2 3 4 5 6 7"
N=$(nproc); T=$(( N < 8 ? N : 8 )); J=$(( N / T ))   # 每个进程 T 个线程，同时 J 个进程
echo "cores $N, threads per run $T, parallel runs $J"
i=0
for s in $SEEDS; do
  c=$(( (i % J) * T ))
  taskset -c $c-$((c + T - 1)) env -u PYTHONPATH uv run python scripts/vehicle/limit_study.py --seed $s \
      --out logs/limit/seed_$s.json > logs/limit/seed_$s.log 2>&1 &
  i=$((i + 1))
  if (( i % J == 0 )); then wait; fi
done
wait
grep -L "^report:" logs/limit/seed_*.log    # 列出没跑完的种子（应为空）
```

耗时：每个种子 8 线程约 18–23 分钟（H200 服务器上 1084 s）。8 个种子大约需要 ⌈8 / J⌉ × 20 分钟，例如 32 核（J = 4）约 40 分钟。核少时可以减少种子数，至少 4 个才有意义。

进度看各自的日志：`tail -n 2 logs/limit/seed_*.log`。每个日志第一行应是 `cpu threads: 8 (of 8 usable cores)`。

**④ 汇总**

```bash
env -u PYTHONPATH uv run python scripts/vehicle/limit_seeds_summary.py logs/limit/seed_*.json --out logs/limit/summary.json
```

输出两部分：
- 每个模型、每档摩擦训练后的均值 ± 标准差和范围；
- **大减小，按种子配对**的差：均值、粗略的 95% 区间（正态近似，种子少时很宽），以及“大模型更好 / 持平 / 种子数”。

**怎么读**：
- 区间整体在 0 以上，说明大模型确实更好，均值就是差距的估计；
- 区间跨过 0，说明在这个训练量下看不出差别，§3.10 的结论 2 要再弱化；
- 同一模型的标准差是单次运行的波动，以后判断别的改动是否有效时要以它为参照。

把 `logs/limit/summary.json` 和 `seed_*.json` 发给我，我写进 §3.10。

## 待办

### 1. 测试（约 2–4 分钟）

- [ ] 全套测试

```bash
env -u PYTHONPATH uv run pytest tests -q
```

**检查**：CPU 上应为 136 passed，19 skipped（跳过的需要 GPU 或 LAFAN1 数据）。车辆相关的五个文件：

| 文件 | 测什么 |
|---|---|
| `test_vehicle_sim.py` | 轮胎公式与 CommonRoad 逐点相等；14 个工况对拍 ≤ 5%；子步数收敛；ABS/ESC 有效；摩擦缩放；批内各行互不影响 |
| `test_vehicle_env.py` | 回放参考指令能复现参考；终止相对参考判定；指令延迟；ξ 采样；回放缓冲区；重置采样器；按摩擦生成参考；原始动作截断与越界惩罚；仿真器非有限值时终止 |
| `test_vehicle_e2e.py` | 学习端对每条转移的打分与环境一致；状态表示的接口约定；小规模跑通 demo 和极限研究 |
| `test_vehicle_jacobian.py` | `logged_step` 与环境逐位相等；两种雅可比估计在已知系统上精确；小规模跑通检查脚本 |
| `test_ppo.py` | PPO 的 GAE 与手算一致；SuperMimic 对比 PPO 的脚本小规模跑通 |
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

- [x] 在 GPU 上测吞吐（H200，2026-11-04，已通过）

```bash
env -u PYTHONPATH uv run python scripts/vehicle/bench_sim.py --device cuda --num-envs 4096 16384 65536
env -u PYTHONPATH uv run python scripts/vehicle/bench_sim.py --device cuda --num-envs 4096 16384 65536 --compile
```

多卡机器上用 `--device cuda:3` 这样的写法选卡。

**检查**：`env` 一列（完整环境步，含执行器、奖励、回放记录、重置）至少有一个规模达到 1,000,000 steps/s，即为通过。

H200 上的实测值（单位：步/秒），可用作对照：

| 环境数 | eager dynamics | eager env | compile dynamics | compile env |
|---|---|---|---|---|
| 4096 | 132,594 | 116,313 | 5,938,580 | 839,451 |
| 16384 | 530,628 | 463,706 | 23,017,056 | 3,310,053 |
| 65536 | 2,112,797 | 1,854,382 | 92,412,499 | 13,111,939 |

eager 下吞吐与环境数近似成正比，说明瓶颈是算子启动开销，所以环境数要开大。compile 后动力学比完整环境快约 7 倍，此时瓶颈在环境的记账逻辑。CPU 参照值：16384 个环境时 eager 约 1.2×10⁵，compile 约 1.7×10⁵。

**不过关时**：先看 `dynamics` 一列。如果动力学本身就不够快，瓶颈在逐元素算子的启动开销：先确认 `--compile` 是否生效，再考虑把子步循环写成 CUDA graph 或 Warp kernel。如果动力学够快而环境慢，瓶颈在环境的记账逻辑（回放写入、重置）。

### 4. 端到端 demo（CPU 约 8 分钟）

- [ ] 跑 demo

```bash
env -u PYTHONPATH uv run python scripts/vehicle/demo_e2e.py --out logs/vehicle_demo/report.json
# GPU：--device cuda（见下方说明）
```

**检查**：
- 最后打印 `PASSED`，退出码 0；
- 种子 0 下，训练前的数字（回报 0.104、坚持步数 86）在任何机器上都应与 `docs/results/vehicle_demo_e2e.json` 相同。训练后的数字（报告中 0.356、254 步、世界模型 8 步误差 0.25）只在同一台机器上可复现，换机器可能明显不同（见常见问题）；
- 成功判据写在脚本的 `check()` 里：回报至少翻倍且至少高 0.15，坚持步数至少翻倍，世界模型误差 < 3。

**在 GPU 上跑 demo**：
- 默认 256 个环境时，GPU 不会比 CPU 快多少。网络和批都很小，时间主要花在算子启动上；
- GPU 上的数字不会与 CPU 的报告逐位相同（浮点运算顺序不同、随机数流不同），但应该同样打印 `PASSED`；
- 加大 `--num-envs` 时，回放缓冲区的行数 = 环境数 × (episode_steps + 50)。超过 2²⁴（约 1678 万，例如 65536 × 300）时，旧代码会在世界模型第一次更新（第 100 次迭代后）崩溃，因为 CUDA 的 `torch.multinomial` 最多支持 2²⁴ 个类别。已修复：超过这个上限时改用等价的均匀抽样。

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
- 种子 0 下数字应与 `docs/results/vehicle_jacobian_check.json`、`vehicle_jacobian_check_large.json` 接近；只在同一台机器上逐位一致（见常见问题）。

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

### 6. 第 2 步：极限附近的策略学习（CPU 约 23 分钟；也可在 GPU 上跑）

- [ ] 跑极限研究（依次训练小、大两种世界模型）

```bash
env -u PYTHONPATH uv run python scripts/vehicle/limit_study.py --out logs/limit/report.json
# 只跑一种：--models large；改摩擦档：--mu-levels 0.5 0.8 1.0
# GPU：--device cuda（多卡时加 CUDA_VISIBLE_DEVICES=3 或 --device cuda:3）
```

**加大规模（GPU，新的实验配置，尚未跑过）**：E4 方向要看更长、更大的训练能否把终止率降下来，例如：

```bash
CUDA_VISIBLE_DEVICES=3 env -u PYTHONPATH uv run python scripts/vehicle/limit_study.py --device cuda --models large \
    --num-envs 4096 --batch 1024 --iters 10000 --refs-per-level 48 --out logs/limit/large_scale.json | tee logs/limit/large_scale.log
```

这组参数没有调过：学习率、`--warmup` / `--policy-warmup` 可能需要随规模调整；结果不能与默认规模的报告直接比较。

**检查**：
- 退出码 0，最后打印一张表（每档摩擦：回报 / 近极限仍在参考上的比例 / 近极限横向误差）；
- “训练前”一行应与 `docs/results/vehicle_limit_study.json` 相同（参考和初始策略在任何机器上都一样）；
- **“训练后”的数字只在同一台机器上可复现**。同一台机器上，结果与线程数无关、逐位确定。换机器后，微小的数值差异会被训练过程放大：会话环境（运行 A）和 H200 服务器（运行 B）上同一命令的结果，回报相差可达 0.21，近极限比例相差可达 48 个百分点。两次的对照见 §3.10；
- 运行 A 的训练后结果：小 0.516 / 0.47、0.381 / 0.49、0.405 / 0.12；大 0.590 / 0.57、0.530 / 0.95、0.489 / 0.49。运行 B：小 0.724 / 1.00、0.521 / 0.33、0.474 / 0.60；大 0.759 / 1.00、0.542 / 0.72、0.582 / 0.80；
- 该看的是大小模型的相对关系，单次运行的绝对数字不可靠；
- 没有通过/不通过的判定，这是测量。

**怎么读日志**：
- `wm 8-step` 在策略开始学习后（第 300 次迭代）会先升到 1–3，再降到 0.1–0.3，与 demo 相同；
- **如果它在后期突然升到 10 以上、`critic` 跟着上千**：这是修正前的发散模式（§3.10）。先检查回放里待执行指令那几维（状态第 18–23 维）是否有界（应 ≤ 3），再检查奖励里 `action_bound` 一项是否生效。

**结果怎么读**：
- “近极限”指参考自身利用率 > 0.8 的步，分母对所有策略相同（`near_limit_steps`）；
- “仍在参考上的比例”是策略在这些步上尚未终止的比例，横向 / 航向 / 侧偏误差只在这些步上平均，所以撑得更久的策略会在更难的段上被计入误差（幸存者偏差）；
- 每档只有 6 条留出参考，终止率的分辨率是 1/6。

### 7. S 弯 demo：SuperMimic 对比 PPO 和 NMPC（H200，约 1.3 小时）

- [ ] 在随机弯道库上分别训练 SuperMimic 和 PPO，在留出弯道和一条近极限 S 弯上比较

**设定**（`scripts/vehicle/compare_ppo.py`）：
- **训练集**：与第 6 步相同的随机弯道库（三档摩擦，侧向加速度峰值 40–95% μg，回合摩擦与参考一致，ABS/ESC 关）。两种方法都**没见过 S 弯**。
- **S 弯**（`ReferenceLibrary.s_curve`）：15 m/s 匀速，直道 → 左弧 → 右弧 → 直道。弧上侧向加速度 0.7μg，由同一个脚本驾驶员生成，实际附着利用率最高 0.97，390 步里约 155 步超过 0.8。每档摩擦一条。
- **相同的部分**：环境、观测、奖励、终止、ξ 随机化、重置采样器，以及网络结构。PPO 的 actor 就是 SuperMimic 的 `Policy`，外加一个可学的动作标准差；critic 就是 SuperMimic 的 `Critic`，输入同样的特权信息。
- **不同的部分**：SuperMimic 用大世界模型、256 个环境（与第 6 步相同）；PPO 用 4096 个环境，因为无模型方法需要更多样本（`src/supermimic/baselines/ppo.py`：裁剪目标、GAE、按 KL 自适应学习率，与 rsl_rl 的做法相同）。
- **NMPC**（`src/supermimic/baselines/nmpc.py`，不训练）：单轨模型，每个轴一条魔术公式侧向力曲线，参数从完整仿真器的随机转向数据拟合（相对 RMS 误差 3–5%）；多重打靶，1 s 视野，代价是奖励里的五项误差加输入变化惩罚，CasADi + IPOPT。有两个变体：`nmpc` 读真实状态和真实摩擦（上限参考）；`nmpc_onboard` 只拿策略拿到的信息（12 维车载测量、状态和摩擦估计器、延迟补偿，位姿精确），是公平对手。脚本同时记录 NMPC 每步求解时间和两个策略每步推理时间。
- **评估条件**对所有方法相同：`--eval-noise`（训练时的测量噪声）、`--eval-delay N`（固定 N 步指令延迟）、`--eval-mismatch`（刚度 ×0.85、质量 ×1.15、质心前移 0.1 m；NMPC 仍用名义模型）。默认是无噪声、无延迟的原评估。`--policies-from policies.pt` 直接评估已存的 SuperMimic 和 PPO，不重新训练。
- **预算**：两者各给相同的训练墙钟时间（`--budget-min`，不含评估时间）。同时记录两条曲线：回报对训练时间，回报对仿真步数。

```bash
mkdir -p logs/compare
CUDA_VISIBLE_DEVICES=3 env -u PYTHONPATH uv run python scripts/vehicle/compare_ppo.py --device cuda --budget-min 30 \
    --out-dir logs/compare/seed0 | tee logs/compare/seed0.log
```

评估每 2 分钟一次（`--eval-every-s`）。总耗时约 2 × 30 分钟训练加上评估时间。

**输出**（`logs/compare/seed0/`）：
- `curves.png`：学习曲线，上排是留出弯道，下排是 S 弯；左列横轴是训练时间，右列是仿真步数；
- `s_curve.png`：每档摩擦一列，上面是参考和两个策略的路径（× 表示在那里终止），下面是横向误差随时间的变化；
- `report.json`：曲线、最终评估（每档摩擦的回报、终止率、近极限指标）和 S 弯上逐步的轨迹；
- 日志最后一张表：两种方法在留出弯道和 S 弯上的最终回报，`ok` 表示跑完了 S 弯，`end` 表示中途终止。

**按迭代数给预算**：墙钟时间预算取决于机器快慢。同一份代码，在会话环境重启后的慢宿主机上，8 分钟只够 SuperMimic 训练 1240 次迭代（§3.11 那次是 2700 次），结果差很多。要在不同机器上复现同样的训练量，用 `--sm-max-iters 2700 --ppo-max-updates 408`，再把 `--budget-min` 设大一些；预算和迭代数哪个先到，就在哪里停。

**评估视频**：`compare_ppo.py` 会把两个训练好的策略存到 `policies.pt`，用它渲染 S 弯视频（三档摩擦依次播放，每档约 8 秒，实时速度；渲染约 2–3 分钟，用 CPU）：

```bash
env -u PYTHONPATH uv run python scripts/vehicle/render_scurve.py --policies logs/compare/seed0/policies.pt --out logs/compare/seed0/s_curve.mp4
# 只看一档：--mu 1.0；没有 ffmpeg 时自动改写 GIF（也可以直接给 --out ...gif）
```

画面左边是跟随各辆车的俯视镜头（车身、前轮转角、行驶轨迹、参考线，图例里实时显示车速、横向误差和附着利用率，终止时打 ×）；右边是整条弯道、横向误差曲线，以及附着利用率曲线（虚线是参考的，1.0 是附着极限）。

没有 NMPC 的旧运行，可以用 `--nmpc`（真实状态）和 `--nmpc-onboard`（车载信息）把它加进视频（不用重新训练）。

示例视频：[`docs/results/vehicle_compare_s_curve.mp4`](results/vehicle_compare_s_curve.mp4)，蓝色 SuperMimic、红色 PPO、紫色 NMPC（车载信息版，无噪声、无延迟的评估）。两个策略来自 CPU 上一次按迭代数给预算的运行（`--sm-max-iters 2700 --ppo-max-updates 408 --ppo-envs 1024`）：SuperMimic 训练 2700 次迭代，用了 69 万仿真步；PPO 训练 408 次更新，用了 1337 万仿真步。三者都跑完了三档摩擦的 S 弯，没有一档中途终止，数字见下面的表格。PPO 的跟踪比 SuperMimic 紧，但用的仿真数据约是 SuperMimic 的 19 倍。

**NMPC 结果**（CPU，§3.11 的训练结果，原始数据 `docs/results/vehicle_nmpc_fair_cpu.json`；完整表格和条件说明在 §3.12）：S 弯平均横向误差 [m]，μ = 0.5 / 0.8 / 1.0。

| 条件 | SuperMimic | PPO | NMPC（特权） | NMPC（车载信息） |
|---|---|---|---|---|
| 干净 | 0.24 / 0.31 / 0.27 | 0.09 / 0.13 / 0.16 | 0.03 / 0.02 / 0.02 | 0.02 / 0.01 / 0.02 |
| 噪声 + 2 步延迟 | 0.25 / 0.31 / 0.25 | 0.08 / 0.11 / 0.13 | 0.03 / 0.08 / 0.19（μ=1.0 终止） | 0.02 / 0.01 / 0.01 |
| 再加参数失配 | 0.26 / 0.33 / 0.24 | 0.08 / 0.10 / 0.12 | 0.04 / 0.16 / 0.25（μ=0.8、1.0 终止） | 0.03 / 0.02 / 0.01 |

每步计算（CPU）：NMPC 约 9 ms，策略约 0.1 ms。

**怎么读**：
- 只用策略拿到的信息时，NMPC 仍是厘米级，比 PPO 小 2–10 倍，比 SuperMimic 小 7–35 倍，噪声、延迟和参数失配都不影响它。特权版在有延迟时反而失败，因为它没有延迟补偿。注意设定对 NMPC 有利（2 维动作、单轨模型够用、质量几何和执行器参数已知），详见 §3.12 的局限；
- 学习策略目前的优势只有部署代价（约 0.1 ms 对 9 ms，不需要在线求解器和估计器）；
- 时间曲线回答“同样的 GPU 时间，谁学得更好”，步数曲线回答“同样的数据量，谁学得更好”。SuperMimic 的卖点在后者：它用世界模型生成梯度，样本效率应高得多；PPO 靠大量并行环境追时间；
- S 弯是泛化测试：训练库里没有这种几何，看两者能否跑完、横向误差多大；
- 这是单种子。§3.10 的经验是，单次运行的数字波动很大，结论要靠多种子（换 `--seed`，再比较）。

**CPU 上的参照结果**（§3.11，各 8 分钟，PPO 1024 个环境，`docs/results/vehicle_compare_ppo_cpu*`）：
- 相同数据量：约 50 万步时 SuperMimic 0.78、PPO 0.39；PPO 要约 200 万步才到 0.79；
- 相同时间结束时：留出弯道 SuperMimic 0.81、PPO 0.96；S 弯上两者都跑完，PPO 横向误差约为 SuperMimic 的一半到三分之一。

H200 上 PPO 的环境数多 4 倍，SuperMimic 的设置不变，预计 PPO 在时间曲线上的优势会更大，步数曲线的关系应该不变。

**已知局限**：
- **PPO 没有调过参**：我只在 CPU 上做了短时间运行，确认它会学，长时间也不发散。超参数取 rsl_rl 的常用默认值（每次更新 32 步、5 个 epoch、4 个 minibatch、学习率 3e-4、KL 目标 0.01、初始 std 约 0.6）。如果 PPO 明显不如预期，先试 `--ppo-envs 16384`、`--ppo-entropy 0.001`；
- **SuperMimic 的设置是在 2500 次迭代内验证过的**。30 分钟预算下，它会跑过这个范围（学习率不衰减）；
- 时间预算对两者是同一张 GPU，但 SuperMimic 只用 256 个环境，GPU 利用率低；加大 `--num-envs` 和 `--batch` 是另一组未验证的配置。

**冒烟测试**（CPU，3 分钟，只检查流程能跑通）：

```bash
env -u PYTHONPATH uv run python scripts/vehicle/compare_ppo.py --budget-min 0.5 --eval-every-s 15 --mu-levels 0.5 1.0 \
    --refs-per-level 2 --eval-refs-per-level 2 --ref-steps 200 --episode-steps 80 --num-envs 32 --batch 32 --warmup 10 \
    --policy-warmup 20 --horizon 6 --ppo-envs 64 --ppo-steps 16 --out-dir logs/compare/smoke
```

### 8. 麋鹿测试 demo（CPU，约 15 分钟）

- [ ] 用第 7 步训练出的策略和 NMPC 跑 ISO 3888-2 避障变道，看谁在多高的入场速度下不碰锥桶

```bash
# 干路 40–75 km/h、湿滑路（μ=0.5）30–50 km/h；参考线已缓存在 docs/results/vehicle_elk_refs.pt
env -u PYTHONPATH uv run python scripts/vehicle/elk_test.py --policies logs/compare/seed0/policies.pt --out-dir logs/elk
# 带测量噪声、2 步延迟和参数失配：加 --eval-noise --eval-delay 2 --eval-mismatch
# 只渲染其中几个：--video-cases 1.0:45,1.0:60,0.5:40；不要视频：--no-video
```

- 赛道（`src/supermimic/vehicles/elk.py`）：段长 12 / 13.5 / 11 / 12.5 / 12 m；车道宽 1.1b+0.25、b+1、1.3b+0.25（至少 3 m）；避让车道与入口车道之间 1 m 间隙（中心相距约 3.5 m）。尺寸取自二手资料，1 m 的含义、出口车道对齐方式、锥桶间距和大小未经标准原文核对（§3.13）；
- 评分：没被终止且没有碰到锥桶（锥桶底座与车道边界相切，车身碰到底座圆算碰桶）。另报“出线步数”。输出 `report.json`、`elk_paths.png`、`elk.mp4`，最后一行是每种方法的最高通过入场速度；
- 参考线：在 0.85 μg 内留出最大间隙（上限 0.10 m）的线，由特权 NMPC 开出。NMPC 自己都过不去的速度会被跳过（目前干路 > 60、湿滑路 > 45 km/h）。生成参考线要 casadi，结果追加进 `--refs-cache`；只评估已有速度不需要；
- 结果（CPU，§3.13）：车载信息版 NMPC 通过所有测试点；PPO 和 SuperMimic 在所有速度都碰桶（PPO 1–3 个，SuperMimic 2–5 个）。带噪声和失配时结果在低速会翻转（擦边接触），单次运行不足以下结论。示例视频 [`docs/results/vehicle_elk.mp4`](results/vehicle_elk.mp4)（干路 45、60 km/h，湿滑路 40 km/h）。

### 9. 下一步（E3/E4）

- [x] E1 的关卡按 §3.9 修订（2026-11-04）
- [x] 第 2 步：策略能否在附着极限附近学好（§3.10）。能；大世界模型明显更好，此后默认用大模型
- [ ] 加长训练（GPU），看终止率能否降到 E4 的量级（< 1%）、大小模型差距是否保持；多种子、更多留出参考
- [ ] 统计终止原因，确认极限附近的失效以横向跑偏为主
- [ ] 世界模型损失对饱和区样本加权或过采样，速度组加大权重，用 `jacobian_check.py` 复查
- [ ] 小集成，用方差做不确定性惩罚
- [ ] E3（视野与频率）和 E4（全库训练），在策略训练里比较大小两种世界模型在极限附近的表现

## 常见问题

- **导入 mujoco 时报 EGL 错误**：加 `MUJOCO_GL=disable`。
- **demo 或检查脚本比上表慢很多，或者占满整台机器**：脚本现在默认限制在 8 个线程（见“环境”一节），先确认日志第一行的 `cpu threads`。几个任务同时跑时，用 `taskset` 绑在不重叠的核上；容器限了 CPU 配额时，用 `--threads` 设成配额的核数。不确定用几个线程时，测一下每次更新的耗时，选最快的：

  ```bash
  for n in 2 4 8 16; do echo -n "threads $n: "; OMP_NUM_THREADS=$n taskset -c 0-$((n-1)) env -u PYTHONPATH uv run python -c "
  import time, torch
  net = torch.nn.Sequential(torch.nn.Linear(40,256), torch.nn.ELU(), torch.nn.Linear(256,256), torch.nn.ELU(), torch.nn.Linear(256,24))
  opt = torch.optim.Adam(net.parameters(), 1e-3); x = torch.randn(512, 40)
  def it():
      h = x; loss = 0
      for k in range(8):
          y = net(h); loss = loss + y.pow(2).mean(); h = torch.cat([h[:, :16], y], -1)
      opt.zero_grad(); loss.backward(); opt.step()
  for _ in range(20): it()
  t0 = time.time()
  for _ in range(200): it()
  print('%.1f ms / update' % ((time.time()-t0)/200*1e3))"; done
  ```

  4 核机器上 4 线程约 16 ms。另外，脚本打印的耗时不含导入。冷启动时导入约 1–1.5 分钟（脚本会间接导入 G1 的 mjlab 依赖链），第一次执行 `uv run` 还要先同步依赖。
- **demo 在 GPU 上第 100 次迭代后崩溃，报 `number of categories cannot exceed 2^24`**：回放缓冲区超过了 CUDA multinomial 的上限，见第 4 步，拉取最新代码即可。
- **数字与 `docs/results/` 不一致**：先确认种子和参数都是默认值。
  - 不涉及训练的结果，换机器也应一致或只差很小：对拍、吞吐以外的检查、各脚本“训练前”的评估；
  - 涉及训练的结果，只在同一台机器上逐位可复现：demo、E1、极限研究。同一台机器上与线程数无关。换机器后，CPU 指令集、库版本等带来的微小数值差异会被训练放大。E1 的结果对此不太敏感；极限研究很敏感（§3.10 的运行 A 与 B）；
  - GPU 上的结果与 CPU 也不同。
- **开启 ABS/ESC 做 E1 或第 2 步**：两个脚本都固定关闭它们（`env_cfg()` 里）。门限逻辑附近的有限差分没有意义；如果要研究开启时的梯度质量，需要另行设计（例如只在远离门限的样本上比较）。
