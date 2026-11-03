# 机器人领域的形式化方法：应用综述报告

> 版本：2026-11-04
> 范围：工业/协作机器人、移动机器人、无人机、足式与人形机器人、手术与特种机器人、基于 LLM/VLA 的机器人
> 姊妹篇：[自动驾驶安全中的形式化方法](AD_Safety_Formal_Methods_Report.md)（本报告不再重复其中已展开的通用理论：STL 鲁棒性、可达性、CBF、NN 验证、conformal 等，只在机器人语境下说明用法和差异）
> 说明：标准与法规状态以检索当日公开信息为准；2026 年的 arXiv 论文多数只读了摘要，标注为"未读全文"。参考文献见第 16 节。

---

## 0. 要点摘要

1. **机器人比自动驾驶更早、更广泛地使用形式化方法，但量产落地更窄。** 机器人领域的形式化方法有三条主线：
   - **任务级：** 时序逻辑规划与反应式综合，从 2009 年前后成为机器人学的经典方向。
   - **控制级：** CBF、HJ 可达性、SOS/Lyapunov 验证，大量在无人机、足式机器人上实机验证。
   - **软件级：** ROS 系统的静态分析与运行时验证，以及 NASA 的需求形式化到运行时监控的工具链。
2. **与 AD 相比，机器人的难点在接触和开放任务。**
   - **接触：** 足式行走、操作、人机物理交互都是混成动力学，可达性和 CBF 的假设容易被打破。
   - **开放任务：** 任务由自然语言给出，规约本身是在运行时由 LLM 生成的。
3. **2025 到 2026 年最活跃的两个方向：**
   - **基础模型机器人的形式化护栏：**
     - 自然语言到 LTL/STL，并用 conformal 方法给出翻译成功率保证。
     - 时序逻辑约束的 LLM 规划器护栏，如 RoboGuard。
     - VLA 输出端的 CBF 安全层。
   - **人形机器人的安全过滤器：**
     - 全身控制中的 CBF。
     - 可达性学习的可恢复性监控。
     - 与跟踪策略联合训练的过滤器。
     - 对安全过滤器的对抗压力测试。
   这和你在 SuperMimic 上的全身跟踪、鲁棒性裕度工作直接相关。
4. **法规正在追上来：**
   - ISO 10218-1/2 于 2025 年改版，功能安全要求更显式，并新增网络安全要求。
   - 人形机器人专用标准 ISO 25785-1 仍是委员会草案。
   - 欧盟机械法规 (EU) 2023/1230 自 **2027-01-20** 起适用，把"带有自演化（机器学习）行为、承担安全功能的系统"列为需要第三方合格评定的高风险类别。这会直接拉动对学习型机器人的可验证证据需求。
5. **主流工程模式与 AD 相同：** 不验证学习策略本身，而是用可验证的安全层（速度与分离监控、CBF/HJ 过滤器、shield、运行时监控）把它包起来，再用统计方法和测试补足。

---

## 1. 机器人与自动驾驶：形式化问题的差异

| 维度 | 自动驾驶 | 机器人 | 对形式化方法的影响 |
|---|---|---|---|
| 动力学 | 车辆动力学较规则，接触只有轮胎与路面 | 足式行走、抓取、推、装配都是多接触混成系统 | 可达性和 CBF 需处理模式切换与不连续，梯度和集合传播都更难 |
| 任务 | 固定任务（安全到达目的地加交通规则） | 任务多样且可组合（"先取杯子再去厨房，避开儿童"） | 时序逻辑规约和综合在机器人中远比在 AD 中重要 |
| 规约来源 | 交通法规、标准（相对稳定） | 用户指令、场景语义、安全规范（常在运行时给出） | 需要自然语言到形式化的翻译，以及翻译正确性的保证 |
| 与人的关系 | 人在车外，以避碰为主 | 人在工作空间内，允许甚至需要物理接触（协作、护理、手术） | 安全不是"不接触"，而是"接触力和压强在允许范围内"（ISO/TS 15066） |
| 规模 | 单车 | 多机器人、集群 | 需要组合式、分布式规约与验证 |
| 软件栈 | 车规级、AUTOSAR | ROS/ROS 2 为主，研究代码比例高 | 软件层验证（ROS 静态分析、运行时监控）有独特价值 |
| 监管成熟度 | 型式认证体系完整 | 工业机器人标准成熟，服务、人形机器人标准滞后 | 研究空间大，但缺少统一的验收口径 |

---

## 2. 标准与法规

| 标准/法规 | 状态 | 关注点 | 形式化方法的切入点 |
|---|---|---|---|
| ISO 10218-1/-2:2025 | 2025 年发布，替代 2011 版 | 工业机器人本体（Part 1，面向制造商）与应用集成（Part 2，面向集成商）的安全要求；功能安全要求更显式，新增网络安全要求 | 安全功能（限速、限位、安全停止）的验证与确认 |
| ISO/TS 15066:2016 | 现行（与 10218 配套） | 协作机器人：功率与力限制（PFL）、速度与分离监控（SSM）的身体部位力/压强阈值 | SSM 本质上是一个可达性问题：保护分离距离是人和机器人可达集之间的距离 |
| ISO 13849-1:2023 / IEC 62061 | 现行 | 机械控制系统安全相关部件（PL / SIL） | 安全控制逻辑的验证，与 IEC 61508 的软件技术表对接 |
| IEC 61508 | 现行（基础功能安全标准） | 软件安全完整性等级 | 在高 SIL 下形式化方法被列为推荐技术（以原文表格为准） |
| ISO 13482:2014 | 现行（修订中） | 个人护理机器人 | 物理人机交互安全 |
| ISO 3691-4:2023 | 现行 | 无人驾驶工业车辆与 AMR | 移动机器人避障、速度区域等安全功能 |
| ANSI/A3 R15.06-2025、R15.08 | 现行 | 美国工业机器人（采纳 10218）、工业移动机器人 | 同上 |
| ISO 25785-1 | 委员会草案（ISO/CD），2025-07 启动，尚未发布 | 动态稳定的工业移动机器人（人形、足式等），Part 2 计划覆盖应用集成 | 动态稳定机器人"断电即倒"的特性使得传统安全停止不再安全，需要可恢复性与安全状态的形式化定义 |
| IEC 80601-2-77 | 现行 | 机器人辅助手术设备 | 医疗器械软件（IEC 62304）加机器人安全 |
| DO-178C + DO-333 | 现行（航空） | 机载软件；DO-333 是形式化方法补充 | 无人机、航空机器人软件，形式化证据可替代部分测试 |
| ASTM F3269 | 现行（航空） | 运行时保障（RTA）架构，用于含复杂功能的无人系统 | Simplex 式架构的标准化 |
| EU 机械法规 (EU) 2023/1230 | 2027-01-20 起适用，替代 2006/42/EC | 安全部件定义纳入软件；带自演化（ML）行为且承担安全功能的机械需第三方合格评定；需考虑全生命周期内行为演化带来的风险 | 学习型安全功能需要可审查证据，这是形式化方法和统计保证的直接需求来源 |
| EU AI Act | 分阶段适用 | 作为安全部件的 AI 系统属高风险 | 与机械法规的衔接仍在厘清 |

**读法：**
- 工业机器人的安全一直靠"把人和机器人隔开，或者把力限制住"，规约简单、验证清晰。
- 人形机器人和学习型机器人打破了这个范式：机器人会跌倒，行为来自数据，任务开放。所以 ISO 25785-1 和欧盟机械法规都在要求新的安全论证形式。

---

## 3. 应用地图：哪类问题用哪类方法

| 层次 | 典型问题 | 主要技术 | 代表成果 | 成熟度 |
|---|---|---|---|---|
| 任务与任务规划 | "巡逻 A、B、C，永不进入 D，看到人就停" | LTL/STL 规约，自动机乘积规划，GR(1) 反应式综合，MILP，概率模型检测 | LTLMoP、Spectra、PRISM 服务机器人规划 | 高（研究），中（落地） |
| 运动规划与控制 | 避障、保持平衡、不超关节限位 | CBF、HJ 可达性、SOS 漏斗库、STL-MPC | 无人机 HJ 安全框架、足式 CBF、多机器人屏障证书 | 中高 |
| 人机物理交互 | 协作机器人接触力、手术虚拟夹具 | SSM 可达性、力阈值监控、定理证明 | ISO/TS 15066 SSM、手术虚拟夹具的 KeYmaera 证明 | 中（工业），低（服务、护理） |
| 学习型策略 | RL 策略、扩散策略、全身跟踪策略 | Shield、安全 RL、闭环 NN 验证、学习型安全过滤器 | 人形机器人安全过滤器（2026） | 低到中 |
| 基础模型机器人 | LLM 规划器、VLA | NL 到 LTL/STL，conformal 规划，时序逻辑护栏，VLA 安全层 | RoboGuard、KnowNo、AutoTAMP、ConformalNL2LTL | 低（快速发展） |
| 软件与系统 | ROS 节点通信、架构配置、运行时异常 | 静态分析、模型检测、运行时验证、Simplex | HAROS、ROSDiscover、ROSMonitoring、NASA FRET + Copilot、RoboChart | 中 |
| 多机器人与集群 | 任务分配、集群涌现行为 | 分布式 LTL/STL 综合、概率模型检测、屏障证书 | 多机器人 LTL 部署、集群行为模型检测 | 中（研究） |

---

## 4. 任务级：时序逻辑规划与反应式综合

这是机器人学中形式化方法最经典、成果最丰富的方向。

### 4.1 基本流程（以 LTL 为例）

1. **离散抽象：** 把工作空间分为区域（单元分解、语义地图），机器人运动抽象为转移系统 T。
2. **规约：** 用 LTL 写任务。例如：
   ```
   φ = G F a ∧ G F b ∧ G ¬d ∧ G (person → X stop)
   ```
   含义：无限次访问 a 和 b；永不进入 d；检测到人则下一步停止。
3. **自动机构造：** φ 转为 Büchi 自动机 B_φ（工具：Spot、LTL2BA）。
4. **乘积与搜索：** 在 T × B_φ 上找"前缀 + 循环"的接受路径，得到满足 φ 的离散计划。
5. **连续实现：** 每段离散转移由一个底层控制器实现。正确性依赖"抽象与连续动力学之间的双模拟或可实现性"，这需要单独论证或用可达性检查。

**代表工作：**
- Kress-Gazit, Fainekos, Pappas（T-RO 2009）：时序逻辑反应式任务与运动规划。
- Smith, Tumova, Belta, Rus（IJRR 2011）：带时序逻辑约束的最优路径。
- Bhatia, Kavraki, Vardi（ICRA 2010）：带时序目标的采样规划。
- Kloetzer & Belta（T-RO 2010）：机器人团队的自动部署。
- 综述：Kress-Gazit, Lahijanian, Raman（Annual Review of CRAS 2018）。

### 4.2 反应式综合

- 环境不受控，例如人可能随时出现。规约写成"环境假设 → 系统保证"，综合出一个对所有满足假设的环境行为都能满足规约的**策略**（有限状态控制器），而不是单条计划。
- 一般 LTL 综合是 2EXPTIME 完全的。**GR(1) 片段**可在多项式时间求解，覆盖大部分机器人任务，是实践中的主力。
- 工具：
  - LTLMoP（Kress-Gazit 组）。
  - Spectra（GR(1) 规约语言与综合工具）。
  - Slugs。
  - Strix（一般 LTL 综合，SYNTCOMP 常胜者）。
- **不可实现性反馈：** 规约无法实现时，给出环境反策略或最小不可满足核，帮助用户修改任务。这一点对人在环的任务定义很有价值。

### 4.3 STL 与优化

- STL 适合带时间窗和连续量的任务（"10 秒内到达 A 并停留至少 2 秒，同时与人保持 0.5 m 以上距离"）。
- **MILP 编码：** Raman、Donzé 等（CDC 2014）把 STL 编码为混合整数约束，嵌入 MPC。
- **基于鲁棒度的优化：** 光滑化的鲁棒度可做梯度优化；后续工作把 STL 鲁棒度用于 RL 奖励与模仿学习。
- **STL 与 CBF 结合：** Lindemann & Dimarogonas（L-CSS 2019）为 STL 任务构造时变 CBF，可在 QP 中在线满足 STL 片段。
- 综述：Belta & Sadraddini（Annual Review of CRAS 2019），从优化视角看控制综合。

### 4.4 概率规约

- 服务机器人在真实环境中执行任务，动作有随机结果。用 MDP 建模，用 PCTL 或 co-safe LTL 规约，用概率模型检测（PRISM、Storm）求"最大化任务完成概率"的策略，并给出**带保证的性能预测**（如期望完成时间）。
- 代表：Lacerda, Faruq, Parker, Hawes（IJRR 2019），带形式化性能保证的移动服务机器人概率规划，在长期部署的服务机器人上落地。

### 4.5 多机器人与集群

- **多机器人 LTL/STL：** 全局任务分解为局部任务，涉及可分解性和通信约束；规模问题通过分层或分布式综合缓解。
- **集群：** 用概率模型检测验证集群算法的涌现性质（如聚集、觅食），代表工作来自 Dixon、Winfield、Fisher 等。

---

## 5. 运动与控制级

通用理论见 AD 报告第 5、6 节，这里只讲机器人的特点和代表成果。

### 5.1 控制屏障函数（CBF）

- **足式机器人：** Nguyen & Sreenath（ACC 2016）提出指数 CBF，处理高相对阶约束，用于双足行走中的踏石（stepping stones）约束。
- **多机器人：** Wang, Ames, Egerstedt（T-RO 2017）的安全屏障证书，用分布式 QP 实现多机器人无碰撞，在 Robotarium 上大规模验证。
- **操作：** 关节限位、自碰撞、末端速度限制都可写成 CBF。接触约束（摩擦锥、单边接触）使问题变成混成 CBF，理论仍在发展。
- **人形全身控制（2026）：**
  - 输入到状态安全（ISSf）CBF 全身控制（arXiv 2605.25546）：在运动学模型上推导保证，再通过分层控制器在扰动下延伸到全身动力学。
  - 投影安全集算法用于杂乱环境中的人形灵巧控制（arXiv 2502.02858）。

### 5.2 Hamilton-Jacobi 可达性

- **无人机：** Fisac 等（IEEE TAC 2019）提出学习型控制的通用安全框架：用 HJ 值函数作为安全监督，学习型控制器只在安全集内部自由探索，接近边界时由最优安全控制接管，并在线更新扰动界。这是"学习 + 可证安全"的代表性实机结果。
- **维数问题：** 机器人状态维度高（人形 70 维以上）。可行的路线有三条：
  - 系统分解（如解耦纵向和横向）。
  - 神经近似（DeepReach）。
  - 学习式可达性（RL 求解折扣安全 Bellman 方程）。
  后两者需要额外的统计验证。

### 5.3 SOS、Lyapunov 与漏斗

- 用平方和（SOS）规划计算闭环系统的吸引域或不变集，为每条轨迹构造"漏斗"（funnel）。
- Majumdar & Tedrake（IJRR 2017）的漏斗库：离线计算一组带鲁棒保证的轨迹漏斗，在线组合，用于无人机在杂乱环境中高速飞行。
- 局限：多项式近似和 SOS 规模限制，难以扩展到高维接触系统。

### 5.4 闭环神经网络验证

- 在机器人上的典型对象：小型无人机控制器、倒立摆、ACC 类控制器（ARCH-COMP AINNCS 基准），以及 ACAS Xu 防撞网络（航空领域的标杆）。
- 工具与局限同 AD 报告 5.4 节：只适用于低维、小网络、短时域，人形全身策略远超可验证规模。

---

## 6. 人机物理交互与接触

### 6.1 协作机器人

- **SSM（速度与分离监控）：** ISO/TS 15066 的保护分离距离公式综合了人的速度、机器人速度、反应时间、停止时间和停止距离，本质上是两个可达集之间的距离检查。
  - 学术上用可达性分析给出更紧的界：基于人体运动模型的人体占据集预测，与机器人的可达集做在线比较。TUM Althoff 组在人机协作中做了这种在线验证。
- **PFL（功率与力限制）：** 按身体部位给出允许的瞬态与准静态接触力/压强，这是一类简单的不变式规约，可做运行时监控。
- **形式化危害分析：** 用时序逻辑和模型检测分析协作单元中的操作员与机器人交互，系统地发现危险情况。代表：Askarpour 等的 SAFER-HRC 方法。

### 6.2 手术机器人

- Kouskoulas 等（HSCC 2013）用 KeYmaera 证明了手术机器人**虚拟夹具**（virtual fixture，阻止器械进入禁区）控制算法的安全性，并在证明过程中发现了原设计的缺陷。这是定理证明在医疗机器人中的经典案例。
- 现状：医疗器械法规更看重过程（IEC 62304）与风险管理（ISO 14971），形式化方法多用于关键算法。

### 6.3 接触丰富的操作

- 抓取、插入、推动等任务的接触模式多、摩擦不确定，严格验证非常困难。目前更多依赖：
  - 力/力矩限制的运行时监控。
  - 学习型安全过滤器：例如潜空间安全过滤器（Nakamura 等, RSS 2025）用世界模型潜空间加 HJ 可达性，处理"打翻物品"这类难以手写的约束。
  - 统计验证（大量仿真加 conformal 界）。

---

## 7. 学习型策略的安全

### 7.1 Shield 与安全 RL

- **Shield 综合：** 从 LTL 安全规约和环境抽象 MDP 综合出"盾"，在 RL 训练和部署时屏蔽不安全动作（Alshiekh 等, AAAI 2018；Könighofer 等, FMSD 2017）。保证限于抽象模型。
- **安全 RL：** 约束 MDP（CMDP）与 Lagrangian 方法、CBF 或 HJ 安全层、恢复策略。综述：Brunke 等（Annual Review of CRAS 2022），从学习型控制到安全 RL。

### 7.2 人形机器人安全过滤器（2026，快速发展）

| 工作 | 要点 | 与你的工作的关系 |
|---|---|---|
| ConstrainedMimic（arXiv 2606.00374） | 在预训练人形跟踪策略的参考和输出两端加 CBF 检查；G1 空手道劈的最小屏障值从无过滤的 −0.41 提升到双过滤的 +0.02 | 与 SuperMimic 同为全身动作跟踪，可直接对照 |
| 过滤器感知微调 CoFiT（arXiv 2610.02341） | 跟踪策略与运行时 CBF 过滤器联合训练，违规时间相比只训练过滤器在 TWIST2 上降低 91%，在 SONIC 上降低 21% | 可微世界模型中的 BPTT 天然适合"把过滤器放进训练回路" |
| ResSafe（arXiv 2609.15988） | 用残差 RL 学习安全过滤器，基于最大生存时间的可达性形式，并声称把经典 CBF 和 HJ 过滤器作为特例 | 生存时间可达性与"鲁棒性裕度"在概念上一致 |
| 主动可达性学习用于可恢复性监控（arXiv 2610.33317） | G1 全身可恢复性监控；指出解析 CBF 对全身可恢复性扩展性差，PINN 类可达方法依赖均匀采样 | 与 `--kind adv` 的"模型预测裕度 vs 仿真裕度"是同一类问题 |
| SPARK 工具包（arXiv 2502.03132）及其对抗压力测试（arXiv 2605.19009） | 人形安全控制工具包；后续工作对其安全过滤器做对抗性压力测试 | 对安全过滤器本身做证伪，与你的最坏方向扰动思路一致 |
| Shield-Loco（arXiv 2606.07193） | 对运动策略做预测式安全过滤 | 只看到检索摘要，方法细节未核实 |

（上表 2026 年工作均只读了摘要或检索片段，数值以原文为准。）

**共同结论：**
- 人形机器人的安全不能简单理解为"不碰撞"，还要包括**可恢复性**（不进入必然跌倒的状态）和**接触与关节约束**。
- 解析方法难以扩展，学习型可达性和过滤器成为主流，因此**这些学到的安全集合本身是否可信、如何验证**成为新的核心问题。

---

## 8. 基础模型机器人的形式化护栏

LLM/VLA 让机器人可以接收开放的自然语言任务，同时引入了新的风险：
- 翻译错误：把指令理解错。
- 幻觉计划：生成不可行或不安全的动作序列。
- 越狱攻击：通过提示让机器人做危险动作。

形式化方法在这里的作用是**提供一个独立于 LLM 的、可检查的约束层**。

### 8.1 自然语言到时序逻辑

| 工作 | 目标逻辑 | 要点 |
|---|---|---|
| nl2spec（Cosler 等, CAV 2023） | LTL | 交互式，把公式分解为子公式供用户逐个确认 |
| Lang2LTL（Liu 等, CoRL 2023） | LTL | 模块化：指代识别、地标落地、提升的公式翻译；输出可接自动机规划器 |
| AutoTAMP（Chen 等, ICRA 2024） | STL | LLM 翻译为 STL，再由 STL 规划器求解，含自动语法/语义纠错回路 |
| ConformalNL2LTL（2025） | LTL | 用 conformal 方法对 LLM 翻译步骤给出用户可设定的成功率保证 |
| VerifyLLM（arXiv 2507.05118） | LTL | 先把指令转为 LTL，再在执行前分析动作序列 |

**关键点：** 翻译这一步本身没有保证，下游的自动机规划只保证"满足翻译出来的公式"。所以 conformal 类方法（给翻译正确率一个统计界）和人在环确认，是把保证延伸到语言端的两种手段。

### 8.2 Conformal 方法与 LLM 规划器

- **KnowNo**（Ren 等, CoRL 2023，《Robots that ask for help》）：对 LLM 规划器的多选动作做 conformal 预测，预测集合不唯一时向人求助，以统计保证控制任务成功率。
- **HERACLEs / conformal 时序逻辑规划：** 把自然语言子任务当作 LTL 原子命题，用 conformal 方法连接逻辑层与 LLM 规划器。
- 综述：Lindemann 等，*Formal Verification and Control with Conformal Prediction*（2024/2025），系统梳理 conformal 在验证与控制中的用法。

### 8.3 护栏与 VLA 安全层

- **RoboGuard**（arXiv 2503.07885）：两阶段护栏。
  - 第一阶段：一个可信根 LLM 把预定义的安全规则在当前环境中落地为上下文相关的 LTL 规约（如"不要导航到检测到的人旁边"）。
  - 第二阶段：用时序逻辑控制综合消解 LLM 计划与规约之间的冲突，在保证安全的前提下尽量少违背用户意图。
  - 实验在 Jackal 机器人加 GPT-4o 规划器上进行，主要针对越狱攻击。
- **VLA 输出端安全层：**
  - VLSA/AEGIS（arXiv 2512.11891）：CBF 安全约束层，不需要重训 VLA。
  - 注意力引导的安全过滤器（arXiv 2606.09749）：用 VLA 自身的注意力确定目标，其余物体作为障碍交给 CBF，无需慢速 VLM 介入。
  - 屏障增强的 flow matching（arXiv 2607.29569）：在动作块（action chunk）层面施加 CBF 约束。
  这三项都以操作任务为主。
- **约束 RL 微调 VLA：** 例如 SafeVLA 类工作，用约束 RL 把安全代价纳入 VLA 微调（无保证，但降低违规率）。

**结构性观察：** 以上方法都把"语义理解"交给基础模型，把"约束执行"交给形式化层。两者的接口是**原子命题的落地**（"这是一个人""这个杯子是热的"），它依赖感知的正确性，是整个链条中最薄弱的一环。

---

## 9. 软件与系统层

### 9.1 ROS/ROS 2 生态

| 工具 | 类型 | 作用 |
|---|---|---|
| HAROS | 静态分析框架 | 提取 ROS 计算图、检查编码规范、对架构做性质检查 |
| ROSDiscover（Timperley 等, ICSA 2022） | 静态架构恢复 | 从代码静态恢复运行时架构，发现话题名错误、未连接的发布者/订阅者等配置缺陷 |
| ROSMonitoring（Ferrando 等, 2020） | 运行时验证 | 在 ROS 话题上插入监控器，按形式化性质在线检查消息流 |
| RTAMT | STL 运行时监控 | 已有 ROS 2 集成示例 |
| Copilot / Ogma（NASA） | 运行时验证 | Copilot 是生成常数时间、常数内存 C 监控器的流式语言；Ogma 把 FRET 需求转换为 Copilot 监控器，并能生成 ROS 2 监控节点 |
| RoboChart / RoboStar（Cavalcanti 等） | 领域专用建模语言 | 用状态机加时间语义建模机器人控制器，自动转换为 CSP（FDR 模型检测）和 PRISM（概率性质） |

### 9.2 NASA 工具链：从需求到运行时监控

1. **FRET**（Formal Requirements Elicitation Tool）：用受限英语（FRETish）写需求，自动转为过去时/未来时 MTL，并支持需求的仿真与一致性检查。
2. **Copilot/Ogma：** 由形式化需求自动生成运行时监控器，部署到飞行软件或 ROS 2。
3. **R2U2：** 面向嵌入式平台（含 FPGA）的 MTL 运行时监控。

这是"需求形式化、自动生成监控、部署"的完整链条，对机器人项目可直接借鉴。

### 9.3 自主体（agent）验证

- 对 BDI 类高层决策智能体做程序模型检测：MCAPL 框架、AJPF 模型检测器、Gwendolen 语言（Dennis、Fisher 等）。验证"智能体在给定信念下是否总是做出符合规则的决定"。
- 用于核电、海上平台、卫星等自主系统的认证研究。英国 RAIN（核环境机器人）和 ORCA（海上机器人）研究中心有系统性工作。
- Fisher 等（2021）提出自主系统认证框架，把形式化验证、仿真、实测作为分层证据。

### 9.4 架构级运行时保障

- **Simplex：** 高性能（学习型）控制器加经验证的安全控制器。广泛用于无人机（ASTM F3269 对 RTA 做了标准化）和移动机器人。
- 关键的验证对象是**切换逻辑**与**安全控制器的可恢复集合**。可以用可达性或 HJ 计算，再用 KeYmaera 等工具证明切换条件正确。

---

## 10. 领域案例速览

| 领域 | 代表性形式化应用 | 落地情况 |
|---|---|---|
| 工业/协作机器人 | 安全 PLC 逻辑验证、SSM/PFL 运行时监控、协作单元的形式化危害分析 | 安全功能已量产；形式化证据多用于安全控制器软件 |
| 移动机器人/AMR | LTL 任务规划、PRISM 服务机器人规划、安全区域速度限制 | 速度区域与急停量产；任务级综合以研究为主 |
| 无人机 | HJ 安全框架、SOS 漏斗库、CBF 集群、DO-178C/DO-333、RTA（ASTM F3269） | 航空领域对形式化证据的接受度最高 |
| 足式/人形 | CBF 踏石行走、全身 CBF、可恢复性监控、学习型安全过滤器 | 研究快速发展，ISO 25785-1 尚为草案，无量产级保证 |
| 手术机器人 | 虚拟夹具的定理证明、力限制监控 | 关键算法级 |
| 空间/核/海上 | NASA FRET + Copilot、BDI 智能体验证、RAIN/ORCA 认证研究 | 任务关键、低频高风险场景，形式化价值最高 |
| 集群 | 概率模型检测验证涌现行为、屏障证书 | 研究为主 |
| 基础模型机器人 | NL 到 LTL/STL、conformal 规划、RoboGuard、VLA 安全层 | 原型阶段 |

---

## 11. 统计验证与测试

- **机器人测试的现实困难：** 仿真保真度（接触、摩擦、传感器）、场景构造成本、硬件在环昂贵。Afzal 等（ICST 2021）对机器人开发者的访谈研究总结了仿真测试的主要障碍。
- **证伪：** 与 AD 相同的工具（S-TaLiRo、Breach、VerifAI）可用于机器人。对学习型策略，可用 RL 或梯度搜索最坏扰动；在可微仿真器或可微世界模型中可直接对初始状态或扰动求梯度（与 SuperMimic `--kind adv` 同构）。
- **统计保证：**
  - conformal 方法：预测误差、学到的安全集合的误判率、LLM 规划成功率。
  - 场景优化/PAC：学到的 CBF 或值函数的认证。
  - 统计模型检测：任务成功率。
- **关键前提：** 仿真分布对真实分布的代表性。实机数据上的校准集与漂移检测，是让统计保证"落地"的必需环节。

---

## 12. 工具链速查

| 类别 | 工具 |
|---|---|
| LTL 到自动机 | Spot, LTL2BA, Owl |
| 反应式综合 | Spectra, Slugs, Strix, LTLMoP |
| 概率模型检测 | PRISM, Storm |
| 时间自动机 | UPPAAL |
| 混成系统证明 | KeYmaera X |
| 可达性 | CORA, JuliaReach, hj_reachability, DeepReach, helperOC |
| SOS/Lyapunov | Drake（含 SOS 与漏斗工具）, SOSTOOLS |
| CBF/QP | 各类 QP 求解器（OSQP, qpOASES）; Robotarium 示例代码 |
| STL 规划与监控 | RTAMT, Breach, S-TaLiRo, stlpy |
| ROS 分析与监控 | HAROS, ROSDiscover, ROSMonitoring, Copilot/Ogma |
| 需求形式化 | NASA FRET |
| 机器人建模语言 | RoboChart/RoboTool（到 FDR、PRISM） |
| 自主体验证 | MCAPL/AJPF |
| NN 验证 | α,β-CROWN, Marabou, NNV, POLAR, Verisig |
| 场景与证伪 | Scenic, VerifAI |

---

## 13. 方法与领域的成熟度矩阵

| 方法 \ 领域 | 工业/协作 | 移动机器人 | 无人机 | 足式/人形 | 操作 | LLM/VLA |
|---|---|---|---|---|---|---|
| 时序逻辑规划/综合 | 中 | 高 | 高 | 低 | 中（TAMP） | 中（翻译层） |
| 模型检测（软件/决策） | 高 | 中 | 高 | 低 | 低 | 低 |
| 可达性/HJ | 中（SSM） | 中 | 高 | 中（学习式） | 低 | 低 |
| CBF/安全过滤器 | 中 | 高 | 高 | 中（快速发展） | 中 | 中（操作 VLA） |
| 定理证明 | 低 | 低 | 中 | 低 | 低（手术例外） | 无 |
| 闭环 NN 验证 | 无 | 低 | 中（小网络） | 无 | 无 | 无 |
| 运行时验证 | 高 | 中 | 高 | 低 | 中 | 中（护栏） |
| 统计保证（conformal 等） | 低 | 中 | 中 | 低到中 | 中 | 中 |

（"高/中/低"是对研究和实机验证程度的定性判断，不是对量产认证程度的判断。）

---

## 14. 开放问题

1. **接触与混成动力学的可扩展验证：** 足式行走和操作中的模式切换使 CBF 和可达性的保证容易失效。混成 CBF、接触不确定性下的鲁棒保证仍不成熟。
2. **学到的安全集合是否可信：** 人形机器人的学习型可达性和过滤器越来越多，但对"学到的安全集合"本身的验证（误判率、在分布外的行为）缺少系统方法。
3. **落地问题（grounding）：** LLM 护栏中的原子命题依赖感知。"杯子是热的""这个人是儿童"一旦判断错，整个形式化链条就会失效。
4. **开放任务的规约正确性：** 用户的自然语言意图与形式化规约之间的一致性，只能做到统计意义上（conformal）或交互式确认。
5. **动态稳定机器人的"安全状态"定义：** 人形机器人断电即倒，传统"安全停止"不再安全。可恢复性、受控跌倒、安全姿态需要新的形式化定义，这也是 ISO 25785-1 正在处理的问题。
6. **全生命周期与持续学习：** 欧盟机械法规要求考虑行为演化。如何对"更新后的策略"做增量验证，而不是从头验证？
7. **多机器人与人群的组合验证：** 分布式规约、通信延迟、异构机器人。
8. **仿真到真实的保证迁移：** 统计保证依赖仿真分布；接触丰富任务的仿真偏差尤其大。

---

## 15. 研究方向建议

结合 SuperMimic（可微世界模型、全身跟踪、`--kind adv` 鲁棒性裕度）与 AD 安全背景：

### A. 世界模型预测的可恢复性裕度作为运行时监控
- **动机：** 主动可达性学习（arXiv 2610.33317）指出解析方法难以扩展到全身可恢复性，学习方法依赖采样。
- **做法：**
  - 用可微世界模型给出"沿最坏方向的扰动裕度"（即 `--kind adv` 的模型预测裕度）。
  - 用仿真作为真值，以 conformal 方法校准，得到"有覆盖保证的可恢复性下界"。
  - 部署为运行时监控：裕度低于阈值时切换到保守策略。
- **贡献：** 一个基于可微世界模型的、有统计保证的人形可恢复性监控器。这一工作直接回答了"世界模型是否知道策略在哪里脆弱"，与 AD 报告方向 A 是同一个问题的两个应用领域。

### B. 用可微世界模型对人形安全过滤器做梯度式压力测试
- **动机：** SPARK 安全过滤器已有对抗压力测试工作（arXiv 2605.19009），但多为黑盒搜索。
- **做法：** 在可微世界模型中对扰动做梯度搜索，寻找"策略加过滤器"闭环的最坏方向，与随机和黑盒搜索比较发现效率，并在仿真中复核。
- **贡献：** 把基于模型的证伪用于安全过滤器本身的验证，补上第 14 节第 2 点"学到的安全集合是否可信"。

### C. 把安全过滤器放进 BPTT 训练回路
- **动机：** CoFiT（arXiv 2610.02341）显示联合训练能大幅减少违规时间。SuperMimic 的 32 步 BPTT 本来就对整条想象轨迹求导。
- **做法：** 把可微的 CBF-QP（或其光滑近似）嵌入想象展开，让策略在训练时"看到"过滤器的干预。比较三种设置：过滤器只在部署时使用、过滤器在训练中使用、无过滤器。度量跟踪精度与违规率的权衡。
- **贡献：** 对"世界模型 + 解析梯度"训练范式特有的安全集成方式给出系统评估。

### D. STL 规约的全身动作约束与可达性
- **做法：**
  - 把关节限位、自碰撞、足底接触、躯干姿态等写成 STL。
  - 用 STL 鲁棒度作为跟踪策略的评估指标（替代只看 MPJPE）。
  - 进一步作为训练信号或过滤器约束。
- **贡献：** 为人形跟踪提供可审查、与标准（ISO 25785-1 草案）概念对齐的安全规约与评测。

### E. 跨领域：同一套安全过滤器理论在 AD 与人形之间迁移
- AD 中的在线验证、fail-safe 轨迹和人形中的可恢复性监控在结构上相同，都是"安全集合加回退策略加干预规则"（Hsu, Hu, Fisac 2023 的统一视角）。
- 以世界模型为共同基座，比较两个领域中模型预测裕度的校准性，可以形成一篇有广度的方法论文章。

---

## 16. 参考文献

### 综述与基础
- Luckcuck, M., Farrell, M., Dennis, L., Dixon, C., Fisher, M. Formal specification and verification of autonomous robotic systems: A survey. ACM Computing Surveys 2019.
- Farrell, M., Luckcuck, M., Fisher, M. Robotics and integrated formal methods: Necessity meets opportunity. iFM 2018.
- Kress-Gazit, H., Lahijanian, M., Raman, V. Synthesis for robots: Guarantees and feedback for robot behavior. Annual Review of Control, Robotics, and Autonomous Systems 2018.
- Belta, C., Sadraddini, S. Formal methods for control synthesis: An optimization perspective. Annual Review of Control, Robotics, and Autonomous Systems 2019.
- Belta, C., Yordanov, B., Gol, E. A. Formal Methods for Discrete-Time Dynamical Systems. Springer 2017.
- Brunke, L. et al. Safe learning in robotics: From learning-based control to safe reinforcement learning. Annual Review of Control, Robotics, and Autonomous Systems 2022.
- Hsu, K.-C., Hu, H., Fisac, J. The safety filter: A unified view of safety-critical control in autonomous systems. Annual Review of Control, Robotics, and Autonomous Systems 2024. arXiv:2309.05837.
- Fisher, M. et al. Towards a framework for certification of reliable autonomous systems. Autonomous Agents and Multi-Agent Systems 2021.

### 任务级规划与综合
- Kress-Gazit, H., Fainekos, G., Pappas, G. Temporal-logic-based reactive mission and motion planning. IEEE T-RO 2009.
- Kloetzer, M., Belta, C. Automatic deployment of distributed teams of robots from temporal logic motion specifications. IEEE T-RO 2010.
- Bhatia, A., Kavraki, L., Vardi, M. Sampling-based motion planning with temporal goals. ICRA 2010.
- Smith, S., Tumova, J., Belta, C., Rus, D. Optimal path planning for surveillance with temporal-logic constraints. IJRR 2011.
- Raman, V., Donzé, A. et al. Model predictive control with signal temporal logic specifications. CDC 2014.
- Lindemann, L., Dimarogonas, D. Control barrier functions for signal temporal logic tasks. IEEE L-CSS 2019.
- Lacerda, B., Faruq, F., Parker, D., Hawes, N. Probabilistic planning with formal performance guarantees for mobile service robots. IJRR 2019.

### 控制与可达性
- Nguyen, Q., Sreenath, K. Exponential control barrier functions for enforcing high relative-degree safety-critical constraints. ACC 2016.
- Wang, L., Ames, A., Egerstedt, M. Safety barrier certificates for collisions-free multirobot systems. IEEE T-RO 2017.
- Fisac, J. et al. A general safety framework for learning-based control in uncertain robotic systems. IEEE TAC 2019.
- Majumdar, A., Tedrake, R. Funnel libraries for real-time robust feedback motion planning. IJRR 2017.
- Nakamura, K., Peters, L., Bajcsy, A. Generalizing safety beyond collision-avoidance via latent-space reachability analysis. RSS 2025. https://arxiv.org/abs/2502.00935

### 人机交互与手术
- ISO/TS 15066:2016 Robots and robotic devices: Collaborative robots.
- Kouskoulas, Y. et al. Certifying the safe design of a virtual fixture control algorithm for a surgical robot. HSCC 2013.
- Askarpour, M. et al. SAFER-HRC: Safety analysis through formal verification in human-robot collaboration. SAFECOMP 2016.

### 学习型策略与人形安全（2025 到 2026，多数只读了摘要）
- Alshiekh, M. et al. Safe reinforcement learning via shielding. AAAI 2018. arXiv:1708.08611.
- Könighofer, B. et al. Shield synthesis. Formal Methods in System Design 2017.
- Safety-critical whole-body control for humanoid robots via input-to-state safe CBFs. https://arxiv.org/abs/2605.25546
- ConstrainedMimic: Constrained whole-body tracking for humanoid robots. https://arxiv.org/abs/2606.00374
- Filter-aware fine-tuning for safe humanoid whole-body tracking. https://arxiv.org/abs/2610.02341
- ResSafe: Learning safety filtering with residual RL for humanoids. https://arxiv.org/abs/2609.15988
- Active reachability learning for humanoid recoverability monitoring. https://arxiv.org/abs/2610.33317
- SPARK: Safe protective and assistive robot kit. https://arxiv.org/abs/2502.03132
- Adversarial stress testing of SPARK humanoid safety filters. https://arxiv.org/abs/2605.19009
- Shield-Loco: Shielding locomotion policies with predictive safety filtering. https://arxiv.org/abs/2606.07193
- Dexterous safe control for humanoids in cluttered environments via projected safe set algorithm. https://arxiv.org/abs/2502.02858

### 基础模型机器人
- Cosler, M. et al. nl2spec: Interactively translating unstructured natural language to temporal logics with large language models. CAV 2023.
- Liu, J. X. et al. Lang2LTL: Translating natural language commands to temporal robot task specification. CoRL 2023.
- Chen, Y. et al. AutoTAMP: Autoregressive task and motion planning with LLMs as translators and checkers. ICRA 2024.
- Ren, A. et al. Robots that ask for help: Uncertainty alignment for large language model planners (KnowNo). CoRL 2023.
- VerifyLLM: LLM-based pre-execution task plan verification for robots. https://arxiv.org/abs/2507.05118
- Safety guardrails for LLM-enabled robots (RoboGuard). https://arxiv.org/abs/2503.07885
- VLSA / AEGIS. https://arxiv.org/abs/2512.11891
- Attention-guided safety filter for VLA models. https://arxiv.org/abs/2606.09749
- Safe VLA via barrier enhanced flow matching. https://arxiv.org/abs/2607.29569
- Lindemann, L. et al. Formal verification and control with conformal prediction. 2024/2025.
- LLMs as planning formalizers: A survey. https://arxiv.org/abs/2503.18971
- Toward safe LLM agents: A survey of specification, verification, and enforcement. https://arxiv.org/abs/2608.14590

### 软件与系统
- Santos, A. et al. HAROS: The high-assurance ROS framework.
- Timperley, C. et al. ROSDiscover: Statically detecting run-time architecture misconfigurations in robotics systems. ICSA 2022.
- Ferrando, A. et al. ROSMonitoring: A runtime verification framework for ROS. TAROS 2020.
- Cavalcanti, A. et al. RoboChart / RoboStar 技术栈.
- NASA FRET; Copilot / Ogma; R2U2.
- Dennis, L., Fisher, M. 等, MCAPL / AJPF 智能体程序验证.
- Afzal, A. et al. Simulation for robotics test automation: Developer perspectives. ICST 2021.

### 标准与法规
- ISO 10218-1:2025, ISO 10218-2:2025. https://www.iso.org/standard/73934.html ; FAQ: https://www.automate.org/robotics/blogs/updated-iso-10218-faq
- ISO/CD 25785-1（动态稳定工业移动机器人）. https://www.iso.org/standard/91469.html
- ISO 13849-1:2023; IEC 61508; IEC 62061; ISO 13482:2014; ISO 3691-4:2023; IEC 80601-2-77.
- ANSI/A3 R15.06-2025; R15.08.
- RTCA DO-178C, DO-333; ASTM F3269.
- Regulation (EU) 2023/1230 on machinery products（2027-01-20 起适用）. 概述：https://www.ul.com/resources/navigating-new-eu-machinery-regulation-eu-20231230 ; https://www.bakermckenzie.com/en/insight/publications/resources/product-risk-radar-articles/machinery-regulation
