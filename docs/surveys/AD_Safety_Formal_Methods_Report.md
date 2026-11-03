# 自动驾驶安全中的形式化方法：综述报告

> 版本：2026-11-03
> 范围：自动驾驶（AD）安全中的规约、验证、运行时保障、形式化测试，以及在学习型/端到端系统中的应用
> 说明：标准与法规状态以检索当日公开信息为准；标注"厂商自述"的内容未经独立验证。参考文献见第 15 节。

---

## 0. 要点摘要

1. **形式化方法在 AD 中的定位已经清晰：它不是"证明整车安全"，而是"在明确假设下，对某一层给出可审查的保证"。** 整车安全最终靠安全论证（safety case），形式化方法提供其中证据强度最高的那几块。
2. **最成熟、最接近量产的形式化技术集中在三处：**
   - 软件层的静态分析和代码验证（ISO 26262 体系内）。
   - 基于可达性分析的在线验证与 fail-safe 轨迹。
   - 以 RSS 为代表的"可形式化的安全包络"（safety envelope）。
3. **规约是最大瓶颈，求解器不是。** 交通规则、"合理可预见的他车行为"（IEEE 2846）、ODD 都需要先形式化，才谈得上验证。STL 加鲁棒性语义是事实标准，LLM 辅助形式化是 2024 年以来的新趋势，但准确率还不够直接用。
4. **神经网络验证在开环、小网络上已经可用，但对感知和端到端驾驶策略仍远不可扩展。** 主流做法是：不验证网络本身，而用一个可验证的安全过滤器（safety filter / simplex / shield）把它包起来。
5. **端到端、VLA、世界模型时代的核心问题变了。** 问题从"证明控制器正确"变成了三件事：
   - 给黑盒策略套上有保证的运行时监控。
   - 把形式化规约用作训练和评估信号。
   - 对"学到的模型"本身的误差给出统计保证（例如 conformal prediction）。
   2026 年多篇工作明确指出：针对真实扰动的端到端驾驶策略形式化验证，基本仍是空白。
6. **法规侧正在从"测试里程"走向"安全论证加过程审计"。** 这一年的几个关键节点：
   - ISO/PAS 8800:2024（AI 安全）和 ISO/TS 5083:2025（ADS 设计验证确认）。
   - UN ADS 全球技术法规（GTR）与 UN 法规：GRVA 于 2026 年 1 月通过，WP.29 于 2026 年 6 月通过。
   - 中国首部 L2 强制性国标 GB 47955-2026（2027-01-01 实施）。
   这些都要求可追溯的证据链，这正是形式化方法能发挥价值的地方。

---

## 1. 为什么自动驾驶需要形式化方法

### 1.1 验证困境

- **里程论证不可行。** Kalra & Paddock（RAND, 2016）估算：要以统计显著性证明 AV 致死率优于人类，需要约 **数十亿英里** 的实际驾驶。纯靠路测，无法在合理时间内完成。
- **长尾与组合爆炸。** 场景参数空间（道路几何、交通参与者、天气、传感器状态）是高维连续的，危险事件是稀有事件。随机测试的覆盖率很难说清。
- **学习型组件不满足传统假设。** ISO 26262 基于"需求可完整写出、实现可追溯"的假设。DNN 感知和端到端策略的行为由数据决定，不满足这一点。由此产生了 SOTIF（ISO 21448）关注的"功能不足"（functional insufficiency），以及 ISO/PAS 8800 关注的 AI 输出不足。

### 1.2 形式化方法能提供什么

| 能力 | 含义 | 典型技术 |
|---|---|---|
| 穷尽性 | 对所有输入/初始状态/扰动给出结论，而不只是采样点 | 模型检测、可达性分析、定理证明 |
| 明确的假设 | 结论附带前提（动力学模型、他车行为界、感知误差界），可被审查和监控 | 假设-保证契约（assume-guarantee contracts） |
| 可量化的裕度 | STL 鲁棒度给出"离违规还有多远"，可用于测试导向和排序 | STL robustness、safety margin |
| 运行时可执行 | 把离线结论变成在线检查或过滤器，在假设成立时保证不违规 | online verification、CBF、HJ filter、shield |
| 反例 | 验证失败时给出具体反例场景，可直接回灌到测试和训练 | 证伪（falsification）、反例引导 |

### 1.3 形式化方法不能提供什么

- **规约缺口（specification gap）：** 证明只对写下来的性质成立。规约漏写，证明就漏掉。
- **模型缺口（model gap）：** 证明只对模型成立。车辆动力学、他车行为、传感器误差的模型偏差，会直接让保证失效。
- **感知缺口：** "图像里有没有行人"无法写成可验证的数学规约，这是端到端系统最难的地方。

因此业界共识（Seshia 等, CACM 2022；Waymo 2023 安全论证论文）是：形式化方法是安全论证中的**一类证据**，必须和仿真、路测、现场监控组合使用。

---

## 2. 标准与法规全景，以及形式化方法的切入点

### 2.1 国际标准

| 标准 | 状态 | 关注点 | 形式化方法的位置 |
|---|---|---|---|
| ISO 26262:2018 | 现行 | 功能安全（E/E 系统随机硬件故障与系统性故障） | Part 6 在较高 ASIL 下把形式化验证、静态分析列为推荐的软件验证方法（具体表格以原文为准）；工具鉴定（tool qualification）适用于验证工具本身 |
| ISO 21448:2022（SOTIF） | 现行 | 预期功能安全：功能不足与触发条件，"未知不安全"场景 | 场景空间探索、证伪、触发条件识别；可达性分析用于论证已知场景的安全 |
| ISO/PAS 8800:2024 | 2024-12 发布；ISO 页面显示待修订（复审日期 2026-02-10），拟议的 Part 2 已于 2026-03 撤销 | 道路车辆中 AI 元素的安全：输出不足、数据集、AI 生命周期 | 对 ML 组件的验证与确认要求；为 NN 验证、鲁棒性证据、运行时监控提供条款落点 |
| ISO/TS 5083:2025 | 2025-04 发布（技术规范） | L3/L4 ADS 的设计安全、验证确认、部署后活动，含网络安全考虑 | 系统级 V&V 的方法框架，形式化分析作为证据之一 |
| ISO 34502:2022 | 现行 | 基于场景的安全评估框架 | 场景抽象、参数空间、危险场景识别，与形式化场景语言对接 |
| ISO 34503:2023 | 现行 | ODD 分类与描述 | ODD 的结构化（可形式化）描述，是规约和运行时 ODD 监控的基础 |
| UL 4600 第 3 版（2023-03） | 现行 | 自主产品的安全论证评估，基于目标/声明 | 要求安全论证的完整性与可信度，形式化证据可作为高可信声明的支撑 |
| IEEE 2846-2022（及 2846a-2025 增补、IEEE 3321-2024 实施指南） | 现行 | 安全相关模型中关于他车"合理可预见行为"的最小假设集（以运动学界表达） | 直接给出可形式化的环境假设，是 RSS 类安全包络、可达集预测的输入 |
| ASAM OpenSCENARIO DSL 2.0 | 现行 | 场景描述语言 | 与 Scenic 类似的抽象场景规约，用于形式化测试 |

### 2.2 法规

| 法规 | 要点 |
|---|---|
| UNECE R157（ALKS） | 首个 L3 型式认证法规。2022 年修订后最高速度为 130 km/h，并允许变道。对"合理可预见且可预防"的碰撞给出参考驾驶员模型，这是一种隐式的安全包络规约 |
| EU 2022/1426 | 欧盟全自动车辆（小批量）ADS 型式认证，要求安全概念、场景覆盖和 in-service monitoring |
| UN ADS GTR 与 UN ADS 法规 | GRVA 于 2026-01 第 24 次会议通过草案。据 UNECE 秘书处说明和律所报道，WP.29 于 2026-06 第 199 次会议通过（AC.3 设立 GTR，AC.1 通过新的 UN 法规）。核心是"安全管理体系加安全论证加审计"的模式。细节以 UNECE 正式会议报告为准 |
| 中国 GB 47955-2026《智能网联汽车 组合驾驶辅助系统安全要求》 | 2026-06-27 批准发布，2027-01-01 实施，是首部 L2 强制性国标。分基础单车道、基础多车道、领航（NOA）三类；NOA 要求累计不少于 72 小时有效道路试验，并强制要求数据记录 |
| 中国 L3/L4 准入试点 | 2023-11 四部委启动准入和上路通行试点；地方条例（如北京）进一步规范 |

### 2.3 读法

法规和标准都在把"安全"从产品属性变成**过程加证据**。对形式化方法而言，这意味着三点：

1. 证据的**假设必须显式化**，并在运行中监控（UL 4600、5083、ADS GTR 都强调 in-service monitoring）。
2. **工具本身要可鉴定**（ISO 26262 tool confidence level），所以学术原型工具进不了量产流程，是一个现实障碍。
3. 对 AI 组件，标准目前要求的是**过程与论证**，没有规定具体的验证技术，给研究留了空间。

---

## 3. 一个分层框架：在哪一层、验证什么

自动驾驶的形式化方法可以按"被验证对象"分为五层：

| 层 | 对象 | 典型性质 | 主要技术 | 成熟度 |
|---|---|---|---|---|
| L1 软件/平台 | C/C++ 代码、RTOS、中间件 | 无运行时错误、内存安全、时序 | 抽象解释（Astrée、Polyspace）、有界模型检测（CBMC）、SPARK、seL4 | 高，已量产 |
| L2 决策逻辑 | 有限状态机、行为规划器、模式切换 | 无死锁、模式切换正确、规则遵守 | 模型检测（nuXmv、UPPAAL、SPIN）、TLA+ | 中高 |
| L3 连续控制与运动规划 | 规划轨迹加车辆动力学 | 无碰撞、保持车道、不可达危险集合 | 可达性分析（CORA、HJ）、微分动态逻辑（KeYmaera X）、CBF | 中，部分在线可用 |
| L4 学习组件 | 感知 DNN、学习型策略、世界模型 | 局部鲁棒性、闭环可达性、感知契约 | NN 验证（Marabou、α,β-CROWN）、闭环 NN 验证（Verisig、NNV、POLAR）、统计保证 | 低到中，规模受限 |
| L5 系统与场景 | 整车在 ODD 中 | 场景覆盖、交通规则遵守、风险率 | 形式化场景语言、证伪、统计模型检测、稀有事件模拟、安全论证 | 中，工程上广泛使用但保证弱 |

**关键观察：越往上层，保证越弱，越需要统计方法和架构手段补位。** 当前研究的重心正从 L3 向 L4/L5 移动。

---

## 4. 规约：先把"安全"写成数学

### 4.1 时序逻辑

- **LTL**（线性时序逻辑）：离散时间，适合决策逻辑，例如"总是在红灯前停车"`G(red → F stop)`。
- **MTL / STL**（度量/信号时序逻辑）：带时间区间，作用在实值信号上，是 AD 规约的事实标准。
  - 语法：`φ ::= μ | ¬φ | φ ∧ φ | φ U_[a,b] φ`，派生 `F_[a,b]`（eventually）和 `G_[a,b]`（always）。
  - **鲁棒性语义**（Donzé & Maler 2010；Fainekos & Pappas 2009）：
    - 谓词 `μ: f(x) ≥ 0`：鲁棒度 `ρ = f(x(t))`。
    - `G`：取区间内最小值；`F`：取最大值；`U`：取 max-min 组合。
    - `ρ > 0` 表示满足，`|ρ|` 是到违规的距离。这是证伪、优化、学习中最常用的标量信号。
- **示例：跟车安全距离（RSS 纵向）：**

  ```
  φ_lon = G_[0,T] ( d_lon(t) ≥ d_min(v_r(t), v_f(t)) )

  d_min = [ v_r·ρ + ½·a_acc·ρ² + (v_r + ρ·a_acc)² / (2·a_brake_min) − v_f² / (2·a_brake_max) ]_+
  ```

  其中 `ρ` 为反应时间，`v_r`/`v_f` 为后车/前车速度。
- **扩展：**
  - STL 的参数挖掘（parametric STL，从数据中拟合阈值）。
  - 时空逻辑（如 SpaTeL）。
  - 带概率的时序逻辑：PCTL/CSL 用于概率模型检测；Chance-constrained STL 用于不确定环境。

### 4.2 交通规则形式化

| 工作 | 内容 |
|---|---|
| Rizaldi 等（iFM 2017），Isabelle/HOL | 德国 StVO 部分规则（超车、安全距离）的形式化与监控 |
| Maierhofer 等（IV 2020） | 德国高速公路规则的 MTL/STL 形式化，集成于 CommonRoad 规则监控 |
| Esterle 等（2020） | 交通规则 LTL 形式化与在线监控 |
| Rulebooks（Censi 等, ICRA 2019，Motional） | 规则之间存在冲突时（如"不压实线"与"不撞人"），用**偏序规则簿**表达优先级与违规代价；后续工作做规则簿的验证与评估 |
| RSS（Shalev-Shwartz 等, 2017，Mobileye） | 把"不承担责任的安全"写成纵向/横向安全距离、危险情况定义、正确响应和优先通行权等闭式规则；IEEE 2846 为其所依赖的他车假设提供标准化框架 |
| NVIDIA Safety Force Field（2019） | 基于"安全程序"和场的势函数约束，与 RSS 思路相近 |

**LLM 辅助形式化（2024 起）：**
- TR2MTL（Manas 等, IV 2024）：LLM 把交通规则翻译为 MTL，设计为人在环系统。
- CoT-TL（同组）：翻译为 LTL，并用模型检测做语法检查。
- AutoLegal：知识增强的 LLM 加 STL，用语法校验和语义回译来检查结果，报告的翻译成功率约 74%。

结论：**LLM 适合生成候选形式化，不能替代人工审核。** 合理的流程是：LLM 起草，语法/语义检查，人工确认，最后在真实轨迹上回归测试。

### 4.3 环境假设与 ODD

- **他车行为假设：** IEEE 2846 用运动学界（最大加速度、最大减速度、反应时间等）表达"合理可预见"。可达集预测（如 SPOT）把这些界当作输入。
- **ODD 规约：** ISO 34503 给出分类。形式化后可用于两件事：
  - **运行时 ODD 监控：** 出了 ODD 就触发最小风险策略（MRM）。
  - **场景采样边界：** 测试只在 ODD 内采样。
- **假设-保证契约**（Nuzzo、Sangiovanni-Vincentelli 等）：组件 i 的保证 G_i 在假设 A_i 下成立，系统级性质通过契约代数组合得到。这是把"感知误差界"与"规划安全"连接起来的理论工具（见 5.4 感知契约）。

---

## 5. 设计时（离线）验证

### 5.1 模型检测

- **适用对象：** 离散决策逻辑、模式切换、通信协议、时间自动机。
- **工具：** nuXmv/NuSMV（符号模型检测）、SPIN（显式状态）、UPPAAL（时间自动机）、TLA+（分布式协议）。
- **概率模型检测：** PRISM、Storm。把交互建模为 MDP 或 DTMC，计算"在策略 π 下，T 秒内发生碰撞的最大概率"，常用于决策层的风险量化。
- **局限：** 需要先把连续动力学抽象为离散模型，抽象是否保守，需要另外论证。

### 5.2 定理证明

- **微分动态逻辑 dL 与 KeYmaera X**（Platzer）：直接对混成系统（离散控制加连续 ODE）做演绎证明。
  - 代表工作：Mitsch 等（IJRR 2017），地面机器人避障的可证明安全控制。
  - ModelPlex：从证明自动生成运行时监控器，把"模型与现实不符"变成可检测事件。这是**把证明的前提变成运行时检查**的典范。
- **Isabelle/HOL、Coq：** 用于交通规则、几何算法、运动原语的形式化。
- **局限：** 证明需要专家投入。适合小而关键的核心（安全包络、紧急制动逻辑），不适合整个规划栈。

### 5.3 可达性分析

**(a) 集合传播（set-based）：**
- 用 zonotope、多面体、Taylor 模型等集合表示，过近似地传播系统所有可能状态。
- 工具：CORA（MATLAB，TUM Althoff 组）、JuliaReach、Flow*、SpaceEx。
- AD 中的应用：
  - 对他车做**占据集预测**（SPOT：基于交通规则和物理约束，过近似他车未来可能占据的区域）。
  - 对自车规划轨迹做跟踪误差的可达集，验证"计划轨迹 ⊕ 跟踪误差"与他车占据集无交。

**(b) Hamilton-Jacobi（HJ）可达性：**
- 把安全问题写成微分博弈：控制 `u` 尽力保持安全，扰动 `d`（他车、模型误差）尽力造成违规。
- 值函数 `V(x,t)` 满足 HJI 变分不等式：

  ```
  min{ ∂V/∂t + max_u min_d ∇V·f(x,u,d),  l(x) − V(x,t) } = 0
  ```

  其中 `l(x) ≥ 0` 表示安全。`{V ≥ 0}` 是可保持安全的集合，同时得到最优安全控制。
- 精确求解受维数灾难限制（网格法实际 ≤ 5 到 6 维）。可扩展方向：
  - 系统分解。
  - **DeepReach**（Bansal & Tomlin, ICRA 2021）：用神经网络近似值函数。
  - 基于 RL 的近似（如折扣安全 Bellman 方程）。
  这些方法牺牲了严格性，需要额外的验证步骤（例如对学到的值函数做 conformal 或场景采样校验）。
- 综述：Bansal 等（CDC 2017）。

### 5.4 神经网络验证

**(a) 开环（单网络性质）：**
- 问题：给定输入集 X（如图像的 L∞ 球），证明输出满足性质 Y。精确验证是 NP 难的（Katz 等, Reluplex, CAV 2017）。
- 方法：
  - 完备方法：SMT/MILP，如 Reluplex、Marabou。
  - 不完备方法：线性松弛与界传播，如 DeepPoly/ERAN、CROWN。
  - 两者结合：分支定界，如 α,β-CROWN，是近年 VNN-COMP 的领先工具。
- **VNN-COMP 2025**（第 6 届，与 CAV 2025 同期举办）：8 支队伍、16 个常规基准加 9 个扩展基准，统一使用 ONNX 和 VNN-LIB 格式，今年取消了时间奖励。
- 现状：对中等规模的 CNN 或 Transformer 的局部鲁棒性可以验证，但"局部 L∞ 鲁棒"与"驾驶安全"之间语义差距很大。

**(b) 闭环（NN 控制器加动力学）：**
- 问题：NN 策略与车辆动力学组成闭环，验证在 N 步内不进入不安全集合。
- 工具：
  - Verisig（Ivanov 等, HSCC 2019）：sigmoid 网络转为混成系统。
  - NNV（Tran 等, CAV 2020）：star set。
  - POLAR（Huang 等, ATVA 2022）：多项式算术。
  - 基准：ARCH-COMP 的 AINNCS 类别。
- 局限：只适用于状态维度低、网络小、时域短的问题，例如 ACC 和车道保持的小型 NN 控制器。

**(c) 感知在环（最难）：**
- 无法写出"所有合法相机图像"的规约，只能用替代手段：
  - **生成模型作为输入空间的规约：** Katz 等（2021/2022）用 GAN/VAE 的潜空间参数化"合理图像"，在潜空间加网络上做闭环验证。
  - **感知契约（perception contracts）：** Hsieh、Mitra 等，Astorga 等（OOPSLA 2023）。把感知模块抽象为"真实状态 → 估计状态"的误差集合，用数据估计并校准这个集合，再在抽象上验证控制，从而把感知的不确定性转化为可验证的有界扰动。
  - **校准的安全概率预测：** 例如 2025 年关于图像控制自主系统"看到什么就预测多安全"的工作，用 conformal 方法给出校准的安全概率。

### 5.5 软件层（L1）

- 抽象解释：Astrée（无运行时错误证明，有航空和汽车案例）、Polyspace Code Prover。
- 有界模型检测：CBMC、ESBMC。
- 安全语言与验证：SPARK/Ada、Rust（加 Kani、Prusti）。
- 形式化验证的微内核：seL4（已有汽车/航空项目使用）。
- 这一层的技术成熟、可鉴定，与 ISO 26262 结合最紧，**在工程上最容易产生立竿见影的价值**。

---

## 6. 运行时保障（online assurance）

离线无法验证整个学习型栈，运行时保障是目前最务实的路径：**被验证的不是复杂策略本身，而是一个简单、可验证的守护层。**

### 6.1 架构范式

| 范式 | 机制 | 保证 | 代表 |
|---|---|---|---|
| Simplex / RTA | 高性能控制器（HPC，不可验证）加经过验证的安全控制器（SC）加切换逻辑 | 只要切换条件正确，系统停留在 SC 的可恢复集合内 | Sha 2001；Black-box Simplex（Mehmood 等, NFM 2022） |
| 在线验证 + fail-safe 轨迹 | 每个规划周期验证"新计划 + 紧急机动"对所有合法他车行为都无碰撞；验证失败则执行上一周期已验证的 fail-safe | 在他车遵守交通规则和物理界、感知正确的假设下，自车不会发生**自车有责**的碰撞 | Althoff & Dolan（T-RO 2014）；Pek 等（Nature Machine Intelligence 2020） |
| CBF 安全过滤器 | 在 QP 中最小修改名义控制：`min ‖u − u_nom‖²  s.t.  L_f h + L_g h·u + α(h) ≥ 0` | 若 h 是有效 CBF 且模型正确，安全集合前向不变 | Ames 等（ECC 2019 综述） |
| HJ 安全过滤器 | 在 `V(x) > ε` 时用名义控制，接近边界时切换到最优安全控制（或用 `∇V` 做 QP） | 与 HJ 值函数的精度绑定 | Bansal 等；Hsu、Hu、Fisac（2023 统一视角） |
| Shield（屏蔽） | 从时序逻辑规约合成离散"盾"，过滤 RL 策略的不安全动作 | 对抽象 MDP 上的规约给出保证 | Alshiekh 等（AAAI 2018） |
| 安全包络（RSS 类） | 检测"危险情况"，强制执行正确响应（按规则制动、让行） | 在 IEEE 2846 式假设下无责 | Mobileye RSS；IEEE 2846 |

**统一视角：** Hsu、Hu、Fisac（2023, *The Safety Filter: A Unified View*）指出，上述方法都可以看作同一结构的实例：**一个安全集合（或其指示函数）加一个回退策略加一个干预规则**。差别只在于安全集合怎么算（解析 CBF、HJ、可达集、学习得到）以及保证的强度。

### 6.2 运行时监控

- **STL 在线监控：** RTAMT（Nickovic & Yamaguchi, ATVA 2020）、Breach、S-TaLiRo 的在线模式。可计算部分轨迹的鲁棒度区间，提前预警。
- **假设监控：** 把离线证明的前提（他车加速度界、定位误差界、ODD 条件）做成在线检查，例如 ModelPlex 生成的监控器。**一个证明的价值等于其假设被监控的程度。**
- **OOD 与不确定性监控：** 感知的置信度、集成方差、特征空间的马氏距离等。缺点是只是启发式，没有保证。
- **Conformal prediction：**
  - Lindemann 等（RA-L 2023）：用 conformal 方法为轨迹预测误差给出有限样本、无分布假设的置信区域，并用于安全规划。
  - Lindemann、Qin、Deshmukh、Pappas（ICCPS 2023）：把 STL 运行时验证与 conformal 结合，给出"未来将满足 φ"的概率保证。
  - 这是**把统计保证接入形式化框架的主流桥梁**。前提是可交换性，在分布漂移下需要自适应变体（adaptive conformal inference）。

### 6.3 运行时保障的代价

- **保守性：** 过近似导致频繁干预，影响通行效率（Pek 2020 中在真实数据上评估了这一点，代价可接受，但在密集交互场景下会变大）。
- **交互性缺失：** 把他车当作最坏情况的对手，会导致"冻结机器人"问题。博弈论和交互感知的可达性是研究前沿。
- **假设脆弱：** 感知漏检会让所有在线验证失效。因此感知层需要独立的冗余与监控。

---

## 7. 形式化测试与证伪

无法证明时，退而求其次：**用规约指导的搜索去找反例**，并对"找不到反例"给出统计意义上的论断。

### 7.1 证伪（falsification）

- 问题：`min_{θ ∈ Θ} ρ(φ, sim(θ))`，在场景参数空间 Θ 上最小化 STL 鲁棒度，`ρ < 0` 即反例。
- 工具：
  - S-TaLiRo（Annpureddy 等, TACAS 2011）、Breach（Donzé, CAV 2010）。
  - **VerifAI**（Dreossi 等, CAV 2019）：面向 AI 系统，内置多种采样器和优化器。
- 优化器：随机采样、交叉熵方法、贝叶斯优化、模拟退火、强化学习。
- **自适应压力测试（AST）**（Lee、Koren、Kochenderfer 等, 2018 起）：把找失败建模为 MDP，用 MCTS 或 RL 搜索最可能的失败轨迹，同时考虑失败的**似然**，而不只是是否失败。
- 综述：Corso 等，*A Survey of Algorithms for Black-Box Safety Validation of Cyber-Physical Systems*（JAIR 2021）。

### 7.2 形式化场景语言

- **Scenic**（Fremont 等, PLDI 2019；Scenic 3, 2023）：概率编程语言，用声明式约束描述场景分布（"一辆车在自车前方 10 到 20 m 的车道上"），可直接驱动 CARLA 等仿真器，与 VerifAI 联用做证伪。
- 从仿真到真实：Fremont 等（ITSC 2020），用 Scenic 加 VerifAI 在仿真中找出失败场景，再在封闭测试场上复现，验证了仿真反例的可迁移性。
- ASAM OpenSCENARIO DSL 2.0：工业界的抽象场景语言，与 ISO 34502 框架对接。

### 7.3 稀有事件与统计保证

- **重要性采样：** 将采样分布偏向危险区域，再用似然比修正，得到无偏的风险率估计（Zhao 等, 2017 起）。
- **密集强化学习（D2RL）：** Feng 等（Nature 2023）。只在"关键"状态上施加对抗性的背景车行为，其余时间按自然分布，显著加速 AV 安全评估，同时保持无偏性。
- **统计模型检测（SMC）：** 用顺序假设检验（SPRT）或 Chernoff 界，以给定置信度判断 `P(φ) ≥ p`。比穷尽方法可扩展得多，但保证是概率性的，且依赖于仿真分布的代表性。
- **覆盖度量：** 场景空间覆盖、STL 子公式覆盖、神经元覆盖（后者与安全的关系存疑）。

### 7.4 生成式世界模型与场景生成（2024 到 2026）

- 扩散模型、自回归模型和 BEV 世界模型被用来生成对抗或长尾场景，例如 CounterScene（arXiv 2603.21104, 2026）在闭环生成式 BEV 世界模型中加入反事实推理来生成安全关键场景。
- **核心风险：** 世界模型的幻觉与误差累积。在世界模型中发现的"失败"可能是模型伪影；反过来，世界模型漏掉的危险也不会被发现。**需要在高保真仿真或实车上复核，并度量世界模型自身预测"失败"的校准程度。**（见第 13 节研究方向 A。）

---

## 8. 概率与统计保证：形式化方法的"软化"

当对象是 DNN 或数据驱动模型时，确定性证明通常不可得。主流的替代形式如下：

| 保证形式 | 表达 | 前提 | 用途 |
|---|---|---|---|
| PAC / 场景优化（scenario approach） | 以置信度 1−β，违规概率 ≤ ε | i.i.d. 样本 | 学习得到的安全集合、控制器的认证 |
| Conformal prediction | 以覆盖率 1−α，真实值落在预测集合内 | 可交换性 | 预测误差界、感知误差界、STL 满足概率 |
| 统计模型检测 | `P(φ) ≥ p` 的假设检验 | 仿真分布代表真实 | 系统级风险估计 |
| 贝叶斯 / 重要性采样风险率 | 后验或无偏估计的事故率 | 先验与模型正确 | 发布决策（如 Waymo 的"无不合理风险"判断） |
| 概率模型检测 | MDP/DTMC 上的精确概率 | 抽象模型正确 | 决策层风险 |

**关键原则：** 统计保证的前提（i.i.d.、可交换、仿真代表性）本身就是需要监控的假设。分布漂移检测与自适应 conformal 是保证"延续到部署中"的关键。

---

## 9. 面向学习型与端到端系统

### 9.1 新问题

端到端（E2E）驾驶（UniAD、VAD 类）、VLA 驾驶模型（DriveVLM、EMMA、Alpamayo 类）和世界模型驱动的规划，带来三个变化：

1. **没有显式的中间表示：** 无法单独验证"感知输出的目标列表"，安全规约只能作用在最终轨迹或动作上。
2. **规模：** 数亿到数十亿参数，任何完备的 NN 验证都不可行。
3. **语言与推理：** VLA 输出的"解释"不能当作安全论证，因为它与实际行为的一致性未经验证。

2026 年的文献对此有共识：一篇关于端到端转向的工作（arXiv 2609.10951）指出，针对车辆实际会遇到的扰动去验证端到端驾驶策略，基本尚未被研究，**限制适用性的是扰动模型，而不是求解器**。ICCV 2025 workshop 的 VLA 驾驶综述也把形式化验证列为开放挑战，认为 SafeAuto 这类逻辑否决只是第一步。

### 9.2 现有路线

| 路线 | 做法 | 代表工作/方向 | 评价 |
|---|---|---|---|
| **外挂安全过滤器** | 在 E2E 或 VLA 的输出上加 CBF、可达性或 RSS 过滤 | 可微 CBF 用于视觉端到端驾驶（arXiv 2203.02401）；VLA 上的 CBF 安全约束层（VLSA/AEGIS, arXiv 2512.11891，偏操作任务）；动作块上的屏障增强 flow matching（arXiv 2607.29569） | 最务实。保证取决于过滤器所用状态估计的正确性，也就是感知问题被推回到了过滤器 |
| **潜空间安全过滤器** | 在生成式世界模型的潜空间里做 HJ 可达性，约束由潜空间分类器给出 | Nakamura、Peters、Bajcsy（RSS 2025, arXiv 2502.00935），以及 2025 年的不确定性感知版本和部分可观测性分析 | 能处理难以手写的危险（如"打翻东西"）。保证依赖世界模型和分类器的准确性，部分可观测时会"短视"。目前主要在机械臂上验证，**迁移到驾驶是空白** |
| **形式化规约作为训练信号** | 用 STL 鲁棒度作可微损失或奖励；规则条件化的模仿学习 | STL-guided RL；rulebook 作为代价；规则约束的扩散规划 | 提高合规性，但不提供保证，仍需评估层验证 |
| **形式化规约作为评估器** | 用 STL 监控器和规则簿对 E2E 输出做闭环评估，替代或补充 L2 误差和 PDMS 类指标 | CommonRoad 规则监控；nuPlan/NAVSIM 中的规则化评分 | 低成本、高价值，是工程上最容易落地的一步 |
| **学到的模型的统计保证** | 对世界模型和预测器的误差给出 conformal 界，再在界上做可达性 | conformal + 可达性；学习型安全过滤器 + 自适应 conformal（arXiv 2604.18482） | 把"模型缺口"量化并纳入保证，是理论上最干净的方向 |
| **LLM 辅助规约与解释** | LLM 生成 STL/LTL，或对场景做规则级解释 | TR2MTL、CoT-TL、AutoLegal | 降低形式化门槛，需人工确认 |

### 9.3 工业实践（公开信息）

- **Waymo：** *Building a Credible Case for Safety*（Favaro 等, 2023, arXiv 2306.01917），以"无不合理风险"为顶层目标，提出分层、动态、可信三个视角，以及 Case Credibility Assessment。Waymo 称 2025-11 TÜV SÜD 审计了其安全论证流程（厂商自述）。
- **Mobileye：** RSS 作为可形式化的安全包络，参与推动 IEEE 2846。
- **NVIDIA：** 2025-03 发布 Halos 全栈安全体系，覆盖平台（DriveOS 6.0 声称符合 ISO 26262 ASIL D）、算法和生态（安全数据集、自动化安全评估、经 ANAB 认证的 AI 系统检测实验室）。以上为厂商自述，未见独立评估。
- **Mercedes-Benz Drive Pilot：** 基于 UNECE R157 的 L3 认证，显示了"有限 ODD + 明确假设 + 冗余"的路线可以通过监管。
- **共同模式：** 量产系统极少声称"形式化验证了 AI"，而是用**形式化的安全包络或规则监控 + 冗余架构 + 统计验证 + 安全论证**组合起来满足标准。

---

## 10. 工具链速查

| 类别 | 工具 | 说明 |
|---|---|---|
| 模型检测 | nuXmv, SPIN, UPPAAL, TLA+ | 离散/时间自动机/协议 |
| 概率模型检测 | PRISM, Storm | MDP/DTMC 风险 |
| 混成系统定理证明 | KeYmaera X | dL，带 ModelPlex 监控器生成 |
| 通用证明助手 | Isabelle/HOL, Coq, Lean | 规则、几何、算法正确性 |
| 集合可达性 | CORA, JuliaReach, Flow*, SpaceEx | zonotope、Taylor 模型 |
| HJ 可达性 | helperOC / hj_reachability（JAX）, DeepReach | 网格法 / 神经近似 |
| 交通场景与规则 | CommonRoad（场景、SPOT 预测、规则监控、可达性） | TUM Althoff 组生态 |
| NN 验证 | α,β-CROWN, Marabou, ERAN, NNV, PyRAT 等 | VNN-COMP 参赛工具 |
| 闭环 NN 验证 | Verisig, NNV, POLAR | ARCH-COMP AINNCS |
| STL 监控与证伪 | RTAMT, Breach, S-TaLiRo, PSY-TaLiRo | 在线/离线鲁棒度 |
| 场景与证伪框架 | Scenic, VerifAI, OpenSCENARIO DSL | 场景分布、采样、证伪 |
| 黑盒验证 | AST Toolbox, POMDPStressTesting.jl | 自适应压力测试 |
| 代码级 | Astrée, Polyspace, CBMC, Frama-C, Kani | 无运行时错误、内存安全 |
| 安全论证 | GSN 编辑器（如 Astah GSN）, AdvoCATE | 论证结构化 |

---

## 11. 开放问题

1. **规约完备性：** 如何论证一组 STL 规则"足够"刻画安全？交通规则存在冲突和隐含常识（如"合理"让行），rulebook 只解决了优先级，没有解决完备性。
2. **模型缺口的量化：** 动力学、他车行为、传感器模型的偏差如何进入保证？conformal 与在线假设监控是方向，但对分布漂移和对抗性情况仍然脆弱。
3. **感知的可验证规约：** "图像中的行人被正确检测"无法直接形式化。感知契约、生成模型潜空间和校准的不确定性是三条路线，都还不能扩展到全场景。
4. **可扩展性：** 闭环 NN 验证仍停留在低维小网络；HJ 方法受维数灾难限制；神经近似牺牲了严格性。
5. **交互与保守性：** 最坏情况假设导致过度保守。如何在保证安全的同时允许合理的交互博弈（并道、无保护左转），是可达性和安全过滤器的核心难题。
6. **端到端与 VLA：** 没有中间表示，只能在输出端加过滤器，而过滤器本身依赖的状态估计又回到了感知问题。潜空间安全过滤器是有前景的方向，但在驾驶上尚无系统性工作。
7. **统计与形式化证据的组合：** 安全论证中如何把"在假设 A 下可证安全"与"以 95% 置信度假设 A 成立"严谨地组合成系统级风险界？
8. **工具鉴定与监管接受度：** 学术工具难以通过 ISO 26262 工具鉴定；监管者对"概率保证"和"学到的安全集合"的接受程度尚不明确。
9. **学到的模型何时"知道"自己在哪里错：** 世界模型或评估器对策略弱点的预测是否校准？这直接决定基于模型的证伪和运行时预警是否可信。

---

## 12. 方法对比总表

| 方法 | 保证类型 | 主要假设 | 可扩展性 | 保守性 | 在线可用 | 适合的 AD 层 |
|---|---|---|---|---|---|---|
| 模型检测 | 穷尽（抽象模型上） | 抽象正确 | 中（状态爆炸） | 依抽象而定 | 否 | 决策逻辑 |
| 定理证明（dL） | 演绎证明 | 混成模型正确 | 低（需专家） | 低 | 通过 ModelPlex 监控 | 安全核心 |
| 集合可达性 | 过近似、可靠 | 动力学与扰动界 | 中 | 中到高 | 是（Pek 2020） | 运动规划 |
| HJ 可达性 | 最优安全集合 | 动力学与扰动界 | 低维（≤ 6）；神经近似可扩展但不严格 | 低到中 | 查表/网络 | 低维交互 |
| CBF | 前向不变 | 有效 CBF、模型正确 | 高（QP） | 依 h 设计 | 是 | 控制层 |
| NN 验证（开环） | 局部性质 | 输入集合有意义 | 中 | 不完备时保守 | 否 | 感知组件 |
| 闭环 NN 验证 | 有界时域可达 | 动力学与 NN 精确 | 低 | 中 | 否 | 小型 NN 控制器 |
| 证伪 | 找反例（无完备性） | 仿真保真度 | 高 | 不适用 | 否 | 系统/场景 |
| 统计模型检测 | 概率性 | 采样分布代表真实 | 高 | 不适用 | 否 | 系统 |
| Conformal | 有限样本覆盖 | 可交换 | 高 | 低到中 | 是 | 预测/感知 |
| 潜空间安全过滤器 | 近似（依世界模型） | 世界模型与约束分类器准确 | 高 | 依学习质量而定 | 是 | 端到端/视觉策略 |

---

## 13. 研究方向建议

以下方向结合了**形式化方法的开放问题**与**可微世界模型、端到端策略、规则条件化评估**等已有积累，按"新颖性 × 可行性"排序。

### A. 世界模型是否知道策略在哪里脆弱：可微世界模型驱动的证伪及其校准

- **问题：** 第 7.4 节和第 11 节第 9 点。基于世界模型的场景生成越来越多，但几乎没有工作度量**世界模型预测的失败与真实（高保真仿真或实车）失败之间的校准程度**。
- **做法：**
  1. 在可微的学习型驾驶世界模型中，对场景参数、他车动作或初始状态求 STL 鲁棒度的梯度，做梯度导向的证伪（与 SuperMimic `--kind adv` 的思路同构：最坏方向 vs 随机方向）。
  2. 在高保真仿真中复核每个反例，统计 agree / missed / false alarm 三类。
  3. 用 conformal 方法把"世界模型预测的安全裕度"校准为有覆盖保证的区间。
- **贡献：** 把"基于模型的证伪"从启发式变成有统计保证的方法，并给出何时可以信任世界模型的诊断指标。
- **风险：** 梯度在接触或离散事件处不稳定，需要平滑或随机化处理。

### B. 驾驶中的潜空间可达性安全过滤器

- **问题：** 潜空间 HJ 过滤器（RSS 2025）在机械臂上验证过，在驾驶这种多智能体、部分可观测的场景中尚无系统性工作。
- **做法：**
  - 在 BEV 或向量化表示的世界模型潜空间中学习安全值函数，约束由规则监控器（STL）自动标注，而不是手工分类器。
  - 用 conformal 方法对值函数的符号误判率给出界。
  - 作为端到端策略的外挂过滤器，在 nuPlan、NAVSIM 或 CARLA 闭环中评估干预率与安全提升。
- **贡献：** 第一个面向端到端驾驶、有统计保证的潜空间安全过滤器。

### C. 规则条件化的神经符号评估器

- **问题：** LLM 或学习型"驾驶评判器"缺乏可审查性，STL 监控器缺乏常识和覆盖面。
- **做法：**
  - 用 STL 或规则簿监控器作为可验证的锚点，LLM 负责把自然语言规则形式化（人工审核），并处理监控器无法表达的情境判断。
  - 用监控器的鲁棒度作为监督或一致性检查信号。
  - 度量 LLM 判断与形式化判断的一致性，并定位分歧。
- **贡献：** 可审查的驾驶行为评估，可直接服务于 GB 47955 / ADS GTR 所要求的证据链。

### D. 假设监控驱动的安全论证

- **问题：** 第 11 节第 7 点。形式化证明的前提（IEEE 2846 运动学界、感知误差界）在部署中是否成立，往往没有被系统地监控。
- **做法：**
  - 从可达性或 RSS 的证明中自动提取假设。
  - 在真实数据（如 Waymo Open、nuPlan）上用 conformal 或极值理论估计假设被违反的频率。
  - 把"有条件保证"与"假设违反率"组合成系统级风险上界，并映射到 GSN 安全论证中。
- **贡献：** 形式化证据与统计证据的严谨组合，是监管和标准都在追问的问题。

### E. 端到端策略的有界扰动验证

- **问题：** 第 9.1 节引用的 2026 工作指出，瓶颈在扰动模型。
- **做法：**
  - 定义物理上有意义的扰动族（他车轨迹扰动、传感器延迟、定位噪声），而不是像素级 L∞ 扰动。
  - 在向量化输入的端到端规划器上，用 α,β-CROWN 类工具在这些语义扰动下给出局部保证。
  - 用证伪工具作为上界参照。
- **贡献：** 介于"像素鲁棒性"与"闭环安全"之间的、有意义的中间层保证。

---

## 14. 阅读路线

**入门（建立全局观）**
1. Seshia, Sadigh, Sastry, *Toward Verified Artificial Intelligence*, CACM 2022：可验证 AI 的五大挑战。
2. Koopman, *How Safe Is Safe Enough?*（2022）：安全论证、度量与标准，工程视角。
3. Hsu, Hu, Fisac, *The Safety Filter: A Unified View of Safety-Critical Control in Autonomous Systems*（2023/2024）：运行时保障的统一框架。
4. Corso 等, *A Survey of Algorithms for Black-Box Safety Validation of CPS*（JAIR 2021）：证伪、AST、稀有事件。

**核心技术**
5. Donzé & Maler（2010）；Fainekos & Pappas（2009）：STL 鲁棒性。
6. Pek 等（Nature Machine Intelligence 2020）：在线验证与 fail-safe 轨迹，这一领域最有代表性的端到端案例。
7. Shalev-Shwartz 等（2017）RSS 加 IEEE 2846：可形式化的安全包络。
8. Ames 等（ECC 2019）CBF 综述；Bansal 等（CDC 2017）HJ 综述。
9. Liu 等, *Algorithms for Verifying Deep Neural Networks*（Foundations and Trends in Optimization 2021）：NN 验证全景。
10. Fremont 等, Scenic（PLDI 2019）加 ITSC 2020 仿真到实车。

**前沿**
11. Lindemann 等, conformal prediction 用于安全规划与 STL（2023）。
12. Feng 等, D2RL（Nature 2023）。
13. Nakamura 等, latent-space reachability（RSS 2025）。
14. Favaro 等, Waymo 安全论证（2023）。
15. Burgio, Ferrando, Villani, *Open Challenges in the Formal Verification of Autonomous Driving*（FMAS 2024）。

**教材**
- Baier & Katoen, *Principles of Model Checking*（2008）。
- Platzer, *Logical Foundations of Cyber-Physical Systems*（2018）。
- Mitra, *Verifying Cyber-Physical Systems*（2021）。
- Kochenderfer 等, *Algorithms for Validation*（验证算法教材，书稿公开）。

---

## 15. 参考文献

### 标准与法规
- ISO 26262:2018, *Road vehicles: Functional safety*.
- ISO 21448:2022, *Road vehicles: Safety of the intended functionality*.
- ISO/PAS 8800:2024, *Road vehicles: Safety and artificial intelligence*. https://www.iso.org/standard/83303.html
- ISO/TS 5083:2025, *Road vehicles: Safety for automated driving systems: Design, verification and validation*. https://www.iso.org/standard/81920.html
- ISO 34502:2022（场景安全评估）; ISO 34503:2023（ODD 分类）.
- UL 4600 Ed. 3（2023-03）. https://www.ul.com/news/ul-4600-edition-3-updates-incorporate-autonomous-trucking
- IEEE 2846-2022, *Assumptions in Safety-Related Models for Automated Driving Systems*. https://standards.ieee.org/ieee/2846/10831/
- UNECE R157（ALKS）; EU 2022/1426.
- UN ADS GTR / UN Regulation 现状：https://www.insideglobaltech.com/2026/05/06/un-regulation-and-gtr-on-automated-driving-systems-current-state-of-play/ ; https://cms.law/de/deu/legal-updates/un-regelwerk-fuer-autonomes-fahren-neue-vorgaben-fuer-ads
- GB 47955-2026《智能网联汽车 组合驾驶辅助系统安全要求》. 新华网报道：https://www.news.cn/sci-tech/20260703/1973b77e2855440ea5aea889f164c988/c.html

### 规约与交通规则
- Donzé, A., Maler, O. Robust satisfaction of temporal logic over real-valued signals. FORMATS 2010.
- Fainekos, G., Pappas, G. Robustness of temporal logic specifications for continuous-time signals. TCS 2009.
- Shalev-Shwartz, S., Shammah, S., Shashua, A. On a formal model of safe and scalable self-driving cars. arXiv:1708.06374, 2017.
- Censi, A. et al. Liability, ethics, and culture-aware behavior specification using rulebooks. ICRA 2019. arXiv:1902.09355.
- Rizaldi, A. et al. Formalising and monitoring traffic rules for autonomous vehicles in Isabelle/HOL. iFM 2017.
- Maierhofer, S. et al. Formalization of interstate traffic rules in temporal logic. IV 2020.
- Manas, K., Zwicklbauer, S., Paschke, A. TR2MTL: LLM based framework for metric temporal logic formalization of traffic rules. IV 2024. https://arxiv.org/abs/2406.05709
- AutoLegal（LLM + STL 交通法规形式化）. https://trid.trb.org/View/2781011

### 验证与可达性
- Althoff, M., Dolan, J. Online verification of automated road vehicles using reachability analysis. IEEE T-RO 2014.
- Pek, C., Manzinger, S., Koschi, M., Althoff, M. Using online verification to prevent autonomous vehicles from causing accidents. Nature Machine Intelligence 2020.
- Althoff, M., Koschi, M., Manzinger, S. CommonRoad: Composable benchmarks for motion planning on roads. IV 2017.
- Bansal, S., Chen, M., Herbert, S., Tomlin, C. Hamilton-Jacobi reachability: A brief overview and recent advances. CDC 2017. arXiv:1709.07523.
- Bansal, S., Tomlin, C. DeepReach: A deep learning approach to high-dimensional reachability. ICRA 2021. arXiv:2011.02082.
- Mitsch, S. et al. Formal verification of obstacle avoidance and navigation of ground robots. IJRR 2017. arXiv:1605.00604.
- Burgio, P., Ferrando, A., Villani, M. Open challenges in the formal verification of autonomous driving. FMAS 2024 (EPTCS 411). https://arxiv.org/abs/2411.14520

### 神经网络验证
- Katz, G. et al. Reluplex: An efficient SMT solver for verifying deep neural networks. CAV 2017. arXiv:1702.01135.
- Wang, S. et al. Beta-CROWN: Efficient bound propagation with per-neuron split constraints. NeurIPS 2021. arXiv:2103.06624.
- Ivanov, R. et al. Verisig: Verifying safety properties of hybrid systems with neural network controllers. HSCC 2019. arXiv:1811.01828.
- Tran, H.-D. et al. NNV: The neural network verification tool for deep neural networks and learning-enabled CPS. CAV 2020.
- Huang, C. et al. POLAR: A polynomial arithmetic framework for verifying neural-network controlled systems. ATVA 2022.
- Liu, C. et al. Algorithms for verifying deep neural networks. Foundations and Trends in Optimization 2021.
- Kaulen, K., Bak, S., Brix, C. et al. VNN-COMP 2025: Summary and results. https://arxiv.org/abs/2512.19007
- Astorga, A. et al. Perception contracts for safety of ML-enabled systems. OOPSLA 2023.

### 运行时保障
- Ames, A. et al. Control barrier functions: Theory and applications. ECC 2019. arXiv:1903.11199.
- Hsu, K.-C., Hu, H., Fisac, J. The safety filter: A unified view of safety-critical control in autonomous systems. Annual Review of Control, Robotics, and Autonomous Systems 2024. arXiv:2309.05837.
- Alshiekh, M. et al. Safe reinforcement learning via shielding. AAAI 2018. arXiv:1708.08611.
- Mehmood, U. et al. The black-box simplex architecture for runtime assurance of autonomous CPS. NFM 2022.
- Nickovic, D., Yamaguchi, T. RTAMT: Online robustness monitors from STL. ATVA 2020.
- Lindemann, L. et al. Safe planning in dynamic environments using conformal prediction. RA-L 2023. arXiv:2210.10254.
- Lindemann, L., Qin, X., Deshmukh, J., Pappas, G. Conformal prediction for STL runtime verification. ICCPS 2023. arXiv:2211.01539.
- Nakamura, K., Peters, L., Bajcsy, A. Generalizing safety beyond collision-avoidance via latent-space reachability analysis. RSS 2025. https://arxiv.org/abs/2502.00935
- Learned safety filters with adaptive conformal inference. https://arxiv.org/pdf/2604.18482

### 测试、证伪与统计验证
- Kalra, N., Paddock, S. Driving to safety. RAND 2016.
- Annpureddy, Y. et al. S-TaLiRo. TACAS 2011; Donzé, A. Breach. CAV 2010.
- Dreossi, T. et al. VerifAI: A toolkit for the formal design and analysis of AI-based systems. CAV 2019. arXiv:1902.04245.
- Fremont, D. et al. Scenic: A language for scenario specification and scene generation. PLDI 2019. arXiv:1809.09310.
- Fremont, D. et al. Formal scenario-based testing of autonomous vehicles: From simulation to the real world. ITSC 2020. arXiv:2003.07739.
- Koren, M., Alsaif, S., Lee, R., Kochenderfer, M. Adaptive stress testing for autonomous vehicles. IV 2018. arXiv:1902.01909.
- Corso, A. et al. A survey of algorithms for black-box safety validation of cyber-physical systems. JAIR 2021. arXiv:2005.02979.
- Feng, S. et al. Dense reinforcement learning for safety validation of autonomous vehicles. Nature 2023.
- CounterScene: counterfactual safety-critical scenario generation in closed-loop generative BEV world models. https://arxiv.org/abs/2603.21104

### 端到端、VLA 与安全
- Testing between the test cases: Proving end-to-end steering in conditions you never drove. https://arxiv.org/pdf/2609.10951
- A survey on vision-language-action models for autonomous driving. ICCV 2025 Workshops. https://openaccess.thecvf.com/content/ICCV2025W/WDFM-AD/papers/Jiang_A_Survey_on_Vision-Language-Action_Models_for_Autonomous_Driving_ICCVW_2025_paper.pdf
- Differentiable control barrier functions for vision-based end-to-end autonomous driving. https://arxiv.org/pdf/2203.02401
- VLSA / AEGIS: VLA with plug-and-play safety constraint layer. https://arxiv.org/abs/2512.11891
- Safe VLA via barrier enhanced flow matching. https://arxiv.org/abs/2607.29569

### 系统与综合
- Seshia, S., Sadigh, D., Sastry, S. Toward verified artificial intelligence. CACM 2022. arXiv:1606.08514.
- Favaro, F. et al. Building a credible case for safety: Waymo's approach for the determination of absence of unreasonable risk. 2023. https://arxiv.org/abs/2306.01917
- Waymo, independent audits of the safety case program（2025-11，厂商博客）. https://waymo.com/blog/2025/11/independent-audits/
- NVIDIA Halos（2025-03，厂商博客）. https://blogs.nvidia.com/blog/halos-safety-system-autonomous-vehicles
- Koopman, P. How Safe Is Safe Enough? Measuring and Predicting Autonomous Vehicle Safety. 2022.
- Nuzzo, P. et al. A platform-based design methodology with contracts and related tools for the design of cyber-physical systems. Proc. IEEE 2015.
