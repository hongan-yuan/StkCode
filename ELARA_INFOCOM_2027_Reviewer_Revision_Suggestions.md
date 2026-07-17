# IEEE INFOCOM 2027 投稿论文修改建议

## 论文题目

**ELARA: Energy-Efficient and Latency-Aware Microservice Orchestration in Space Computing Power Networks**

---

## 一、总体评价

本文研究动态低轨卫星计算网络中的微服务编排问题，联合考虑微服务执行卫星选择、跨时隙星间链路路由、微服务副本动态部署，以及端到端时延和能耗优化。

论文提出 ELARA 框架，包括：

1. 基于 GNN-PPO 的服务执行节点选择；
2. 基于最小费用流思想的跨时隙数据路由；
3. 基于 UCB 多臂老虎机的副本重部署。

论文选题较新，应用场景符合卫星计算、星上智能和服务化计算的发展趋势。当前稿件已形成较完整的系统模型、算法框架和实验流程，但仍存在以下核心问题：

- 问题定义覆盖范围较宽，主线略显分散；
- 三个算法模块之间存在一定“拼接感”；
- “联合优化”的表述与实际分层求解过程不完全一致；
- 跨时隙路由模型存在若干需要澄清的系统语义和数学问题；
- PPO 和 Bandit 模块的理论定义尚不够完整；
- 实验规模、统计严谨性和运行开销分析不足。

以 INFOCOM Reviewer 的标准看，当前版本更接近 **Weak Reject / Borderline Reject**。若能重点补强问题建模、算法一致性和实验评估，论文具备进一步提升的潜力。

---

# 二、主要修改建议

## 1. 重新审视“联合优化”的表述

论文在摘要、Introduction 和 Problem Formulation 中多次使用：

- jointly optimizes；
- joint orchestration；
- joint service routing and replica deployment。

但从实际算法流程看，ELARA 是以下三个层次的分解：

1. PPO 选择微服务执行节点；
2. 固定源节点和目的节点后，由路由模块传输数据；
3. 每隔若干时隙，由 Bandit 模块调整副本部署。

三个模块的决策变量、时间尺度和反馈机制均不同，因此更准确的表述应为：

- hierarchical orchestration；
- multi-timescale coordinated optimization；
- decomposed online optimization。

### 建议修改

增加一个“Problem Decomposition”小节，明确写出：

$$
\mathcal{P}
\rightarrow
\mathcal{P}_{\mathrm{selection}}
+
\mathcal{P}_{\mathrm{routing}}
+
\mathcal{P}_{\mathrm{deployment}}.
$$

并解释：

- serving-node selection：请求级决策；
- routing：服务阶段级决策；
- replica redeployment：窗口级决策。

还应讨论这种分解的合理性，以及与全局最优联合求解相比可能带来的性能损失。

---

## 2. 修正 Problem Formulation 中的变量和目标函数

### 2.1 优化变量定义不一致

当前问题中写为：

$$
\min_{y_{r,i,v},\Pi_r^{ru},A} J_r,
$$

其中 $A$ 没有明确说明是 action set、deployment action，还是所有重部署动作集合。

建议统一写为：

$$
\min_{\mathbf{y}_r,\boldsymbol{\Pi}^{ru}_r,\mathbf{K}(t)}.
$$

同时统一使用 $K(t)$ 或 $\mathbf{K}(t)$ 表示副本部署动作。

### 2.2 请求级目标与窗口级部署目标混在一起

当前 $J_r$ 是单请求目标，但副本重部署是基于一个窗口内多个请求做出的慢时间尺度决策。二者不宜写入同一个 request-level objective。

建议拆分为：

#### 快时间尺度目标

$$
\min
\mathbb{E}
\left[
T_r^{e2e}
+
\lambda E_r^{tot}
+
\rho I_r^{fail}
\right].
$$

#### 慢时间尺度目标

$$
\min
\frac{1}{|\mathcal{R}_W|}
\sum_{r\in\mathcal{R}_W}
\left(
\alpha T_r^{e2e}
+
\beta E_r^{tot}
+
\rho I_r^{fail}
\right)
+
\eta C^{mig}.
$$

这样可清楚区分在线请求调度和周期性副本部署。

### 2.3 副本迁移开销未完整进入总目标

论文称考虑 redeployment overhead，但当前总能耗和总时延主要包括：

- 通信能耗；
- 计算能耗；
- 通信时延；
- 计算时延。

建议进一步考虑：

- 服务镜像传输能耗；
- 镜像传输时延；
- 冷启动时延；
- 部署期间的服务不可用；
- 迁移流量与正常业务流的链路竞争；
- 存储写入和初始化开销。

可定义：

$$
E^{tot}
=
E^{ru}
+
E^{cp}
+
E^{mig},
$$

$$
T^{tot}
=
T^{ru}
+
T^{cp}
+
T^{cold}.
$$

---

## 3. 重新梳理 MCMF 路由模型

这是全文最需要优先修改的部分之一。

### 3.1 “Min-Cost Max-Flow”表述可能不准确

当前问题的目标并非最大化总吞吐量，而是在固定数据量下最小化时延与能耗。因此更准确的术语可能是：

- min-cost flow；
- splittable flow；
- minimum completion-time flow；
- deadline-constrained flow。

建议重新审视“max-flow”是否确实是该模块的核心。

### 3.2 多跳路径时延模型需明确转发机制

当前路径时延写为：

$$
\delta_p^h
=
\sum_{e\in p}
T_e^{cm}(d_p^h,t_h).
$$

该式隐含严格 store-and-forward 假设，即每一跳完整接收后下一跳才能发送。

建议明确采用哪种机制：

- store-and-forward；
- cut-through；
- pipelined forwarding。

若采用流水传输，应根据瓶颈速率和传播时延重新建模，而不是简单将每跳完整传输时延相加。

### 3.3 共享链路上的流量与时延计算不一致

公式中虽然用：

$$
f_e^h
=
\sum_{p:e\in p}
d_p^h
$$

表示共享边上的总流量，但路径时延仍使用单路径流量 $d_p^h$ 计算。

这可能低估共享链路上的拥塞与序列化时延。

建议采用 edge-flow formulation：

$$
\sum_{e\in\delta^+(v)}f_e
-
\sum_{e\in\delta^-(v)}f_e
=
b_v,
$$

$$
0
\le
f_e
\le
C_e(t).
$$

然后基于链路负载、瓶颈速率或阶段完成时间构造目标函数。

### 3.4 多路径“并行完成时间”假设需修正

当前使用：

$$
\Delta_h^{ru}
=
\max_p \delta_p^h.
$$

但当多条路径共享部分 ISL 时，它们并不是真正独立并行，完成时间不能简单由最慢路径决定。

建议考虑共享链路上的调度和序列化约束。

### 3.5 跨时隙剩余数据的位置不清楚

当前描述为：若本时隙未传完，则下一个时隙对 residual data 重新路由。

需要明确剩余数据在哪里：

1. 仍位于原始源节点；
2. 已到达某个中间卫星；
3. 多路径上的不同数据块分散在多个卫星。

若允许 store-carry-forward，则需要在模型中加入：

- 中间节点缓存；
- 缓存容量；
- 跨时隙数据位置；
- 下一个时隙的实际源节点；
- 多路径残余数据状态。

否则 cross-slot residual routing 的系统语义不完整。

---

## 4. 完善 PPO 的 MDP 定义

### 4.1 状态定义需要进一步说明 Markov 性

当前状态包括：

- 当前拓扑；
- 节点计算资源；
- 链路状态；
- 当前服务阶段；
- 待执行微服务链；
- 数据量。

但系统还包含：

- ISL 队列动态；
- 计算队列动态；
- 在途数据；
- 其他请求带来的竞争；
- 背景流量变化。

建议进一步说明：

- 当前状态是否足以满足 Markov property；
- 背景流量如何演化；
- 队列长度如何更新；
- 是否需要加入历史窗口或短期预测。

### 4.2 未来拓扑信息的使用需明确

候选节点特征和路由代价中包含 future link unavailability risk。

由于 LEO 拓扑具有可预测性，这一设计可以成立，但需要明确：

- 可获得未来多少时隙的拓扑；
- 使用精确轨道预测还是统计预测；
- 未来拓扑对所有 baseline 是否同样可用；
- 背景流量和队列是否也可预测。

否则容易产生不公平比较。

### 4.3 Reward 存在重复计算风险

局部奖励已经包含每阶段的时延和能耗：

$$
r_{r,i}^{loc}
=
-
\left(
\alpha T_{r,i}^{step}
+
\beta E_{r,i}^{step}
+
\lambda N_{r,i}^{cross}
\right).
$$

终止奖励又包含完整端到端时延和总能耗：

$$
r_r^{term}
=
-
\left(
\alpha T_r^{e2e}
+
\beta E_r^{tot}
\right).
$$

这会重复计算已经在局部奖励中出现的成本。

建议选择以下两种方式之一：

- 仅使用 dense incremental reward；
- 局部 reward + 只包含 deadline/failure 的 terminal reward。

### 4.4 统一折扣因子和奖励塑形参数

当前 $\omega_r$、$\gamma$ 等符号的作用容易混淆。

建议统一为：

- PPO discount factor：$\gamma$；
- terminal reward coefficient：$\lambda_{\mathrm{term}}$。

---

## 5. 进一步证明 GNN 的必要性和有效性

当前使用两层 mean aggregation GNN，类似 GraphSAGE。

但两层 GNN 只能聚合有限范围的邻居信息，而候选执行节点可能距离当前节点多个 hops。论文又通过人工构造的候选特征 $e_v$ 输入路由时延、链路风险、瓶颈容量和计算成本，这可能弱化 GNN 本身的作用。

### 建议增加以下消融

- PPO without GNN；
- MLP-PPO；
- GAT-PPO；
- 不同 GNN 层数；
- 不同 embedding dimension；
- 不使用候选路由特征 $e_v$；
- GNN-Greedy；
- weighted-cost heuristic。

当前 ELARA-NR 仅将整个 PPO 替换为 nearest-replica，无法证明：

- GNN 是否真正有效；
- PPO 是否优于普通启发式；
- 性能是否主要来自人工特征。

---

## 6. 规范 Bandit-based Redeployment 模块

### 6.1 明确 Bandit 类型

当前 arm 定义为：

$$
k=(o,m,p_s,p_t).
$$

但相同 plane-level arm 最终还需选择具体卫星，因此同一个 arm 在不同时间可能对应不同执行结果，reward distribution 具有明显非平稳性。

建议说明该问题属于：

- contextual bandit；
- non-stationary bandit；
- combinatorial bandit；
- delayed-feedback bandit；

而不是直接使用普通 UCB 表述。

### 6.2 说明 arm 数量和可扩展性

理论 arm 数量可能达到：

$$
3\times M\times P^2.
$$

需要说明：

- 如何筛选 candidate arms；
- 实际每轮评估多少 arms；
- UCB 统计如何维护；
- 新 arm 如何初始化。

### 6.3 解决多动作情况下的 credit assignment

Algorithm 2 允许一个部署窗口内同时执行多个 actions，但后续 reward 来自整个窗口的：

- 成功率；
- 平均时延；
- 平均能耗。

因此很难区分每个 action 的独立贡献。

建议：

- 每个窗口只执行一个 arm；
- 或采用 marginal contribution；
- 或采用 counterfactual reward；
- 或将多个动作定义为一个 combinatorial arm。

### 6.4 明确定义延迟反馈

需要说明：

- reward 延迟多少个窗口反馈；
- action 与 reward 如何匹配；
- 多个尚未结算 action 如何并存；
- 窗口长度如何设置；
- delayed reward 是否会导致非平稳偏差。

---

## 7. 强化实验严谨性

当前实验部分是全文最薄弱的部分。

### 7.1 增加随机种子和统计区间

当前只使用四个随机种子，且多数图没有：

- error bars；
- standard deviation；
- confidence intervals；
- significance test。

建议至少使用 8–10 个随机种子，报告：

- mean；
- standard deviation；
- 95% confidence interval。

### 7.2 增加系统负载实验

当前主要围绕微服务链长度展开，缺少关键 workload sensitivity。

建议增加：

1. latency vs. request arrival rate；
2. energy vs. request arrival rate；
3. success rate vs. request arrival rate；
4. deadline violation vs. request arrival rate；
5. latency vs. ISL background load；
6. latency vs. CPU background load；
7. performance vs. replica count；
8. performance vs. hotspot skewness。

### 7.3 报告在线决策开销

应补充：

- GNN-PPO 推理时间；
- 路由求解时间；
- 每时隙 augmenting path 次数；
- Bandit 决策时间；
- 内存开销；
- 随星座规模增长的可扩展性。

建议增加如下表格：

| Constellation Size | PPO Inference | Routing Computation | Deployment Decision |
|---|---:|---:|---:|
| Small |  |  |  |
| Medium |  |  |  |
| Large |  |  |  |

并将这些时间与 topology slot duration 对比。

### 7.4 说明训练测试划分

需要明确：

- 训练使用多少 topology cycles；
- 测试是否使用未见过的时间窗口；
- 训练和测试是否共享 request trace；
- 不同负载是否需要重新训练；
- 星座规模变化后是否重新训练。

建议使用：

- training windows；
- validation windows；
- unseen test windows。

### 7.5 补充参数来源

建议对以下参数给出数值和引用：

- CPU cycle demand；
- service image size；
- storage demand；
- CPU power；
- background load；
- ISL degradation factor；
- request data size；
- deadline；
- migration cost。

---

## 8. 改进 baseline 设计和适配说明

当前使用：

- SECO；
- SC-NFV；
- SP-Routing。

但这些工作的原始问题与本文并不完全一致。

建议增加 “Baseline Adaptation” 小节，说明：

- 原始算法解决什么问题；
- 如何适配到微服务链；
- 是否支持跨时隙；
- 是否支持动态计算队列；
- 是否支持副本重部署；
- 是否使用未来拓扑；
- 参数如何调优。

### 建议增加更基础的 baseline

- shortest path + nearest replica；
- minimum estimated latency greedy；
- weighted latency-energy greedy；
- static placement + MCMF；
- PPO without GNN；
- oracle with short horizon；
- small-scale MILP optimum。

尤其建议加入小规模最优解，以报告 optimality gap。

---

## 9. 调整对性能提升的表述

当前 ELARA 相对 SECO：

- 时延提升约 4.4%；
- 能耗几乎相同。

对于一个包含 GNN-PPO、MCMF 和 Bandit 的复杂框架，这一增益不算特别大。

建议谨慎使用：

- significantly；
- substantially；
- consistently superior。

除非提供：

- 置信区间；
- 显著性检验；
- 高负载场景下更明显的增益。

还应解释：

- 为什么相对 SECO 提升较小；
- 复杂度是否值得；
- ELARA 在什么场景下最有优势。

---

## 10. 完整展示所有指标

论文列出了：

- average end-to-end latency；
- average energy；
- success rate；
- deadline acceptance rate；
- communication delay；
- slot crossings。

但当前图中主要展示时延和能耗。

建议增加综合结果表：

| Method | Latency | Energy | Success Rate | Deadline Violation | Slot Crossings |
|---|---:|---:|---:|---:|---:|
| ELARA |  |  |  |  |  |
| SECO |  |  |  |  |  |
| SC-NFV |  |  |  |  |  |
| SP-Routing |  |  |  |  |  |

此外，当前 deadline acceptance 接近 100%，可能说明 deadline 设置过宽。

建议设置多档 deadline tightness：

$$
Z_r
=
\kappa T_r^{reference},
\quad
\kappa
\in
\{1.1,1.3,1.5,2.0\}.
$$

---

# 三、系统模型方面的具体问题

## 1. 明确源卫星和目的卫星的定义

请求模型为：

$$
r
=
\langle
v_s,m_1,\ldots,m_L,v_d,\tau_{in}
\rangle.
$$

但 Introduction 中请求来自 terrestrial user。

建议说明：

- 用户请求如何映射到 source satellite；
- destination 为什么定义为 satellite；
- 是否忽略 ground-satellite access；
- feeder link 和 satellite-ground handover 是否超出本文范围。

可明确写为：本文聚焦请求进入星座后的星间微服务执行过程。

---

## 2. 明确 stateless microservice 假设

实际应用可能涉及：

- intermediate state；
- model cache；
- session state；
- shared database；
- replica consistency。

建议明确：

- 本文只考虑 stateless microservices；
- intermediate data 随请求传输；
- 不考虑副本间状态同步；
- stateful microservices 留作未来工作。

---

## 3. 修正计算能耗模型解释

当前：

$$
E_m^{cp}(v,t)
=
P_v^{cp}
\frac{C_m}{F_v(t)}.
$$

若 $F_v(t)$ 变化来自 DVFS，则功率不应固定。

建议明确：

- CPU 工作频率固定；
- $F_v(t)$ 的变化来自 background load 或 available capacity；
- 本文不研究 DVFS。

否则需采用频率相关功率模型。

---

## 4. 考虑接收能耗

当前通信能耗主要考虑发送端。

建议至少写为：

$$
E_e
=
(P_e^{tx}+P_e^{rx})
\frac{f_e}{R_e}.
$$

若忽略接收能耗，也应说明理由。

---

## 5. 补充队列模型

当前 queuing delay 主要描述为“可探测”，但仿真中必须有明确更新机制。

建议说明：

- FCFS 或其他调度纪律；
- queue update equation；
- task arrival process；
- packet arrival process；
- CPU sharing rule；
- 是否允许 preemption；
- 是否采用 M/M/1 近似。

---

# 四、算法表达和伪代码修改建议

## 1. 分离训练和在线推理

当前 Algorithm 1 同时包含：

- 训练时采样；
- 推理时 argmax；
- 存储 transition；
- PPO 更新。

建议拆成：

1. Offline Training of GNN-PPO；
2. Online Service Orchestration。

并明确系统部署后是否还在线训练。

---

## 2. 单独给出跨时隙路由算法

第 9 行仅写“执行 MCMF 路由”，不足以体现论文核心贡献。

建议单独给出：

- residual graph construction；
- available slot capacity calculation；
- augmenting path search；
- traffic splitting；
- residual data update；
- topology switch；
- rerouting。

---

## 3. 给出算法复杂度

建议分别给出：

- GNN forward complexity；
- candidate scoring complexity；
- routing complexity；
- Bandit candidate evaluation complexity。

例如路由模块可分析为：

$$
O(K_{\max}|E|\log |V|),
$$

具体形式根据所采用的 shortest augmenting path 实现确定。

---

## 4. 定义 safety margin

Algorithm 2 中出现：

> if estimated reward exceeds the safety margin

但正文没有明确定义 safety margin。

建议说明：

- 具体表达式；
- 默认参数；
- 与 migration cost 的关系；
- 参数敏感性。

---

# 五、写作和结构优化建议

## 1. 强化 Introduction 中的 research gap

建议将 gap 凝练为三个明确问题：

1. 现有工作通常假设请求在单快照或单时隙内完成；
2. 现有服务路由未统一考虑跨时隙残余数据、链路状态和计算队列；
3. 现有副本部署方法动作空间过大，难以在线适应动态星座。

然后逐项对应论文贡献。

---

## 2. Contributions 应突出机制，而非算法名称

当前贡献主要强调：

- GNN-PPO；
- min-cost max-flow；
- multi-armed bandit。

仅使用这些现有算法本身不构成强创新。

建议突出：

- topology-aware candidate scoring；
- residual-aware cross-slot routing；
- orbital-plane-level action abstraction；
- delayed-feedback redeployment；
- multi-timescale coordination。

---

## 3. Related Work 增加对比表

建议增加：

| Work | Microservice Chain | Dynamic Topology | Cross-Slot Execution | Computation-Routing Coordination | Replica Redeployment |
|---|---|---|---|---|---|
| SECO |  |  |  |  |  |
| SC-NFV |  |  |  |  |  |
| SP-Routing |  |  |  |  |  |
| ELARA | ✓ | ✓ | ✓ | ✓ | ✓ |

---

## 4. 简化 Fig. 3

当前框架图信息密度较高，在双栏打印下可读性有限。

建议：

- 放大三个核心模块；
- 减少小字号文字；
- 明确 fast-timescale 和 slow-timescale；
- 区分 request flow、state feedback 和 deployment update；
- 减少重复图标和装饰元素。

---

## 5. 拆分训练曲线

Fig. 4 同时使用 reward 和 PPO loss 双纵轴，容易误读。

建议拆成：

- reward convergence；
- actor/critic loss convergence。

并显示多个随机种子的均值与置信区间。

---

# 六、语言、符号和排版问题

建议重点检查以下问题：

- “calssic Walker-Delta” 改为 “classic Walker-Delta”；
- “as show in Fig. 1” 改为 “as shown in Fig. 1”；
- “Related works” 改为 “Related Work”；
- “shows and analyses the Simulation results” 改为 “presents the performance evaluation”；
- “a geographically close replica’s satellite” 改为 “a satellite hosting a geographically nearby replica”；
- 统一 “mean execution energy” 和 “average energy consumption”；
- 统一 “deadline acceptance rate” 和 “deadline violation rate”；
- Equation (34) 中 “edge $m$” 应改为 “edge $e$”；
- 检查公式编号是否从 (21) 跳到 (23)；
- 检查公式 (17)、(18) 的求和范围和排版；
- 避免 $H$、$H_{r,i}$、routing horizon 混用；
- 副本数量上下界建议由 $R^{min},R^{max}$ 改为 $N_m^{min},N_m^{max}$；
- 避免 $A$ 同时表示 satellite number 和 action notation；
- 统一 time slot、topology slot、slot window 等术语。

---

# 七、建议优先补充的实验

## 必须补充

1. Request arrival rate sensitivity；
2. Runtime overhead；
3. PPO/GNN 细粒度消融；
4. Cross-slot routing 消融；
5. Small-scale optimal solution；
6. 95% confidence interval；
7. Baseline adaptation details。

## 强烈建议补充

8. Different topology slot durations；
9. Different constellation sizes；
10. Different hotspot distributions；
11. Different replica image sizes；
12. Different latency-energy weights；
13. Deadline tightness sensitivity；
14. Unseen topology generalization；
15. Different routing horizons；
16. Different maximum augmenting paths。

---

# 八、可能的 Reviewer 最终意见

> The paper studies an interesting and timely problem of microservice orchestration in dynamic LEO satellite computing networks. The proposed framework integrates GNN-PPO-based serving-node selection, cross-slot flow routing, and bandit-based replica redeployment. However, the current formulation does not fully support the claimed joint optimization, as the three modules are largely decoupled and optimized at different timescales. The cross-slot multipath routing model also has several unclear assumptions regarding shared-link contention, intermediate buffering, and residual data locations. In addition, the experimental evaluation is limited in workload diversity, statistical rigor, runtime overhead, and comparison against optimal or stronger learning-based baselines. Therefore, I am not yet convinced that the current version provides sufficient technical novelty and evaluation depth for INFOCOM.

---

# 九、最优先修改顺序

建议按照以下顺序修改：

1. 修正跨时隙路由模型，明确 residual data 和中间缓存语义；
2. 将“联合优化”改为多时间尺度协调优化，并重写 Problem Formulation；
3. 补充 arrival rate、confidence interval、runtime overhead 和 GNN/PPO 消融；
4. 增加 baseline adaptation 说明和小规模最优解；
5. 修正 Bandit 的多动作 delayed feedback 和 credit assignment；
6. 将迁移开销纳入系统模型和总目标；
7. 统一符号、公式和算法流程；
8. 压缩背景叙述，突出问题特定的机制创新。

---

## 结论

当前稿件的研究方向和整体框架是合理的，但要达到 IEEE INFOCOM 的录用标准，还需进一步强化：

- 数学建模的严谨性；
- 三个模块之间的协调逻辑；
- 跨时隙路由的系统语义；
- GNN-PPO 和 Bandit 的必要性证明；
- 实验的统计严谨性和系统覆盖度。

最关键的问题不是重新设计整个 ELARA，而是将已有框架进一步整理为一个逻辑闭环、定义一致、实验充分的多时间尺度微服务编排方案。
