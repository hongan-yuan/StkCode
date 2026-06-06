# 卫星微服务三层协同仿真实现大纲（跨 Slot 路由与隐式并发负载修订版）


## 1. 背景、动机与总体架构

### 1.1 研究背景

低轨卫星网络由数以百计甚至更多的低轨卫星节点组成，卫星之间通过星间链路（Inter-satellite Link, ISL）形成动态网络拓扑。按照链路所连接卫星的轨道关系，ISL 可以分为两类：

```text
同轨 ISL：连接同一轨道平面内的相邻卫星。
跨轨 ISL：连接相邻轨道平面内的卫星。
```

同轨 ISL 中卫星之间的相对位置较为稳定，因此链路可用性、传播距离和通信质量通常较稳定。跨轨 ISL 更容易受到极区、可视范围、多普勒效应和相对运动变化的影响，链路可用性和有效传输速率具有明显时变特征。因此，低轨卫星网络不是静态通信网络，而是具有周期性、时变性和链路异质性的动态网络。

随着卫星载荷、散热能力和星上计算硬件的发展，低轨卫星不再只是通信转发节点，也逐渐具备一定通用计算、存储和内存资源，可以承担部分星上数据处理任务。然而，单颗低轨卫星仍然受到板载空间、供电能力、散热能力和存储容量的限制，难以像地面数据中心一样部署和运行体量较大的单体应用。

微服务架构为星上计算提供了可行的软件组织方式。它将大规模单体应用拆分为若干功能相对独立、资源需求较小的微服务，使每个微服务能够单独部署、独立执行。对于低轨卫星星座而言，可以将一个应用拆分后的微服务集合以分布式方式部署在多颗卫星上，通过多星协同完成用户任务，从而形成在轨服务能力。

### 1.2 研究动机

本研究关注低轨卫星星座中跨 time slot 的微服务请求链执行问题。实际系统中，微服务请求可能在任意时刻到达，而不是严格在 time slot 起点到达。一个请求链的执行过程包括多个阶段：

```text
源卫星 -> 第 1 个微服务副本 -> 第 2 个微服务副本 -> ... -> 最后一个微服务副本 -> 目的卫星
```

每一段都可能包含：

```text
通信排队时延 + 发送时延 + 传播时延 + 计算等待时延 + 计算执行时延
```

因此，即使单个请求链本身不长，也可能因为链路带宽、排队、计算负载和跨轨路径变化而跨越多个 time slots。跨 slot 后，网络拓扑、链路可用性、链路背景负载、卫星计算能力和微服务副本部署状态均可能变化。

实际系统还存在多请求并发。显式模拟所有请求链的并发资源竞争会显著增加仿真复杂度，因此我们拟采用隐式并发近似：同一时刻只显式推进一条 service chain，但用随机背景通信负载和随机背景计算负载刻画其他并发请求造成的资源退化。

### 1.3 请求链建模

一个微服务请求链表示为：

```text
r = <v_src, m_1, m_2, ..., m_L, v_dst, τ_arr>
```

其中：

```text
v_src：请求起点卫星。
v_dst：请求终点卫星。
τ_arr：请求到达的连续时间，可以位于任意 time slot 内。
L：服务链长度。
m_i：第 i 个微服务。
```

每个微服务链上的微服务具有：

```text
CPU cycles：计算开销。
image_size：镜像大小，用于副本迁移成本。
input_data_size：输入数据量。
output_data_size：输出数据量或中间数据量。
```

微服务采用无状态副本部署方式。同一个微服务可以在多颗卫星上部署等价副本，副本之间不需要强状态同步。

### 1.4 时变拓扑与跨 Slot 建模

整个星座运行周期被划分为若干 time slots：

```text
Δt_1, Δt_2, ..., Δt_T
```

当前仿真中每个 slot 默认为 10 秒。在一个 slot 内，星座拓扑、星间链路距离、链路额定速率、链路可用性和背景负载状态近似保持不变；不同 slots 之间，这些状态随卫星运动和随机背景负载变化而更新。

对于某一时刻 τ，其所属 slot 为：

```text
slot(τ) = floor(τ / Δt)
```

一条请求链可能跨越多个 slots。跨 slot 后（尤其是涉及到跨 time slot 后，原本建立的星间传输链路可能会失效的情况）：

```text
1. 路由路径需要重新检查；
2. 若原路径链路断开，需要分段重路由；
3. 计算资源折扣可能变化；
4. 待执行服务需要基于最新副本部署重新选择执行节点；
```

### 1.5 仿真总体架构


```text
0. Background Load Generator
   生成每个 slot 的背景链路负载和背景计算负载。

1. PPO-GNN Service Execution Node Selector
   在每个 service hop 选择 next-hop microservice 的执行卫星节点。其输入包括当前请求的微服务链特征、当前 slot 的星座 ISL 通信条件、未来 K 个 time slots 内星间链路的额定速率与可用性预测，以及服务链中各微服务副本所部署卫星在当前 slot 的计算负载和算力状态。PPO-GNN 不输入未来 slots 的背景链路负载。

2. Cross-slot Route Planner
   在 next-hop microservice 的执行卫星节点确定后，将源节点与目标节点之间的数据传输建模为按 slot 更新的容量受限传输问题。当前版本采用 slot-level min-cost max-flow，在最大化当前 slot 可端到端送达数据量的同时，最小化由时延、能耗和断链风险组成的综合费用。

3. Bandit Replica Placement / Migration Agent
   每隔 n 个 slots 基于长期 service pressure 调整副本部署。

4. Chain Simulator
   显式推进单条服务链的通信、计算、跨 slot 状态演化和完成/失败判定。
```

总体目标为：

```text
在动态 LEO 拓扑、背景负载扰动、有限星上计算资源和跨 slot 链路断开风险下，降低 service chain 的端到端时延、能耗、失败率。
```

---

## 2. 隐式并发背景负载模型

### 2.1 隐式并发仿真假设

不显式维护所有并发请求链的资源占用，而采用如下抽象：

```text
同一时刻只显式处理一条 service chain；
其他并发请求被抽象为 stochastic background communication load 和 stochastic background computing load；
背景负载以统一口径影响链路有效速率、链路排队时延、CPU 利用率和有效计算资源折扣，避免同一拥塞因素在容量、时延和奖励中被重复惩罚。
```

这种方法可以近似多请求并发造成的统计性资源退化，但不刻画请求级别的精确资源预留、先后顺序、公平性和相互阻塞。

### 2.2 背景链路流量到达

对每条 ISL 边 e，在 time slot t 内的背景通信请求数量服从泊松分布：

```text
N_e(t) ~ Poisson(λ_e(t) · Δt)
```

其中：

```text
λ_e(t)：链路 e 在 slot t 的背景通信请求到达率。
Δt：slot 长度。
```

每个背景通信请求的数据量(单位为Gb)可以从给定分布中采样：

```text
S_{e,j}(t) ~ DataSizeDistribution
```

背景通信总需求为(单位为Gb)：

```text
B_bg,e(t) = Σ_{j=1}^{N_e(t)} S_{e,j}(t)
```

### 2.3 链路利用率与速率折扣

链路 e 在 slot t 的额定传输速率为(这个可以预先计算出来，这部分数据可以从WalkerDeltaConstellationSimu目录下的csv文件中获取)：

```text
R_e(t)
```

背景利用率定义为：

```text
ρ_e(t) = min( B_bg,e(t) / (R_e(t) · Δt), ρ_max )
```

链路有效速率折扣系数采用较为平滑的指数映射：

```text
η_e(t) = exp(-κ_link · ρ_e(t))
```

因此有效链路速率为：

```text
R_eff,e(t) = R_e(t) · η_e(t)
```

为避免背景负载被重复惩罚，本版本采用如下统一口径：

```text
1. 链路容量只使用 R_eff,e(t) 计算，不再额外乘以 (1 - ρ_e(t))；
2. 链路排队时延 D_queue,e(t) 使用同一个 ρ_e(t) 推导，作为时延项加入；
3. 奖励函数默认不再单独加入 background_congestion_penalty；
4. 若后续实验需要拥塞正则项，只作为消融实验开启，并降低权重。
```

### 2.4 链路排队时延：M/M/1 近似

在隐式并发模型中，链路排队时延使用 M/M/1 近似。设：

```text
λ_e(t)：背景通信请求到达率。
μ_e(t)：链路服务率。
ρ_e(t) = λ_e(t) / μ_e(t)。
```

当 ρ_e(t) < 1 时，平均系统逗留时间为：

```text
D_system,e(t) = 1 / ( μ_e(t) - λ_e(t) )
```

平均排队等待时间可近似为：

```text
D_queue,e(t) = ρ_e(t) / ( μ_e(t) - λ_e(t) )
```

工程实现中推荐使用利用率映射形式，避免 `λ_e(t)`、`μ_e(t)` 与数据量单位不一致：

```text
D_queue,e(t) = D0_link · ρ_e(t) / (1 - ρ_e(t) + ε)
```


### 2.5 链路总时延

对当前 service chain 的数据量 b，边 e 在 slot t 的单位传输时延为：

```text
D_e(t,b) = D_queue,e(t) + b / R_eff,e(t) + distance_e(t) / c
```

其中：

```text
D_queue,e(t)：链路排队时延。
b / R_eff,e(t)：发送时延。
distance_e(t) / c：传播时延。
```

### 2.6 背景计算负载到达

对每颗卫星 v，在 slot t 内的背景计算任务数量服从泊松分布：

```text
N_v^c(t) ~ Poisson(λ_v^c(t) · Δt)
```

每个背景计算任务的 CPU cycles 为：

```text
C_{v,j}^bg(t) ~ ComputeDemandDistribution
```

背景计算总需求为：

```text
C_bg,v(t) = Σ_{j=1}^{N_v^c(t)} C_{v,j}^bg(t)
```

### 2.7 计算负载状态：Markov 状态模型

卫星计算负载不是相邻 slot 完全独立的随机变量，而具有时间相关性。原因是：星上后台任务、载荷处理、缓存传输、热控约束、电源状态和业务热点通常会在连续时间段内保持一定状态，而不会在每 10 秒 slot 中完全随机跳变。

因此，本版本采用 Markov 状态模型表示每颗卫星的计算负载状态：

```text
Z_v(t) ∈ {Idle, Light, Medium, Heavy}
```

状态转移为：

```text
P_v[a,b] = Pr( Z_v(t+1)=b | Z_v(t)=a )
```

每个状态对应不同的背景任务到达率、计算利用率范围和计算能力折扣范围：

```text
Idle:       λ_v^c low,    ξ_v(t) ∈ [0.80, 1.00]
Light:      λ_v^c small,  ξ_v(t) ∈ [0.60, 0.80]
Medium:     λ_v^c medium, ξ_v(t) ∈ [0.40, 0.60]
Heavy:      λ_v^c high,   ξ_v(t) ∈ [0.20, 0.40]
```

计算能力折扣为：

```text
F_eff,v(t) = F_v · ξ_v(t)
```

其中 F_v 是卫星 v 的额定计算能力(单位为FPLOs)：。



### 2.8 微服务执行前等待时间

微服务执行前等待时间不直接服从泊松分布，而由计算利用率推导。设卫星 v 在 slot t 的计算利用率为：

```text
ρ_v^c(t) = min( C_bg,v(t) / (F_v · Δt), ρ_max )
```

等待时间可近似为：

```text
W_v(t) = W0_compute · ρ_v^c(t) / (1 - ρ_v^c(t) + ε)
```

---

## 3. 跨 Slot 分段执行模型


### 3.1 连续时间状态维护

每条 service chain 在仿真过程中维护如下运行状态：

```text
request_id：请求编号。
arrival_time：请求到达时间 τ_arr。
current_time：当前时间 τ。
current_slot：floor(current_time / Δt)。
current_node：当前数据所在卫星。
current_service_index：当前待执行微服务编号 i。
remaining_chain：尚未执行的微服务序列。
remaining_data：当前 hop 尚未完成传输的数据量。
accumulated_delay：累计端到端时延。
accumulated_energy：累计能耗。
slot_crossings：跨 slot 次数。
status：running / finished / failed。
```

请求可以在任意连续时间到达：

```text
τ_arr ∈ [t·Δt, (t+1)·Δt)
current_slot = floor(τ_arr / Δt)
```

后续所有通信和计算过程均直接推进 `current_time`。

### 3.2 每个 Service Hop 的执行逻辑

当请求链执行到第 i 个微服务 `m_i` 时，系统按如下顺序推进：

```text
1. 根据当前时间 τ 和最新副本部署状态，枚举 m_i 的可用执行卫星节点集合；
2. 节点选择智能体根据当前请求链特征、当前决策 slot 的全星座 ISL 通信状态、未来 K 个 slots 内 ISL 可用性、额定速率与传播距离预测，以及服务链相关微服务副本所在卫星在当前 slot 的计算负载和算力状态，选择执行节点 v_exec；
3. 选定 v_exec 后，源节点 current_node 与目标节点 v_exec 已确定；
4. Cross-slot Route Planner 使用 slot-level min-cost max-flow，按 slot 分段传输 current_node -> v_exec 的数据；
5. 若传输跨越 slot，则在 slot 边界基于最新拓扑、链路容量和背景负载重新规划剩余数据传输；
6. 数据到达 v_exec 后，进入计算执行阶段；
7. 微服务计算允许跨 slot，但一旦开始执行则不中断；
8. 计算完成后，current_node 更新为 v_exec，current_service_index 加 1，进入下一 service hop。
```

因此，跨 slot 行为由“连续时间推进 + slot 边界重规划”实现。当前版本在单条端到端路径内部采用 store-and-forward 时延/能耗口径：同一份数据需要逐跳发送，每跳都会产生发送时延、传播时延、排队时延、switch penalty 和通信能耗。与此同时，本版本不建模跨 slot 的中继缓存：只有端到端到达目标节点 `v_exec` 的数据才从 `remaining_data` 中扣除；未到达目标的数据不可以停留在中继节点，下一 slot 仍从原始 `current_node` 重新规划剩余数据传输。

### 3.3 通信跨 Slot 处理

给定源节点 `u`、目标节点 `v`、待传输数据量 `B` 和开始时间 `τ`，通信过程按 slot 分段执行：

```text
remaining_data = B
current_time = τ
current_node = u

while remaining_data > 0:
    current_slot = floor(current_time / Δt)
    slot_end = (current_slot + 1) · Δt
    Δt_remain = slot_end - current_time

    基于当前 slot 的拓扑、链路容量、当前 slot 背景负载和目标节点 v，构造当前 slot 的路径候选图；
    构造当前 slot 的容量-费用图；
    运行 min-cost max-flow，最大化当前 slot 可端到端送达目标的数据量，并最小化传输费用；
    只对已经端到端到达目标 v 的数据更新 remaining_data；
    未到达目标的数据不在中继节点保留，视为仍停留在 current_node；
    更新 current_time、累计时延和能耗；

    if remaining_data > 0:
        current_time = slot_end
        在下一 slot 仍以 current_node 为源重新规划。
```

如果当前 slot 内无可行路径，则请求可以等待到下一 slot。若连续超过 `route_horizon_slots=5` 仍无法传输完成，则判定该 hop 路由失败。

### 3.4 计算跨 Slot 处理

若微服务 `m_i` 在卫星 `v_exec` 上开始执行，计算开始时间为：

```text
compute_start_time = data_arrival_time + W_v(data_arrival_time)
```

然后按 slot 累计可执行 CPU cycles：

```text
remaining_cycles = CPU_cycles(m_i)
current_time = compute_start_time

while remaining_cycles > 0:
    slot = floor(current_time / Δt)
    slot_end = (slot + 1) · Δt
    available_time = slot_end - current_time
    executable_cycles = F_eff,v(slot) · available_time
    done_cycles = min(remaining_cycles, executable_cycles)
    remaining_cycles -= done_cycles
    current_time += done_cycles / F_eff,v(slot)

compute_finish_time = current_time
```

该模型表示计算任务不中断、不迁移，但计算速度可随不同 slot 的有效算力变化。

### 3.5 服务链顺序约束

请求链必须按照如下顺序执行：

```text
m_1 -> m_2 -> ... -> m_L
```

每完成一个微服务后，才允许进入下一个微服务。每个 service hop 开始时，系统都基于当前决策 slot 的时间、当前数据所在卫星、当前副本部署状态、当前 slot 通信拓扑状态和当前 slot 计算负载状态重新选择执行卫星节点。


---

## 4. Cross-slot Route Planner

### 4.1 模块职责

Cross-slot Route Planner 负责以下传输：

```text
1. 请求源卫星 -> 当前微服务副本。
2. 当前微服务副本 -> 下一个微服务副本。
3. 最后一个微服务副本 -> 请求目的卫星。
4. 微服务副本 add/move 过程中的镜像传输。
```


其中，Cross-slot Route Planner 不再承担“选择哪个微服务副本执行”的职责，而是在给定：

```text
source node: v_src^hop
目标执行节点: v_dst^hop
待传输数据量: B_hop
开始时间: τ
未来 K 个 slots 的链路可用性、额定速率和距离
当前 slot 的背景链路负载、有效速率和排队时延
```

之后，当前版本使用 slot-level min-cost max-flow，计算当前 slot 内可以端到端送达目标节点的数据量，并在满足容量约束的前提下最小化时延、能耗和断链风险构成的综合费用。到达 slot 边界后，路由模块根据最新链路状态、当前 slot 背景负载和剩余数据量重新规划。未来 slots 的链路可用性、额定速率和距离可用于估计断链风险，但未来 slots 的背景负载不作为 PPO 输入；路由模块也只在进入对应 slot 后读取该 slot 的背景负载。

### 4.2 背景负载感知边容量与边代价

对每条 ISL 边 e=(u,v)，slot t 内的有效传输速率为：

```text
R_eff,e(t) = R_e(t) · η_e(t)
```

在 slot t 的剩余可传输时间为：

```text
Δt_remain = slot_end(t) - current_time
```

因此，该边在当前 slot 中可承载的最大数据量可以定义为：

```text
Cap_e(t) = R_eff,e(t) · Δt_remain
```

这里不再额外使用 `Cap_e(t) = R_eff,e(t) · Δt_remain · (1 - ρ_e(t))`，因为 `R_eff,e(t)` 已经由 `ρ_e(t)` 折扣得到。`ρ_e(t)` 仍用于计算链路排队时延 `D_queue,e(t)`，但不再二次压缩容量。

边费用用于 min-cost max-flow 中的流量分配：

```text
cost_e(t) =
    α · edge_delay_per_gb(e,t)
  + β · edge_energy_per_gb(e,t)
  + δ · future_break_risk(e,t:t+K)
```

其中：

```text
edge_delay_per_gb(e,t)：包括在该边发送 1 GB 数据的发送时延、传播时延、当前 slot 背景排队时延和 switch penalty。路径级 delay 按 store-and-forward 逐边累加。
edge_energy_per_gb(e,t)：按当前链路发送 1 GB 数据的通信能耗估计。路径级 communication energy 按逐边发送能耗累加。
future_break_risk(e,t:t+K)：基于未来 K 个 slots 的 ISL 可用性预测得到，不使用未来背景负载。
```

### 4.3 Slot-level Min-cost Max-flow 数据传输

在节点选择智能体已经确定目标执行卫星后，当前 service hop 的数据传输可写成：

```text
source = 当前数据所在卫星 v_cur
target = next-hop microservice 的执行卫星 v_exec
remaining_data = B_hop
```

在每个 slot 内，构建当前可用的局部传输图：

```text
G_route(t) = (V, E_t)
```

其中边 e∈E_t 的容量为 `Cap_e(t)`。当前版本在该局部图上构造单源单汇 min-cost max-flow 问题：

```text
maximize    delivered_data(t)
minimize    Σ_e f_e(t) · cost_e(t)

subject to  0 ≤ f_e(t) ≤ Cap_e(t), ∀e∈E_t
            flow conservation at intermediate satellites
            source node = v_cur
            sink node   = v_exec
            delivered_data(t) ≤ remaining_data
```

该问题的含义是：先尽可能多地把当前 slot 能端到端送达目标节点的数据送达；在可送达数据量相同的情况下，选择综合费用最低的链路流量分配。

边费用定义为：

```text
cost_e(t) =
    α · edge_delay_per_gb(e,t)
  + β · edge_energy_per_gb(e,t)
  + δ · future_break_risk(e,t:t+K)
```

其中：

```text
edge_delay_per_gb(e,t)：由当前 slot 的有效速率、传播时延、排队时延和 switch penalty 得到，路径级时延按 store-and-forward 逐边累加；
edge_energy_per_gb(e,t)：按当前链路发送 1 GB 数据的通信能耗估计，路径级通信能耗按逐边发送能耗累加；
future_break_risk(e,t:t+K)：由未来 K 个 slots 的 ISL 可用性预测得到，不使用未来背景负载。
```

工程实现中使用残量网络和最短增广路求解当前 slot 的 min-cost max-flow。求解完成后，将正流分解为若干 source-to-target 路径，用于记录 `slot_paths`、传输时延、能耗和断链风险。

### 4.4 中继节点语义

当前版本采用端到端送达语义，不建模中继节点缓存：

```text
1. 若某部分数据在当前 slot 内沿候选路径完整到达 target，则计入 delivered_data；
2. 若某部分数据只到达中继节点而未到达 target，则不计入 delivered_data；
3. 未送达 target 的数据不保留在中继节点；
4. 下一 slot 重新规划时，remaining_data 仍从原 source/current_node 出发；
5. 因此，当前版不维护 relay buffers、fragment states 或 multi-source single-sink flow。
```

这种简化保留了路径内部的 store-and-forward 时延/能耗口径，但不记录中继节点上的跨 slot fragment，因此显著降低仿真状态复杂度。后续若需要更真实的链路层缓存模型，可扩展为记录每个中继节点上的 fragment，并在下一 slot 构造多源到单汇传输问题。

### 4.5 Baseline：K-shortest Paths + Bottleneck Allocation

当前代码主流程只保留 min-cost max-flow 路由。若后续需要做消融实验，可以重新引入 K-shortest paths + bottleneck capacity allocation 作为 baseline：

```text
1. 在 G_route(t) 上计算 source 到 target 的 K 条候选路径；
2. 对每条路径 P_k 计算瓶颈容量；
3. 根据路径总代价、瓶颈容量和链路断开风险对候选路径排序；
4. 按路径瓶颈容量为 remaining_data 分配可发送数据；
5. 只有能够在当前 slot 内端到端到达 target 的数据才计为 delivered_data。
```

如果进一步升级到跨 slot 中继缓存版本，还需要新增：

```text
relay_buffer[v]
fragment_id
fragment_remaining_data
multi-source single-sink routing state
```

### 4.6 分段重路由策略

当一段通信不能在当前 slot 内完成，或路径在下一 slot 断开时，采用分段重路由：

```text
1. 在当前 slot 内，根据当前 slot 的可用链路容量和边费用运行 min-cost max-flow，尽可能端到端送达 remaining_data；
2. 记录本 slot 实际送达目标节点的数据量；
3. 未送达目标节点的数据不保留在中继节点；
4. 到达 slot 边界时，更新拓扑、链路容量、当前 slot 背景负载和 remaining_data；
5. 如果 remaining_data > 0，则在下一 slot 仍以原 current_node 为源重新构造候选路径并继续传输；
6. 若当前 slot 无可行路径或候选路径容量为 0，则等待到下一 slot；
7. 若超过 route_horizon_slots 仍无法完成，则判定路由失败。
```


## 5. PPO-GNN Replica Selector 更新

### 5.1 决策时机

PPO-GNN 不是针对整条链一次性决策，而是在每个 service hop 开始时决策：

```text
当前数据所在卫星：v_cur
当前时间：τ
当前待执行微服务：m_i
候选副本集合：C_i(τ) = { v | x_{v,m_i}(τ)=1 且 replica 状态为 active }
```

这里的决策对象是：

```text
选择 next-hop microservice m_i 的执行卫星节点 v_exec。
```

在节点选择完成后，源节点 `v_cur` 和目标节点 `v_exec` 已经确定，后续数据传输交由 Cross-slot Route Planner 使用 slot-level min-cost max-flow 处理。

### 5.2 节点选择智能体输入


节点选择智能体的输入由三类信息组成。

第一类是当前请求的微服务链特征：

```text
1. 当前请求链 r = <v_src, m_1, ..., m_L, v_dst, τ_arr>；
2. 当前服务阶段 i 和剩余服务链长度 L-i+1；
3. 当前待执行微服务 m_i 的 CPU cycles、输入数据量、输出数据量和镜像大小；
4. 后续微服务序列或集合 {m_{i+1}, ..., m_L}；
5. 当前数据所在卫星 v_cur 与最终目的卫星 v_dst；
6. 请求到达时间、当前时间、当前 slot、当前 slot 剩余时间；
7. 当前已累计时延、能耗、slot crossings。
```

第二类是星座 ISL 通信条件。PPO-GNN 可以使用当前 slot 的完整通信状态，以及未来 K 个 time slots 内的确定性/可预测链路信息，但不使用未来 slots 的随机背景负载：

```text
1. 当前 slot 的星座拓扑 G_t = (V, E_t)；
2. 当前 slot 内每条 ISL 的可用性、额定速率 R_e(t)、有效速率 R_eff,e(t)、传播距离和传播时延；
3. 当前 slot 内背景链路利用率 ρ_e(t) 和链路排队时延 D_queue,e(t)；
4. 跨轨链路、同轨链路类型编码；
5. 未来 K 个 slots 内每条 ISL 的可用性预测；
6. 未来 K 个 slots 内每条 ISL 的额定速率 R_e(t+k) 和传播距离；
7. 基于未来 K 个 slots 链路可用性得到的断链风险和可持续时间；
8. 当前节点 v_cur 到各候选执行副本的当前 slot 可达性、min-cost flow 可送达容量、瓶颈容量和最低费用摘要。
```

注意，未来 K 个 slots 的输入只包括星座动力学可预测或链路表可预计算的信息，例如 ISL 是否可用、额定速率、传播距离和断链风险；不包括未来 slots 的 `ρ_e(t+k)`、`η_e(t+k)`、`R_eff,e(t+k)`、`D_queue,e(t+k)` 等由随机背景负载生成的信息。

第三类是微服务链上各微服务副本所部署卫星在当前 slot 的计算负载和算力水平：

```text
1. 当前待执行微服务 m_i 的 active candidate replicas 集合 C_i(s)；
2. 后续每个微服务 m_j, j >= i 的副本部署集合 C_j(s)，用于描述服务链后续执行环境；
3. 每个相关部署卫星的额定算力 F_v；
4. 每个相关部署卫星在当前 slot 的有效算力 F_eff,v(s)；
5. 当前 slot 的 Markov 计算负载状态 Z_v(s)；
6. 当前 slot 的计算利用率 ρ_v^c(s)；
7. 当前 slot 的微服务执行前等待时间 W_v(s)；
8. active replica count 和 draining 状态；
9. 每颗相关卫星当前部署服务数量、剩余部署容量和资源约束。
```

因此，智能体看到的是：

```text
当前请求链上下文
+ 当前 slot 的背景负载感知通信状态
+ 未来 K 个 slots 的 ISL 可用性、额定速率和距离预测
+ 当前 slot 内服务链相关微服务副本所在卫星的计算负载与算力状态
```

然后智能体输出当前 hop 的执行节点：

```text
v_exec = π_θ(state_s, candidate_set_s)
```


### 5.3 奖励函数设计

逐 hop 奖励建议为：

```text
r_i = -(
    α · step_delay
  + β · step_energy
  + γ · slot_crossings
  + δ · route_failure_risk
)
```

其中：

```text
step_delay = route_delay + compute_waiting_time + compute_execution_time
```

若发生路由失败、服务不可达或超出 deadline，则给予失败惩罚。



## 6. 计算执行模型

### 6.1 计算允许跨 Slot，但不中断

当数据到达某个服务副本后，微服务计算可以跨越多个 slots，但一旦开始执行，不进行抢占或迁移。


### 6.2 推荐版分段计算模型

按 slot 累计完成的 CPU cycles：

```text
remaining_cycles = CPU_cycles(m_i)
current_time = compute_start_time

while remaining_cycles > 0:
    slot = floor(current_time / Δt)
    slot_end = (slot+1)·Δt
    available_time = slot_end - current_time
    executable_cycles = F_eff,v(slot) · available_time
    done_cycles = min(remaining_cycles, executable_cycles)
    remaining_cycles -= done_cycles
    current_time += done_cycles / F_eff,v(slot)

compute_finish_time = current_time
```

在该模型中，计算过程不中断，但其执行速度随 slot 的有效计算能力变化而变化。

---

## 7. 微服务重部署与 Active Replica 规则

### 7.1 慢层更新频率

副本部署更新动作每 n=10 个 time slots 执行一次：

```text
migration_interval_slots = n
```

### 7.2 正在执行服务不受迁移影响

若某个微服务副本正在处理 service chain 的当前 service hop，则本次执行期间该副本保持有效。

### 7.3 已选择副本保持有效

已经选择的微服务副本在本次服务执行期间保持有效，即使慢层在执行过程中触发迁移，当前服务仍继续在原副本上完成。

### 7.4 后续服务重新选择副本

待执行的后续服务不绑定旧部署状态，而是在每个 service hop 开始时根据最新部署状态重新枚举候选副本并由 PPO-GNN 决策。

### 7.5 删除副本限制

删除副本时不能删除正在服务中的 active replica：

```text
if active_count(v,m) > 0:
    replica cannot be removed
```

### 7.6 Draining Replica 机制

迁移或删除时可使用 draining 状态：

```text
1. 被标记为 draining 的副本不再接收新的 service hop；
2. 正在执行的请求可以继续完成；
3. 当 active_count 降为 0 后，副本才真正删除；
4. 新增副本在镜像传输完成后才变为 active。
```

---

## 8. Bandit Replica Placement 更新

### 8.1 从 Hotness 到 Service Pressure

原始 hotness 只统计服务出现次数。引入背景负载后，慢层应使用 service pressure：

```text
service_pressure(m) =
    call_frequency(m)
  + λ1 · average_route_delay_to_m
  + λ2 · average_compute_waiting_time_of_m
  + λ3 · route_failure_count_related_to_m
  + λ4 · p95_delay_of_m
  + λ5 · replica_utilization_imbalance(m)
```

### 8.2 轨道平面级压力

对每个微服务 m 和轨道平面 p：

```text
plane_pressure(m,p) = demand(m,p) - capacity(m,p)
```

其中：

```text
demand(m,p)：来自轨道平面 p 或需要经过 p 的服务需求。
capacity(m,p)：轨道平面 p 上可用的 m 副本能力。
```

Bandit 的 move 动作应优先考虑：

```text
从低压力平面迁出；
向高压力平面迁入。
```

### 8.3 迁移动作约束

迁移动作需要满足：

```text
1. 每颗卫星部署容量约束；
2. 每个微服务副本数上下界；
3. active replica 不可删除；
4. draining replica 不接收新请求；
5. 新副本镜像传输成功后才可用；
6. 迁移成本由 cross-slot route planner 计算。
```

---

## 9. 服务链执行流程

完整执行流程如下：

```text
1. 请求在任意时间 τ 到达；
2. 确定当前 slot s = floor(τ / Δt)；
3. 对当前待执行微服务 m_i，枚举 active 候选副本 C_i(s)；
4. 节点选择智能体读取三类当前 slot 输入：
   - 当前请求的微服务链特征；
   - 当前 slot 的背景负载感知 ISL 通信条件，以及未来 K 个 slots 的 ISL 可用性、额定速率和距离预测；
   - 微服务链上各微服务副本所部署卫星在当前 slot 的计算负载和算力水平；
5. 对每个候选执行节点，基于当前 slot 通信图和当前 slot 计算状态构造 candidate feature，例如：
   - 当前 slot 可达性；
   - 当前 slot 最短/最低代价通信估计；
   - 当前 slot min-cost flow 的可送达容量、瓶颈容量和最低费用估计；
   - 当前 slot 链路拥塞和排队状态；
   - 未来 K 个 slots 内到候选节点方向的链路可用性和断链风险摘要；
   - 候选卫星当前计算等待、有效算力和执行时间；
   - 后续微服务副本分布形成的服务链上下文特征；
6. PPO-GNN Service Execution Node Selector 选择 next-hop microservice 的执行卫星节点 v_exec；
7. 在执行节点确定后，source=v_cur, target=v_exec 固定，Cross-slot Route Planner 开始负责数据传输；
8. 路由模块使用 min-cost max-flow，在当前 slot 基于可用 ISL、有效容量、边费用和 remaining_data 估计可端到端送达目标节点的数据量，并选择最低综合费用的链路流量分配；
9. 若当前 slot 未能完成传输，则在 slot 边界更新拓扑、链路容量、当前 slot 背景负载和 remaining_data；未送达目标的数据不保留在中继节点，下一 slot 仍从原 current_node 重新规划；
10. 数据到达 v_exec 后，在该卫星执行微服务 m_i，计算可跨 slot 但不中断；
11. 服务完成后，current_node 更新为 v_exec，进入下一微服务；
12. 整条链完成后，将结果路由到目的卫星 v_dst；
13. Bandit 每若干 slots 根据 service pressure 调整副本部署。
```

最终职责划分为：

```text
PPO-GNN Service Execution Node Selector:
    解决“next-hop microservice 在哪颗卫星上执行”。
    输入使用当前决策 slot 的通信状态、计算状态和请求链特征，并允许使用未来 K 个 slots 的 ISL 可用性、额定速率和距离预测；不使用未来 slots 的背景负载信息。

Cross-slot Route Planner:
    解决“数据如何从当前节点传到已选定的执行节点”。
    在源宿确定后，按 slot 构造容量-费用图。当前版本采用 min-cost max-flow，并坚持端到端送达计数；路径内部的时延和能耗按 store-and-forward 逐跳累加，未到达目标的数据不保留在中继节点。K-shortest paths + bottleneck allocation 仅作为后续消融实验 baseline。

Bandit Replica Placement Agent:
    解决“长期来看微服务副本应该如何增删迁移”。
    在慢时间尺度上根据 service pressure 调整副本部署。
```
