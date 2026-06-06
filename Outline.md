# 卫星微服务三层协同仿真实现大纲

## 1. 背景、动机与总体架构

### 1.1 研究背景

低轨卫星网络由数以百计甚至更多的低轨卫星节点组成，卫星之间通过星间链路（Inter-satellite Link, ISL）形成动态网络拓扑。按照链路所连接卫星的轨道关系，ISL 可以分为两类：

```text
同轨 ISL：连接同一轨道平面内的相邻卫星
跨轨 ISL：连接相邻轨道平面内的卫星
```

同轨 ISL 中卫星之间的相对位置较为稳定，因此链路可用性、传播距离和通信质量通常较稳定。跨轨 ISL 则更容易受到极区、可视范围、多普勒效应和相对运动变化的影响，链路可用性和有效传输速率具有明显时变特征。因此，低轨卫星网络不是一个静态通信网络，而是一个具有周期性、时变性和链路异质性的动态网络。

与此同时，随着卫星载荷、散热能力和星上计算硬件的发展，低轨卫星不再只是通信转发节点，也逐渐具备一定的通用计算、存储和内存资源，可以承担部分星上数据处理任务。但是，单颗低轨卫星仍然受到板载空间、供电能力、散热能力和存储容量的限制，难以像地面数据中心一样部署和运行体量较大的单体应用。

微服务架构为星上计算提供了可行的软件组织方式。它将大规模单体应用拆分为若干功能相对独立、资源需求较小的微服务，使每个微服务能够单独部署、独立执行。对于低轨卫星星座而言，可以将一个应用拆分后的微服务集合以分布式方式部署在多颗卫星上，通过多星协同的方式完成用户任务，从而形成在轨服务能力。

### 1.2 研究动机

提供在轨服务的核心动机主要来自两个方面。

第一，低轨卫星采集的数据量通常很大。例如卫星云图、地质勘探、海洋观测和灾害监测等任务可能产生 GB 甚至几十 GB 级原始数据。如果全部通过星地链路回传到地面数据中心再处理，会面临很高的通信成本和不稳定性。星地链路受到大气衰减、可视窗口、多普勒效应和地面站覆盖限制的影响，无法始终提供稳定、低时延、高吞吐的数据回传能力。

第二，卫星本身已经具备一定通用计算能力，并且可以通过太阳能获取相对经济的能源供给。因此，如果能够在星座内部合理调度计算任务，先在轨完成数据预处理、筛选、压缩或特征提取，再将处理结果回传地面，就有机会显著降低星地链路负担，提高任务响应速度。

但是，把微服务部署到低轨卫星星座并不意味着问题自然解决。实际系统至少面临三类挑战：

```text
1. 微服务副本部署问题：
   每颗卫星资源有限，每个微服务需要维护若干副本，副本数量、位置和迁移成本需要动态权衡。

2. 微服务执行节点选择问题：
   对于请求链中的每个微服务，需要在多个候选副本中选择执行节点，同时考虑通信时延、计算等待、执行时延和能耗。

3. 星间数据路由问题：
   服务链执行过程中需要在卫星之间传输中间数据，而 ISL 尤其是跨轨链路具有时变性，单条最短路径容易在 slot 切换时失效。
```

因此，本研究的目标不是单独优化“服务部署”或“路由”，而是构建一个慢层部署调整、快层执行决策和路由层数据传输协同工作的完整仿真框架。

### 1.3 请求链建模

在实际场景中，地面用户或星上任务通常不会调用应用中的全部微服务，而是调用其中一个子集。该请求可以建模为一条微服务调用链：

```text
<m_1, m_2, ..., m_L>
```

其中 `L` 表示服务链长度，`m_i` 表示第 `i` 个微服务。每个微服务具有计算开销，服务 `m_i -> m_{i+1}` 之间具有通信数据量开销。

一个完整请求进一步建模为：

```text
<v_src, m_1, m_2, ..., m_L, v_dst, t_arr>
```

其中：

```text
v_src：请求起点卫星
v_dst：请求终点卫星
t_arr：请求到达时间
```

每个微服务采用无状态副本部署方式，即同一个微服务可以在多颗卫星上部署等价副本，副本之间无需状态同步。这一假设使得微服务副本可以被新增、删除和迁移，也使快层代理能够在多个候选副本中选择当前最合适的执行位置。

### 1.4 时变拓扑建模

根据低轨卫星运动规律和 ISL 建模策略，整个星座运行周期被划分为若干 time slots：

```text
Delta t_1, Delta t_2, ..., Delta t_T
```

当前仿真中每个 slot 统一设置为 10 秒。在一个 time slot 内，网络拓扑、链路速率、链路距离和卫星额定负载近似保持不变；不同 time slots 之间，这些状态会随卫星运动和负载变化而更新。

由于一条微服务请求链的通信和计算过程可能跨越多个 time slots，系统需要在动态拓扑上持续做三类决策：

```text
1. 当前微服务 hop 应选择哪个副本执行 
2. 两个卫星节点之间的数据应如何跨 slot 传输
3. 哪些微服务副本应该新增、删除或迁移
```

### 1.5 总体架构

当前仿真面向 Walker Delta 低轨卫星星座中的微服务请求链执行问题。系统采用三层协同架构：

```text
慢层：Bandit-based Replica Placement / Migration Agent
快层：PPO + GNN Service Execution Agent
路由层：Orbit-aware Latency-first Dual-path Routing
```

整体目标是在星间链路快速变化、卫星计算能力有限、每星部署容量有限的条件下，尽可能降低微服务链端到端时延、能耗和执行失败率。

## 2. 当前实验配置

星座配置：

```text
轨道平面数：10
每个轨道平面卫星数：18
卫星总数：180
ISL time slot 数：606
统一 slot 时长：10 s
```

微服务配置：

```text
微服务总数：30
每颗卫星最多部署微服务数：3
每个微服务最少副本数：5
每个微服务最多副本数：10
CPU cycles：按配置范围随机生成，或从固定请求模板中复用
```

请求配置：

```text
共 14 条请求模板：
8 条长度为 5 的请求链
4 条长度为 10 的请求链
2 条长度为 15 的请求链
```

训练阶段按 time slot 逐步运行。每个 slot 内，14 条请求模板分别按泊松分布到达，到达时间在当前 slot 的 10 秒范围内随机生成。

## 3. 慢层：多臂老虎机副本部署调整

### 3.1 模块职责

慢层位于 `Simulation/migration.py`，核心类为：

```python
ReplicaPlacementMigrationAgent
```

它每隔若干 time slots 根据最近窗口内的请求热度和历史 arm 奖励，调整微服务副本部署。当前默认每 10 个 time slots 执行一次。

支持动作：

```text
add     在目标轨道平面内新增一个微服务副本
remove  从源轨道平面内删除一个已有副本
move    将副本从源轨道平面迁移到目标轨道平面
no-op   不调整部署，仅作为候选 arm
```

### 3.2 Bandit Arm 设计

当前采用轨道平面级 coarse-grained arm：

```text
arm = (action, service_id, source_plane, target_plane)
```

示例：

```text
("move", 12, 3, 7)
```

表示将微服务 12 的一个副本从轨道平面 3 迁移到轨道平面 7。

采用轨道平面级 arm 的原因是降低动作空间规模，提高 arm 在训练过程中的重复命中率，使 UCB 能从持续探索进入有效利用阶段。

### 3.3 候选 Arm 生成

慢层首先根据最近窗口内请求链统计服务热度：

```text
hotness(m) = m 在最近窗口所有请求链中出现的次数
```

然后根据当前副本部署状态生成候选：

```text
若 replicas(m) < 10，则可生成 add(m, target_plane)
若 replicas(m) > 5，则可生成 remove(m, source_plane)
若目标平面有空闲卫星，则可生成 move(m, source_plane, target_plane)
```

卫星部署容量约束：

```text
sum_m x(v,m) <= 3
```

副本数量约束：

```text
5 <= replicas(m) <= 10
```

### 3.4 轨道平面 Arm 到具体卫星动作的解析

Bandit 先选择轨道平面级 arm，然后解析为具体卫星级操作：

```text
add(m, target_plane):
    在 target_plane 中选择一个未部署 m 且部署数量最少的卫星 b
    从 m 的已有副本中选择到 b 迁移成本最低的源卫星 a*
    调用路由层传输服务镜像
    在 b 上新增 m 的副本

remove(m, source_plane):
    在 source_plane 中选择一个已有 m 副本的卫星 a
    删除 a 上的 m 副本

move(m, source_plane, target_plane):
    在 source_plane 中选择一个已有 m 副本的卫星 a
    在 target_plane 中选择一个未部署 m 且有空闲容量的卫星 b
    调用路由层传输服务镜像
    删除 a 上的旧副本，在 b 上新增副本
```

### 3.5 UCB 选择策略

候选 arm 按 UCB 分数排序：

```text
UCB(arm) = mean_reward(arm)
         + c * sqrt(log(total_pulls + 1) / pull_count(arm))
```

其中：

```text
c = 1.25
未尝试过的 arm 的 UCB 分数为 +inf，优先探索
```

每次慢层决策最多执行 4 个有效动作。

### 3.6 奖励设计

当前 Bandit 有两类奖励信号。

第一类是决策时的估计奖励：

```text
estimated_reward = ExpectedSaving - MigrationCost
```

服务热度收益：

```text
base = hotness(m) * 0.1

ExpectedSaving(add)    = base
ExpectedSaving(move)   = 0.75 * base
ExpectedSaving(remove) = max(0, 0.25 - 0.1 * base)
```

迁移成本：

```text
MigrationCost =
    delay_weight * migration_route_delay
  + energy_weight * migration_energy / 1000
  + slot_switch_penalty_weight * slot_crossings
  + migration_weight * image_size
  + 0.01 * startup_delay
  + move_extra_cost
```

其中 `move_extra_cost=0.02`，`remove` 的成本固定为 `0.01`。

若：

```text
estimated_reward <= migration_safety_margin
```

则该动作不会真正执行。

第二类是执行后的反馈奖励：

```text
execution_quality =
    success_rate
  - delay_weight  * average_delay / 100
  - energy_weight * average_energy / 10000

execution_feedback_reward =
    execution_quality - 0.1 * migration_cost
```

反馈奖励用于评价上一个慢层窗口中已经执行过的迁移动作，对同一个 arm 的统计量继续更新。

### 3.7 训练过程

训练脚本中，每个 epoch 对应一个 time slot。默认流程：

```text
1. 在当前 slot 生成泊松到达请求
2. 快层执行这些请求
3. 把请求和执行结果加入 Bandit 窗口缓存
4. 每 10 个 slot：
   a. 使用当前窗口结果反馈上一轮迁移动作
   b. 根据当前窗口请求热度选择新的迁移动作
   c. 将新动作设为 pending actions，等待下一窗口反馈
5. 星座周期结束时：
   a. 反馈仍 pending 的迁移动作
   b. 重置卫星部署状态
   c. 保留 Bandit arm_stats，继续跨周期学习
```

### 3.8 策略保存与测试阶段复用

训练阶段保存：

```text
bandit_actions.csv
bandit_arm_stats.csv
```

其中 `bandit_arm_stats.csv` 保存每个 arm 的：

```text
pull_count
reward_sum
mean_reward
estimated_count
estimated_mean_reward
execution_count
execution_mean_reward
```

测试阶段 `Simulation/evaluate.py` 默认读取：

```text
model_dir/bandit_arm_stats.csv
```

并恢复 UCB 所需的 arm 统计量，使测试阶段复用训练好的 Bandit 策略。若需要从空 Bandit 开始测试，可使用：

```powershell
python -m Simulation.evaluate --no-load-bandit
```

### 3.9 Bandit 伪代码

```text
Algorithm 1: Bandit-based Replica Placement / Migration

Input:
    microservice replicas R
    recent requests W
    arm statistics S
    constellation context G_t

Output:
    migration action list A

1. hotness <- Count service appearances in W
2. service_count_by_node <- Count deployed services on each satellite
3. candidate_arms <- {("no-op", 0, None, None)}
4. for each hot service m:
5.     for each available target plane p_t:
6.         if replicas(m) < 10 and p_t has feasible target:
7.             add ("add", m, None, p_t) to candidate_arms
8.     for each source plane p_s hosting m:
9.         if replicas(m) > 5:
10.            add ("remove", m, p_s, None)
11.        for each available target plane p_t != p_s:
12.            if p_t has feasible target:
13.                add ("move", m, p_s, p_t)
14. Sort candidate_arms by UCB score descending
15. A <- empty list
16. for arm in candidate_arms:
17.     if |A| >= 4: break
18.     resolve arm to concrete source/target satellites
19.     if action is illegal: continue
20.     migration_cost <- Route-aware migration cost
21.     expected_saving <- Hotness-based expected saving
22.     reward <- expected_saving - migration_cost
23.     update S[arm] with estimated reward
24.     if reward <= migration_safety_margin: continue
25.     apply add/remove/move to replica placement R
26.     append concrete action to A
27. return A
```

## 4. 快层：PPO + GNN 微服务执行代理

### 4.1 模块职责

快层位于 `Simulation/ppo_gnn_agent.py`，核心类为：

```python
PPOGNNExecutionAgent
```

它在每个微服务 hop 选择当前服务 `m_i` 应在哪个副本卫星上执行。

动作空间为当前微服务的候选副本集合：

```text
C_i = {v | satellite v hosts service m_i}
```

不可达副本会被 mask，不进入 softmax 采样。

### 4.2 状态编码

当前状态由三类信息组成。

卫星图节点特征 `encode_satellite_graph()`：

```text
1. CPU 基准频率归一化
2. 当前 slot 队列等待时延归一化
3. 当前卫星已部署服务数量 / 最大容量
4. 轨道平面编号归一化
5. 轨道内卫星位置归一化
6. 是否为当前节点
7. 是否为请求终点节点
8. 是否部署当前服务
9. 图节点度数归一化
```

服务链特征 `encode_service_chain()`：

```text
1. 链长度归一化
2. 当前服务位置归一化
3. 剩余服务比例
4. 当前服务 CPU cycles 归一化
5. 当前服务镜像大小归一化
6. 请求链总数据量归一化
```

候选副本额外特征：

```text
1. 路由通信时延归一化
2. 队列等待时延归一化
3. 计算执行时延归一化
4. 通信与计算能耗归一化
```

### 4.3 网络结构

当前网络由三部分组成：

```text
GraphMessagePassing:
    对卫星图节点特征做多层邻居平均消息传递

Service Chain Projection:
    将服务链特征映射到 hidden_dim

Candidate Scoring Head:
    concat(candidate_embedding,
           current_node_embedding,
           destination_embedding,
           chain_embedding,
           candidate_extra_features)
    -> MLP
    -> candidate logit
```

同时使用 value head 输出当前全局状态价值：

```text
V(s) = MLP(concat(current_node_embedding,
                 destination_embedding,
                 chain_embedding))
```

### 4.4 Candidate Masked Softmax

对于每个候选副本，先调用路由层和计算模型估计：

```text
route(current_node, candidate_node)
compute(service_i, candidate_node)
```

若路由不可达，则该候选副本被标记为 unreachable，不参与 softmax。

对 reachable candidates 计算 logits：

```text
pi(a_i | s) = softmax(logits over reachable candidates)
```

训练模式下从该分布采样；测试模式下选择概率最大的副本。

### 4.5 逐步奖励

每个微服务 hop 执行完成后产生逐步奖励：

```text
r_i = -(
    delay_weight * step_delay
  + energy_weight * step_energy / 1000
  + slot_switch_penalty_weight * slot_crossings
)
```

其中：

```text
step_delay = communication_delay + queue_delay + compute_delay
step_energy = communication_energy + compute_energy
```

若某个候选副本全部不可达或路由失败，则记录失败惩罚。

### 4.6 PPO 更新

当前训练中，每收集若干 time slots 的 transitions 后执行一次 PPO 更新，默认：

```text
ppo_update_slots = 5
```

PPO 更新过程：

```text
Return_t = r_t + gamma * Return_{t+1}
Advantage_t = Return_t - V(s_t)
ratio = exp(log_prob_new - log_prob_old)
policy_loss = -mean(min(ratio * A, clip(ratio) * A))
value_loss = MSE(V(s), Return)
loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
```

训练结束时保存：

```text
ppo_gnn_latest.pth
ppo_gnn_epoch_{final_epoch}.pth
```

### 4.7 PPO + GNN 伪代码

```text
Algorithm 2: PPO-GNN Service Execution

Input:
    request chain [m_1, ..., m_L]
    source satellite v_src
    destination satellite v_dst
    current replica placement R
    dynamic graph G_t

Output:
    execution plan and transitions

1. current_node <- v_src
2. current_time <- request arrival time
3. for i = 1 to L:
4.     service <- m_i
5.     candidates <- satellites hosting service
6.     node_features <- EncodeSatelliteGraph(G_t, current_node, v_dst, service)
7.     chain_features <- EncodeServiceChain(request, i)
8.     for each candidate c in candidates:
9.         route_c <- RouteData(current_node, c)
10.        if route_c is unreachable:
11.            mask c
12.        else:
13.            compute_c <- EstimateCompute(service, c, route_c.arrival_time)
14.            candidate_extra_c <- [route delay, queue delay, compute delay, energy]
15.    if all candidates masked:
16.        record failure penalty and terminate
17.    h <- GNN(node_features, adjacency)
18.    z_chain <- Linear(chain_features)
19.    logits <- CandidateScoringHead(h, z_chain, candidate_extra)
20.    probs <- MaskedSoftmax(logits)
21.    if training:
22.        selected <- sample(probs)
23.    else:
24.        selected <- argmax(probs)
25.    execute route and compute on selected
26.    r_i <- step reward
27.    store transition(log_prob, value, reward, entropy, done)
28.    current_node <- selected
29.    current_time <- compute finish time
30. route output data from current_node to v_dst
31. return execution result

Algorithm 3: PPO Update

Input:
    collected transitions D

1. Compute discounted returns R_t
2. Compute advantages A_t = R_t - V(s_t)
3. Normalize advantages
4. For each PPO update epoch:
5.     ratio <- exp(log_prob_new - log_prob_old)
6.     clipped_ratio <- clip(ratio, 1-epsilon, 1+epsilon)
7.     policy_loss <- -mean(min(ratio*A, clipped_ratio*A))
8.     value_loss <- MSE(V(s), R)
9.     entropy_bonus <- mean(entropy)
10.    loss <- policy_loss + value_coef*value_loss - entropy_coef*entropy_bonus
11.    Backpropagate loss
12.    Clip gradients
13.    Update network parameters
14. Clear transition buffer
```

## 5. 路由层：轨道感知低时延双路径路由

### 5.1 模块职责

路由层位于 `Simulation/routing.py`，核心函数为：

```python
route_data(source, target, data_gb, start_time, context)
```

它为以下传输过程提供统一路由：

```text
请求源卫星 -> 服务副本
服务副本 -> 下一个服务副本
最后一个服务副本 -> 请求终点卫星
微服务副本 add/move 过程中的镜像传输
```

### 5.2 链路代价

单条边的单位代价：

```text
edge_unit_delay =
    1 GB / link_rate
  + propagation_delay
  + switch_penalty
```

其中：

```text
propagation_delay = distance / speed_of_light
switch_penalty = 0.02 s
```

### 5.3 最短路径预计算与缓存

仿真构建阶段会为每个 time slot 预计算全源最短路径树：

```text
global shortest path trees
same-plane restricted shortest path trees
```

运行时使用：

```text
cached_shortest_path_for_slot()
```

避免在训练中重复运行 Dijkstra。

同时对完整路由结果做缓存：

```text
route_cache_key = (source, target, data_gb, start_time)
```

每个 epoch 结束后清空 route result cache，避免跨 slot 保存过多细粒度路径结果。

### 5.4 同轨道路由

若源卫星和目标卫星在同一个轨道平面：

```text
1. 只在同轨道平面内部搜索最短路径
2. 计算传输时延、传播时延、能耗
3. 返回 same_orbit_shortest_path
```

同轨道不启用双路径并行传输。

### 5.5 跨轨道双路径路由

若源卫星和目标卫星不在同一个轨道平面：

```text
1. 在当前 slot 计算当前最短路径 P_now
2. 在下一 slot 计算下一 slot 最短路径 P_next
3. 若 P_next 当前也可用，且不同于 P_now，则加入并行路径集合
4. 当前 slot 最多维护两条并行路径
5. 按路径容量比例分配数据量
6. 若当前 slot 剩余时间不足导致容量为 0，则推进到下一 slot 重新计算
7. 最多跨越 route_horizon_slots=3 个 slots
8. 超过 horizon 仍未完成则判定路由失败
```

路径容量估计：

```text
capacity(path) =
    bottleneck_rate * usable_time / hop_penalty

usable_time = remaining_slot_time - propagation_time
hop_penalty = 1 + hop_penalty_lambda * hop_count
```

传输分配：

```text
data_on_path_i =
    remaining_data * capacity_i / sum(capacity_j)
```

### 5.6 失败判定

路由失败原因包括：

```text
same_orbit_no_path
cross_orbit_no_current_path
zero_capacity
route_horizon_exceeded
```

其中 `zero_capacity` 只有在已经推进到 horizon 最后一个 slot 后仍没有传输容量时才失败；如果还在 horizon 内，则推进到下一个 slot 重新计算路径。

### 5.7 路由伪代码

```text
Algorithm 4: Orbit-aware Latency-first Dual-path Routing

Input:
    source satellite s
    target satellite d
    data amount B
    start time t
    dynamic graph snapshots {G_t}

Output:
    route result

1. if s == d or B <= 0:
2.     return local route
3. if s and d are in the same orbit plane:
4.     slot <- SlotFromTime(t)
5.     P <- Same-plane shortest path in slot
6.     if P does not exist:
7.         return failure(same_orbit_no_path)
8.     delay, energy <- PathDelayAndEnergy(P, B)
9.     return same_orbit_shortest_path result
10. else:
11.    current_time <- t
12.    remaining_data <- B
13.    for offset = 0 to route_horizon_slots - 1:
14.        slot <- SlotFromTime(current_time)
15.        remaining_time <- slot_end_time - current_time
16.        P_now <- Global shortest path in current slot
17.        if P_now does not exist:
18.            return failure(cross_orbit_no_current_path)
19.        parallel_paths <- {P_now}
20.        if offset < route_horizon_slots - 1:
21.            P_next <- Global shortest path in next slot
22.            if P_next exists and P_next != P_now and P_next usable now:
23.                parallel_paths <- parallel_paths union {P_next}
24.        for each path P in parallel_paths:
25.            capacity_P <- EstimateCapacity(P, remaining_time)
26.        if sum(capacity_P) <= 0:
27.            if offset < route_horizon_slots - 1:
28.                current_time <- next slot start
29.                continue
30.            else:
31.                return failure(zero_capacity)
32.        for each path P in parallel_paths:
33.            allocated_data <- remaining_data * capacity_P / sum(capacity)
34.            sent_data <- min(allocated_data, capacity_P, remaining_data)
35.            delay_P, energy_P <- PathDelayAndEnergy(P, sent_data)
36.            update total delay, total energy and slot path records
37.            remaining_data <- remaining_data - sent_data
38.        if remaining_data <= epsilon:
39.            return cross_orbit_dual_path result
40.        current_time <- next slot start
41.    return failure(route_horizon_exceeded)
```

## 6. 训练与测试入口

主要代码模块：

```text
Simulation/config.py                    全局仿真参数
Simulation/topology.py                  读取 Walker Delta ISL CSV
Simulation/constellation.py             卫星编号与轨道映射
Simulation/service.py                   卫星资源与微服务资源模型
Simulation/request.py                   请求模板与泊松到达生成
Simulation/routing.py                   路由层
Simulation/migration.py                 Bandit 慢层
Simulation/ppo_gnn_agent.py             PPO-GNN 快层
Simulation/env.py                       仿真环境
Simulation/train.py                     固定请求模板训练入口
Simulation/train_dyn.py                 固定链结构、随机源终点训练入口
Simulation/evaluate.py                  测试入口
Simulation/pics/plot_training_curves.py 训练曲线绘制
```

训练输出：

```text
request_templates.csv
training_metrics.csv
request_metrics.csv
bandit_actions.csv
bandit_arm_stats.csv
ppo_gnn_latest.pth
training_run_summary.json
```

测试默认复用：

```text
request_templates.csv
ppo_gnn_latest.pth
bandit_arm_stats.csv
```

## 7. 评价指标

系统级指标：

```text
RequestSuccessRate
AverageEndToEndDelay
P95EndToEndDelay
AverageEnergyConsumption
AverageRewardPerRequest
AverageRewardPerHop
```

Bandit 指标：

```text
BanditTotalPulls
KnownArmCount
PositiveArmCount
AverageArmReward
TotalAppliedActions
ExecutionFeedbackUpdates
AppliedAddCount
AppliedRemoveCount
AppliedMoveCount
RejectedNonPositiveArmCount
UnresolvedArmCount
```

PPO-GNN 指标：

```text
PPOPolicyLoss
PPOValueLoss
PPOEntropy
PPOTransitionCount
AverageRewardPerRequest
AverageRewardPerHop
```

路由指标：

```text
AverageCommunicationDelay
AverageSlotCrossings
RouteModeDistribution
RoutingRouteResultHits
RoutingRouteResultMisses
RoutingShortestPathHits
RoutingShortestPathMisses
```

## 8. 推荐运行命令

训练：

```powershell
python -m Simulation.train
```

动态源终点训练：

```powershell
python -m Simulation.train_dyn
```

测试，默认复用 PPO 和 Bandit 策略：

```powershell
python -m Simulation.evaluate --model-dir .\Simulation\fix_rep_pattern_train_data --output-dir .\Simulation\eval_outputs
```

测试，不加载训练好的 Bandit：

```powershell
python -m Simulation.evaluate --model-dir .\Simulation\fix_rep_pattern_train_data --no-load-bandit
```

绘图：

```powershell
python .\Simulation\pics\plot_training_curves.py --input-dir .\Simulation\fix_rep_pattern_train_data --window 50
```
