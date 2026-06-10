# This is a sample Python script.
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from microservices.packets import Packet
from satellites.satellite_node import Satellite_node
from service_pressure.microservices.microservice_flow import Microservice_flow
from utils.utils import *


class LCRA:
    def __init__(self, deployment, constellation):
        self.all_flows = []
        self.all_satellites = []
        self.deployment = deployment
        self.constellation = constellation
        self.max_throughput = 150
        self.completed_flows = defaultdict(list)  # 微服务流完成情况
        self.flow_input = []
        self.flow_output = []
        self.hop_distances = {}
        self.isl_distances = {}

    def compute_hop_distances(self):
        if not isinstance(self.constellation, nx.DiGraph):
            raise ValueError("Constellation must be a NetworkX DiGraph.")

        self.hop_distances = defaultdict(dict)
        nodes = list(self.constellation.nodes)
        for source in nodes:
            for target in nodes:
                if source == target:
                    self.hop_distances[source][target] = 0
                else:
                    try:
                        distance = nx.shortest_path_length(self.constellation, source=source, target=target)
                        self.hop_distances[source][target] = distance
                    except nx.NetworkXNoPath:
                        self.hop_distances[source][target] = float('inf')

    def compute_isl_distances(self):
        if not isinstance(self.constellation, nx.DiGraph):
            raise ValueError("Constellation must be a NetworkX DiGraph.")

        self.isl_distances = defaultdict(dict)
        nodes = list(self.constellation.nodes)
        for source in nodes:
            for target in nodes:
                if source == target:
                    self.isl_distances[source][target] = 0
                else:
                    try:
                        distance = nx.shortest_path_length(
                            self.constellation,
                            source=source, target=target,
                            weight=lambda x, y, z: 1 + self.get_satellite_by_id(y).total_queue_length / self.get_satellite_by_id(y).trans_throughput)
                        self.isl_distances[source][target] = distance
                    except nx.NetworkXNoPath:
                        self.isl_distances[source][target] = float('inf')

    def get_hop_distance(self, source, target):
        return self.hop_distances.get(source, {}).get(target, float('inf'))

    def get_isl_distance(self, source, target):
        return self.isl_distances.get(source, {}).get(target, float('inf'))

    def get_shortest_hop_distance(self, source, targets):
        min_distance = float('inf')
        for target in targets:
            distance = self.hop_distances.get(source, {}).get(target, float('inf'))
            if distance < min_distance:
                min_distance = distance
        return min_distance

    def get_shortest_isl_distance(self, source, targets):
        min_distance = float('inf')
        for target in targets:
            distance = self.isl_distances.get(source, {}).get(target, float('inf'))
            if distance < min_distance:
                min_distance = distance
        return min_distance

    def get_flow_by_id(self, id):
        return self.all_flows[id - 1]

    def get_satellite_by_id(self, id):
        return self.all_satellites[id - 1]

    def send_packets_to_src(self, flow, lambda_rate, static=False, info=None):
        # 向源卫星注入流量
        src = self.get_satellite_by_id(flow.src)
        if static:
            # 静态到达
            n_flows = len(self.all_flows)
            weights = [data['weight'] for _, _, data in self.constellation.edges(data=True)]
            c_avg = sum(weights) / len(weights)
            # num = int(c_avg * 4 / n_flows)
            num = int(c_avg * 4)
            dst = get_satellites_with_service(self.deployment, flow.root)
            for i in range(num):
                packet = Packet(flow, dst, flow.root, 0, info)
                src.isl.append(packet)
        else:
            # 泊松到达
            num = np.random.poisson(lambda_rate)
            dst = get_satellites_with_service(self.deployment, flow.root)
            for i in range(num):
                packet = Packet(flow, dst, flow.root, 0, info)
                src.isl.append(packet)

    def update_completed_flows(self, completed_flows):
        for flow_id in set(completed_flows.keys()):
            self.completed_flows[flow_id].extend(completed_flows[flow_id])

    def step(self, static=False):
        # TODO 实现 constellation 变化
        self.compute_hop_distances()
        self.completed_flows.clear()
        rates = {1: np.random.poisson(100), 2: np.random.poisson(100), 3: np.random.poisson(100),
                 4: np.random.poisson(100)}
        input = 0
        for flow in self.all_flows:
            self.send_packets_to_src(flow, rates[flow.id], static)
            input += rates[flow.id]
        self.flow_input.append(input)

        for satellite in self.all_satellites:
            satellite.receive_packets_lcra()
            satellite.increment_life_time_lcra()
            satellite.step_c_lcra()

        for satellite in self.all_satellites:
            satellite.step_t_lcra()


if __name__ == "__main__":
    deployment, constellation = load_deployment_and_topology("graph/6_12_4_6_deployment.json")
    constellation = fill_weights(constellation, False, None)
    env = LCRA(deployment, constellation)
    for i in range(4):
        root, dependency = load_root_and_dependency_graph(f"graph/dependency_60_{i + 1}.json")
        flow = Microservice_flow(1 + i, dependency, root, 1 + i * 5, 72 - i * 3)
        env.all_flows.append(flow)
    for i in range(72):
        env.all_satellites.append(Satellite_node(i + 1, constellation, deployment, env))

    t = 0
    # 初始化存储每个flow的临时吞吐量和延迟
    temp_throughput = [[] for _ in range(4)]  # 4个flow的吞吐量临时存储
    temp_delay = [[] for _ in range(4)]  # 4个flow的延迟临时存储

    # 初始化存储每个flow的平均吞吐量和延迟
    avg_throughput = [[] for _ in range(4)]  # 每个flow的吞吐量均值
    avg_delay = [[] for _ in range(4)]  # 每个flow的延迟均值

    while t < 10000:
        env.step(False)

        # 计算每个flow的throughput和delay
        flow_output = [len(env.completed_flows[i]) for i in range(1, 5)]
        env.flow_output.append(sum(flow_output))
        flow_delay = [
            np.mean(env.completed_flows[i]) if len(env.completed_flows[i]) > 0 else float(
                'inf')
            for i in range(1, 5)
        ]

        # 将每个flow的数据加入临时存储
        for i in range(4):
            temp_throughput[i].append(flow_output[i])  # 每个flow的吞吐量
            temp_delay[i].append(flow_delay[i])  # 每个flow的延迟

        print(f'time slot {t}, throughput {flow_output}, delay {flow_delay}.')

        # 每隔100个时间单位计算平均值
        if (t + 1) % 100 == 0:
            for i in range(4):
                # 计算每个flow的吞吐量和延迟均值
                avg_throughput[i].append(np.mean(temp_throughput[i]))
                avg_delay[i].extend(temp_delay[i])
                # 重置临时存储
                temp_throughput[i] = []
                temp_delay[i] = []
            for satellite in env.all_satellites:
                print(f"satellite {satellite.id}: {len(satellite.queue_t) + len(satellite.queue_c)}")

        t += 1

    # 输出每个flow的平均吞吐量和延迟
    print("Average Throughput for each flow:", avg_throughput)
    print("Average Delay for each flow:", avg_delay)

    # 绘制 throughput 图像
    for i in range(4):  # 对每个 flow 进行绘图
        plt.plot(range(len(avg_throughput[i])), avg_throughput[i], label=f'Flow {i + 1}')

    # 添加图例、标题和坐标轴标签
    plt.title('Average Throughput of Each Flow')
    plt.xlabel('Time Slot (x100)')
    plt.ylabel('Throughput')
    plt.legend()
    plt.grid(True)

    # 显示图像
    plt.show()

    filtered_avg_delay = {}
    p90_delay = {}
    for flow_id, delays in enumerate(avg_delay):
        # 剔除 inf 值
        valid_delays = [d for d in delays if not np.isinf(d)]
        # 计算平均值
        filtered_avg_delay[flow_id] = np.mean(valid_delays) if valid_delays else 0
        index = int(len(valid_delays) * 0.9)
        valid_delays.sort()
        p90_delay[flow_id] = valid_delays[index]

    # 绘制柱状图
    flow_ids = list(filtered_avg_delay.keys())
    average_delays = list(filtered_avg_delay.values())
    p90_delays = list(p90_delay.values())

    print(
        f'output: {np.mean(env.flow_output)}, input: {np.mean(env.flow_input)}, deliver rate: {np.mean(env.flow_output) / np.mean(env.flow_input)}')
    print(f'avg: {np.mean(average_delays)}, p90: {np.mean(p90_delays)}')

    bar_width = 0.4  # 设置柱的宽度
    x = np.arange(len(flow_ids))  # 为每个flow设置位置

    # 绘制并列柱
    plt.bar(x - bar_width / 2, average_delays, width=bar_width, color='skyblue', alpha=0.8, label='Filtered Avg Delay')
    plt.bar(x + bar_width / 2, p90_delays, width=bar_width, color='orange', alpha=0.8, label='P90 Delay')

    # 添加标题和坐标轴标签
    plt.title('Filtered Average Delay and P90 Delay of Each Flow')
    plt.xlabel('Flow ID')
    plt.ylabel('Delay')
    plt.xticks(x, [f'Flow {fid + 1}' for fid in flow_ids])  # 设置x轴标签
    plt.legend()  # 显示图例
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 显示图像
    plt.show()
