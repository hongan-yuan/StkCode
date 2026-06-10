import heapq
import json
import random
from collections import defaultdict

import networkx as nx
from matplotlib import pyplot as plt
from networkx.readwrite import json_graph

from microservices.microservice_flow import Microservice_flow


def create_leo_constellation(num_orbits, num_sats_per_orbit):
    """
    创建一个LEO星座的网络拓扑。

    参数:
        num_orbits (int): 轨道数量
        num_sats_per_orbit (int): 每个轨道上的卫星数量

    返回:
        nx.DiGraph: 表示LEO星座拓扑的图
    """
    constellation_graph = nx.DiGraph()

    for orbit in range(num_orbits):
        for sat in range(num_sats_per_orbit):
            u = num_sats_per_orbit * orbit + sat + 1
            constellation_graph.add_node(u)

            next_sat = (sat + 1) % num_sats_per_orbit
            v = num_sats_per_orbit * orbit + next_sat + 1
            constellation_graph.add_edge(u, v)
            constellation_graph.add_edge(v, u)

    for orbit in range(num_orbits):
        for sat in range(num_sats_per_orbit):
            u = num_sats_per_orbit * orbit + sat + 1
            next_orbit = (orbit + 1) % num_orbits
            v = num_sats_per_orbit * next_orbit + sat + 1
            constellation_graph.add_edge(u, v)
            constellation_graph.add_edge(v, u)

    constellation_graph = fill_weights(constellation_graph, False, None)
    return constellation_graph


def fill_weights(constellation_graph, randomize=False, seed=None):
    """
    为给定的 constellation_graph 填充节点和边的权重（卫星计算吞吐量和链路传输吞吐量）。

    参数:
        constellation_graph (nx.Graph): 星座图。
        randomize (bool): 是否随机化权重。如果为 False，则使用固定权重。
        seed (int): 随机种子。如果 randomize 为 True，则用于控制随机化的结果一致性。

    返回:
        nx.Graph: 填充权重后的星座图。
    """
    if randomize and seed is not None:
        random.seed(seed)

    for node in constellation_graph.nodes:
        if randomize:
            constellation_graph.nodes[node]["weight"] = random.randint(200, 400)
        else:
            constellation_graph.nodes[node]["weight"] = 300

    for edge in constellation_graph.edges:
        if randomize:
            constellation_graph.edges[edge]["weight"] = random.randint(50, 100)
        else:
            constellation_graph.edges[edge]["weight"] = 75

    return constellation_graph


def deploy_microservices(num_microservices, constellation_graph, max_replicas, max_deployments):
    """
    将微服务部署在LEO星座网络中，满足以下约束条件：
    1. 每颗卫星上至少部署一个微服务；
    2. 每个微服务至少布置在一颗卫星上；
    3. 每个微服务的副本数在 1 到 max_replicas 之间；
    4. 每个卫星的部署总数不超过 max_deployments。

    参数:
        num_microservices (int): 微服务数量。
        constellation_graph (nx.Graph): LEO星座网络拓扑。
        max_replicas (int): 每个微服务的最多副本数。
        max_deployments (int): 每个卫星的最多部署数。

    返回:
        dict: 微服务部署策略，格式为 {vi: [m1, m2, ...], ...}。
    """
    satellites = list(constellation_graph.nodes)
    deployment = {sat: [] for sat in satellites}
    microservice_allocation = {f"m{i}": [] for i in range(1, num_microservices + 1)}

    # 初始化可用的卫星
    available_sats = satellites.copy()

    # 为每个微服务分配至少一个副本
    for microservice in microservice_allocation:
        if not available_sats:
            available_sats = satellites.copy()
        sat = random.choice(available_sats)
        deployment[sat].append(microservice)
        microservice_allocation[microservice].append(sat)
        available_sats.remove(sat)

    # 确保每个卫星至少部署一个微服务
    for sat in satellites:
        if not deployment[sat]:
            assigned_service = random.choice(list(microservice_allocation.keys()))
            deployment[sat].append(assigned_service)
            microservice_allocation[assigned_service].append(sat)

    # 分配剩余副本，确保不超过约束条件
    for microservice in microservice_allocation:
        current_replicas = len(microservice_allocation[microservice])
        while current_replicas < max_replicas:
            valid_sats = [
                sat for sat in satellites
                if microservice not in deployment[sat] and len(deployment[sat]) < max_deployments
            ]
            if not valid_sats:
                break
            sat = random.choice(valid_sats)
            deployment[sat].append(microservice)
            microservice_allocation[microservice].append(sat)
            current_replicas += 1

    return deployment


def get_services_on_satellite(deployment, satellite):
    """
    查询指定卫星上的所有微服务集合 D(v_i)。

    参数:
        deployment (dict): 微服务部署策略，格式为 {vi: [m1, m2, ...], ...}。
        satellite: 指定的卫星 vi，格式为 (轨道编号, 卫星编号)。

    返回:
        list: 部署在该卫星上的所有微服务集合 D(v_i)。
    """
    return deployment.get(satellite, [])


def get_satellites_with_service(deployment, microservice):
    """
    查询指定微服务部署的所有卫星集合 V(m_i)。

    参数:
        deployment (dict): 微服务部署策略，格式为 {vi: [m1, m2, ...], ...}。
        microservice (str): 指定的微服务名称 m_i。

    返回:
        list: 部署该微服务的所有卫星集合 V(m_i)。
    """
    if not microservice:
        return []
    if isinstance(microservice, int):
        microservice = f'm{microservice}'
    satellites = []
    for satellite, services in deployment.items():
        if microservice in services:
            satellites.append(satellite)
    return satellites


def save_deployment_and_topology(deployment, constellation_graph, filepath):
    """
    将微服务部署结果和星座拓扑保存为文件。

    参数:
        deployment (dict): 微服务部署结果，格式为 {vi: [m1, m2, ...], ...}。
        constellation_graph (nx.Graph): LEO星座拓扑。
        filepath (str): 保存文件的路径。
    """
    data = {
        "deployment": {
            str(sat): services for sat, services in deployment.items()
        },
        "topology": json_graph.node_link_data(constellation_graph),
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    print(f"部署结果和星座拓扑已保存到 {filepath}")


def load_deployment_and_topology(filepath):
    """
    从文件中读取微服务部署结果和星座拓扑。

    参数:
        filepath (str): 文件路径。

    返回:
        tuple: 包括微服务部署结果和星座拓扑。
            - deployment (dict): 微服务部署结果，格式为 {vi: [m1, m2, ...], ...}。
            - constellation_graph (nx.Graph): LEO星座拓扑。
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    deployment = {
        eval(sat): services for sat, services in data["deployment"].items()
    }
    constellation_graph = json_graph.node_link_graph(data["topology"])

    return deployment, constellation_graph


def get_total_microservices(deployment):
    """
    从部署结果中计算微服务的总数。

    参数:
        deployment (dict): 微服务部署结果，格式为 {vi: [m1, m2, ...], ...}。

    返回:
        int: 微服务的总数（不重复计数）。
    """
    microservices = set()
    for services in deployment.values():
        microservices.update(services)
    return len(microservices)


def generate_microservice_dependency_graph(num_microservices, max_microservices, max_children, max_branches=1):
    """
    生成一个连通的随机微服务依赖图。

    参数:
        num_microservices (int): 参与的微服务数。
        max_microservices (int): 微服务总数。
        max_children (int): 一个微服务最多的子微服务数量。
        max_branches (int): 最多分支数。

    返回:
        tuple: 包括根节点和表示微服务依赖关系的有向图 (root, nx.DiGraph)
    """
    dependency_graph = nx.DiGraph()

    all_microservices = [f"m{i}" for i in range(1, max_microservices + 1)]
    microservices = random.sample(all_microservices, num_microservices)

    root = random.choice(microservices)
    dependency_graph.add_node(root)

    queue = [root]
    remaining_services = set(microservices)
    remaining_services.remove(root)
    num_branches = 0

    while queue and remaining_services:
        current_service = queue.pop(0)

        num_children = 1
        if random.random() < 0.5 and num_branches < max_branches:
            num_children = random.randint(1, min(max_children, len(remaining_services)))
            num_branches += 1

        children = random.sample(remaining_services, num_children)
        for child in children:
            dependency_graph.add_edge(current_service, child)
            queue.append(child)
            remaining_services.remove(child)

    return root, dependency_graph


def get_next_executable_services(dependency_graph, current_service):
    """
    给定一个微服务依赖图和当前正在执行的微服务，返回下一步可以执行的微服务列表。

    参数:
        dependency_graph (nx.DiGraph): 微服务依赖关系的有向图。
        current_service (str): 当前正在执行的微服务节点。

    返回:
        list: 下一步可以执行的微服务列表。
    """
    return list(dependency_graph.successors(current_service))


def save_dependency_graph_and_root(dependency_graph, root, filepath):
    """
    将依赖图和根顶点保存为 JSON 文件。

    参数:
        dependency_graph (nx.DiGraph): 微服务依赖关系图。
        root (str): 根顶点（root）。
        filepath (str): 保存文件的路径。
    """
    data = {
        "root": root,
        "dependency_graph": json_graph.node_link_data(dependency_graph)
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)
    print(f"依赖图和 root 已保存到 {filepath}")


def load_root_and_dependency_graph(filepath):
    """
    从 JSON 文件中读取依赖图和根顶点。

    参数:
        filepath (str): JSON 文件路径。

    返回:
        tuple: 根顶点和依赖图 (root, nx.DiGraph)。
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    root = data["root"]
    dependency_graph = json_graph.node_link_graph(data["dependency_graph"])

    return root, dependency_graph


def generate_augment_graph(flow: Microservice_flow, deployment, constellation_graph):
    """
    根据微服务流和星座模型生成增广图。

    参数:
        flow (Microservice_flow): 微服务流。
        constellation_graph (nx.Graph): LEO星座拓扑。

    返回:
        augmented_graph (nx.Graph): 卫星-微服务流融合增广子图
    """
    augmented_graph = nx.DiGraph()
    src = f'{flow.src}t'
    dst = f'{flow.dst}t'
    augmented_graph.add_node(src)  # src 作为 t 类虚拟节点
    augmented_graph.add_node(dst)  # dst 作为 t 类虚拟节点

    next_microservices = [flow.root]
    while next_microservices:
        current_microservice = next_microservices.pop()
        next_microservices.extend(flow.get_next_microservice(current_microservice))

        current_nodes = get_satellites_with_service(deployment, current_microservice)
        if current_microservice == flow.root:
            prev_nodes = [flow.src]
        else:
            prev_nodes = get_satellites_with_service(deployment, flow.get_prev_microservice(current_microservice))
        for node in current_nodes:
            v_node_c = f'{node}c'
            v_node_t = f'{node}t'
            augmented_graph.add_node(v_node_c)
            augmented_graph.add_node(v_node_t)
            augmented_graph.add_edge(v_node_c, v_node_t)
            for prev_node in prev_nodes:
                augmented_graph.add_edge(f'{prev_node}t', v_node_c)

        if not flow.get_next_microservice(current_microservice):
            for node in current_nodes:
                if node != flow.dst:
                    augmented_graph.add_edge(f'{node}t', dst)
    return augmented_graph


def draw_graph_with_weights(graph):
    """
    绘制带有节点权重和边权重的图。
        参数:
        graph (nx.Graph): 填充了权重的星座图。
    """
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=700, node_color="lightblue", font_size=10, font_weight="bold")
    node_labels = {node: f"{data['weight']}" for node, data in graph.nodes(data=True)}
    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_color="black", font_size=8, font_weight="bold")
    edge_labels = {(node1, node2): f"{data['weight']}" for node1, node2, data in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color="red", font_size=8)
    plt.show()


def get_flow_clusters(flows, rates, deployment):
    """
    RMPR Step1 微服务流分组
    根据每个微服务的依赖图，将请求流量划分为与其部署的卫星数量对应的分组。

    参数：
    - flows: 微服务请求流对象列表，每个流包含一个依赖图。
    - rates: 字典，键为流的标识符，值为请求流的到达率。
    - deployment: 部署依赖。

    返回：
    - clusters: 字典，键为微服务标识符，值为分组后的请求流 ID 列表。
    """
    all_ms = set()
    for flow in flows:
        all_ms.update(flow.dependency_graph.nodes)

    clusters = defaultdict(list)

    for ms in all_ms:
        # 获取微服务 ms 部署的服务器数量 k
        k = len(get_satellites_with_service(deployment, ms))

        # 获取属于该微服务的流及其到达率
        ms_flows = [flow for flow in flows if ms in flow.dependency_graph.nodes]
        ms_rates = {flow.id: rates[flow.id] for flow in ms_flows}

        # 使用 Karmarkar-Karp 分组
        cluster = karmarkar_karp_multiway_heap(flows, ms_rates, k)

        # 保存分组结果
        clusters[ms] = cluster

    return clusters


def karmarkar_karp_multiway_heap(flows, ms_rates, k):
    """
    使用扩展的 Karmarkar-Karp 算法，将请求流划分为 k 组，使每组的总和尽可能接近。

    参数：
    - ms_rates: 字典，键为流 ID，值为流的到达率。
    - k: 分组数量。

    返回：
    - groups: 分组后的流列表。
    """
    heap = [(0, i) for i in range(k)]  # 每个分组用 (当前和, 分组索引) 表示
    heapq.heapify(heap)

    groups = [[] for _ in range(k)]

    sorted_flows = sorted(ms_rates.items(), key=lambda x: x[1], reverse=True)

    for flow_id, rate in sorted_flows:
        current_sum, group_index = heapq.heappop(heap)

        groups[group_index].append(flows[flow_id - 1])
        current_sum += rate

        heapq.heappush(heap, (current_sum, group_index))

    return groups


def determine_routing_by_RMPR(all_flows, rates, constellation, deployment, all_satellites):
    """
    通过 RMPR 方法计算服务路由路径
    """
    clusters = get_flow_clusters(all_flows, rates, deployment)
    resources = [-1]
    resources.extend([constellation.nodes[id]['weight'] for id in range(1, len(constellation.nodes) + 1)])
    satellites_for_flows = defaultdict(lambda: defaultdict(int))
    for ms, cluster in clusters.items():
        satellites = get_satellites_with_service(deployment, ms)
        satellites.sort(key=lambda x: resources[x], reverse=True)
        for flows in cluster:
            satellite = satellites.pop(0)
            for flow in flows:
                satellites_for_flows[flow.id][ms] = satellite
                resources[satellite] -= rates[flow.id]

    routes = defaultdict(list)
    for flow in random.sample(all_flows, len(all_flows)):
        routes[flow.id] = [flow.src]
        current_ms = flow.root
        while current_ms != 'done':
            current_dst = satellites_for_flows[flow.id][current_ms]
            current_src = routes[flow.id].pop()
            # 使用 dijkstra 算法找到 constellation 中从 current_src 到 current_dst 的最短路，并将路径添加到 routes 中
            path = nx.shortest_path(constellation, source=current_src, target=current_dst,
                                    weight=lambda x, y, z: len(all_satellites[y - 1].isl) + 1)
            routes[flow.id].extend(path)
            current_ms = flow.get_next_microservice(current_ms)[0]
        current_dst = flow.dst
        current_src = routes[flow.id].pop()
        path = nx.shortest_path(constellation, source=current_src, target=current_dst)
        routes[flow.id].extend(path)

    return satellites_for_flows, routes


if __name__ == "__main__":
    # root, dependency_graph = generate_microservice_dependency_graph(8, 60, 1, 1)
    # save_dependency_graph_and_root(dependency_graph, root, "../graph/dependency_60_4.json")
    # # constellation = create_leo_constellation(6, 12)
    # # constellation = fill_weights(constellation)
    # # deployment = deploy_microservices(60, constellation, 4, 6)
    # # save_deployment_and_topology(deployment, constellation, "../graph/6_12_4_6_deployment.json")
    # # deployment, constellation = load_deployment_and_topology("../graph/6_12_4_6_deployment.json")
    # # for i in range(60):
    # #     print(get_satellites_with_service(deployment, i+1))
    deployment, constellation = load_deployment_and_topology("../graph/6_12_4_6_deployment.json")
    constellation = fill_weights(constellation, False, None)
    all_flows = []
    for i in range(4):
        root, dependency = load_root_and_dependency_graph(f"../graph/dependency_60_{i + 1}.json")
        flow = Microservice_flow(1 + i, dependency, root, 1 + i * 5, 72 - i * 3)
        all_flows.append(flow)
    rates = {1: 100, 2: 130, 3: 100, 4: 200}
    satellites, routes = determine_routing_by_RMPR(all_flows, rates, constellation, deployment)
    print(satellites[1], routes[1])
