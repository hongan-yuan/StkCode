import random
from collections import defaultdict
from copy import deepcopy

from microservices.packets import Packet
from utils.utils import get_satellites_with_service, get_services_on_satellite


class Satellite_node:
    def __init__(self, id, constellation, deployment, env):
        self.env = env
        self.id = id  # 对应的真实卫星节点
        self.constellation = constellation  # 所在的星座
        self.deployment = deployment
        self.backlog_queue_c = defaultdict(lambda: defaultdict(list))  # c类虚拟节点
        self.backlog_queue_t = defaultdict(lambda: defaultdict(list))  # t类虚拟节点
        self.neighbors = list(self.constellation.successors(self.id))
        self.cal_throughput = constellation.nodes[self.id]['weight']
        self.trans_throughput = sum([constellation.edges[(self.id, neighbor)]['weight'] for neighbor in self.neighbors])
        self.isl = []  # 模拟星间链路

        self.total_queue_length = 0

        # rmpr
        self.queue_c = []
        self.queue_t = []

    def update_throughput(self):
        self.cal_throughput = self.constellation.nodes[self.id]['weight']
        self.trans_throughput = sum([self.constellation.edges[(self.id, neighbor)]['weight'] for neighbor in self.neighbors])

    def receive_packets(self):
        completed_flows = defaultdict(list)
        while self.isl:
            item = self.isl.pop()
            if self.id == item.flow.dst and item.next_microservice == 'done':
                # 到达dst节点
                completed_flows[item.flow.id].append(item.life_time)
                continue
            elif self.id in item.dest_satellites:
                # 作为c类虚拟节点接管这个数据包
                self.receive_packet_c(item.flow.id, item.next_microservice, item)
                self.total_queue_length += 1
            else:
                # 作为t类虚拟节点接管这个数据包
                self.receive_packet_t(item.flow.id, item.next_microservice, item)
                self.total_queue_length += 1
        self.env.update_completed_flows(completed_flows)

    def receive_packets_rmpr(self):
        completed_flows = defaultdict(list)
        while self.isl:
            item = self.isl.pop()
            satellites, route = item.info
            current_dst = route.pop(0)
            if self.id != current_dst:
                # 转发错误
                del item
                print('error')
                continue
            if self.id == item.flow.dst and item.next_microservice == 'done':
                # 到达dst节点
                completed_flows[item.flow.id].append(item.life_time)
                continue
            elif self.id == satellites[item.next_microservice]:
                # 作为c类虚拟节点接管这个数据包
                item.info = (satellites, route)
                self.queue_c.append(item)
            else:
                # 作为t类虚拟节点接管这个数据包
                item.info = (satellites, route)
                self.queue_t.append(item)
        self.env.update_completed_flows(completed_flows)

    def receive_packets_lcra(self):
        completed_flows = defaultdict(list)
        while self.isl:
            item = self.isl.pop()
            if self.id == item.flow.dst and item.next_microservice == 'done':
                # 到达dst节点
                completed_flows[item.flow.id].append(item.life_time)
                continue
            elif self.id in item.dest_satellites:
                # 作为c类虚拟节点接管这个数据包
                self.queue_c.append(item)
            else:
                # 作为t类虚拟节点接管这个数据包
                self.queue_t.append(item)
        self.env.update_completed_flows(completed_flows)

    def receive_packet_c(self, flow_id, next_microservice, packet):
        self.backlog_queue_c[flow_id][next_microservice].append(packet)

    def pop_packet_c(self, flow_id, next_microservice):
        if self.backlog_queue_c[flow_id][next_microservice]:
            packet = self.backlog_queue_c[flow_id][next_microservice].pop()
            return packet
        return None

    def queue_length(self):
        queue_length = defaultdict(int)
        all_flows = set(self.backlog_queue_c.keys()).union(set(self.backlog_queue_t.keys()))
        for flow_id in all_flows:
            len = 0
            all_microservices = set(self.backlog_queue_c[flow_id].keys()).union(
                set(self.backlog_queue_t[flow_id].keys()))
            for microservice in all_microservices:
                len += self.query_queue_c(flow_id, microservice)
                len += self.query_queue_t(flow_id, microservice)
            queue_length[flow_id] = len
        return queue_length.items()

    def queue_length_total(self):
        queue_length = 0
        all_flows = set(self.backlog_queue_c.keys()).union(set(self.backlog_queue_t.keys()))
        for flow_id in all_flows:
            all_microservices = set(self.backlog_queue_c[flow_id].keys()).union(
                set(self.backlog_queue_t[flow_id].keys()))
            for microservice in all_microservices:
                queue_length += self.query_queue_c(flow_id, microservice)
                queue_length += self.query_queue_t(flow_id, microservice)
        return queue_length

    def query_queue_c(self, flow_id, next_microservice):
        return len(self.backlog_queue_c[flow_id].get(next_microservice, []))

    def receive_packet_t(self, flow_id, next_microservice, packet):
        self.backlog_queue_t[flow_id][next_microservice].append(packet)

    def pop_packet_t(self, flow_id, next_microservice):
        if self.backlog_queue_t[flow_id][next_microservice]:
            packet = self.backlog_queue_t[flow_id][next_microservice].pop()
            return packet
        return None

    def query_queue_t(self, flow_id, next_microservice):
        return len(self.backlog_queue_t[flow_id].get(next_microservice, []))

    def step_c(self):
        # 模拟c类虚拟节点的运行
        gradients = []
        all_flows = set(self.backlog_queue_c.keys())
        for flow_id in all_flows:
            all_microservices = set(self.backlog_queue_c[flow_id].keys())
            for microservice in all_microservices:
                flow = self.env.get_flow_by_id(flow_id)
                next_microservices = flow.get_next_microservice(microservice)
                u_count = self.query_queue_c(flow_id, microservice)
                v_count = 0
                for next_ms in next_microservices:
                    v_count += self.query_queue_t(flow_id, next_ms)
                gradient = u_count - v_count
                gradients.append((flow_id, microservice, gradient))

        gradients.sort(key=lambda x: x[2], reverse=True)

        total = self.cal_throughput
        # TODO 当前c类虚拟节点资源分配方式：按积压梯度降序分配，用尽全部计算吞吐量
        allocation = {}
        for flow_id, microservice, gradient in gradients:
            needs = self.query_queue_c(flow_id, microservice)
            if gradient <= 0:
                break
            if total < needs:
                allocation[(flow_id, microservice)] = total
                break
            else:
                allocation[(flow_id, microservice)] = needs
                total -= needs

        for (flow_id, microservice), num in allocation.items():
            flow = self.env.get_flow_by_id(flow_id)
            next_microservices = flow.get_next_microservice(microservice)
            for _ in range(num):
                packet = self.pop_packet_c(flow_id, microservice)
                for next_ms in next_microservices:
                    dst = get_satellites_with_service(self.deployment, next_ms)
                    packet_copy = deepcopy(packet)
                    packet_copy.dest_satellites = dst
                    packet_copy.next_microservice = next_ms
                    # self.buffer.append(packet_copy)
                    if self.id in dst:
                        self.receive_packet_c(flow_id, next_ms, packet_copy)
                    else:
                        self.receive_packet_t(flow_id, next_ms, packet_copy)
                del packet

    def step_t(self):
        # 模拟t类虚拟节点运行
        neighbors = [self.env.get_satellite_by_id(i) for i in list(self.constellation.successors(self.id))]
        all_flows = set(self.backlog_queue_t.keys())

        # TODO 当前t类节点资源分配方式：先选择边，然后按梯度降序分配在该边上传输的微服务流
        # for neighbor in neighbors:
        #     total = self.constellation.edges[(self.id, neighbor.id)]['weight']
        #     gradients = []
        #     for flow_id in all_flows:
        #         all_microservices = set(self.backlog_queue_t[flow_id].keys())
        #         for microservice in all_microservices:
        #             dest = get_satellites_with_service(self.deployment, microservice)
        #             if neighbor.id in dest:
        #                 # t->c
        #                 u_count = self.query_queue_t(flow_id, microservice)
        #                 v_count = neighbor.query_queue_c(flow_id, microservice)
        #                 gradient = u_count - v_count
        #                 gradients.append((flow_id, microservice, gradient))
        #             else:
        #                 # t->t
        #                 u_count = self.query_queue_t(flow_id, microservice)
        #                 v_count = neighbor.query_queue_t(flow_id, microservice)
        #                 gradient = u_count - v_count
        #                 gradients.append((flow_id, microservice, gradient))
        # TODO 当前t类节点资源分配方式：加入可行传播区域
        for neighbor in random.sample(neighbors, len(neighbors)):
            total = self.constellation.edges[(self.id, neighbor.id)]['weight']
            gradients = []
            for flow_id in all_flows:
                flow = self.env.get_flow_by_id(flow_id)
                all_microservices = set(self.backlog_queue_t[flow_id].keys())
                for microservice in all_microservices:
                    dest = get_satellites_with_service(self.deployment, microservice)
                    if microservice == 'none':
                        dest = [flow.dst]
                    u_dist = self.env.get_shortest_hop_distance(self.id, dest)
                    v_dist = self.env.get_shortest_hop_distance(neighbor.id, dest)
                    if neighbor.id in dest:
                        # t->c
                        u_count = self.query_queue_t(flow_id, microservice)
                        v_count = neighbor.query_queue_c(flow_id, microservice)
                        gradient = u_count - v_count
                        # gradient = u_dist * u_count - v_dist * v_count
                        gradients.append((flow_id, microservice, gradient))
                    else:
                        # t->t
                        u_count = self.query_queue_t(flow_id, microservice)
                        v_count = neighbor.query_queue_t(flow_id, microservice)
                        gradient = u_count - v_count
                        # gradient = u_dist * u_count - v_dist * v_count
                        # if microservice == 'done' and neighbor.id == flow.dst:
                        #     gradient = float('inf')
                        if u_dist < v_dist:
                            gradient = -1
                        # if self.env.get_shortest_isl_distance(self.id, dest) < self.env.get_shortest_isl_distance(
                        #         neighbor.id, dest):
                        #     gradient = -1
                        gradients.append((flow_id, microservice, gradient))

            gradients.sort(key=lambda x: x[2], reverse=True)
            for flow_id, microservice, gradient in gradients:
                needs = self.query_queue_t(flow_id, microservice)
                if gradient <= 0:
                    break
                elif total < needs:
                    for _ in range(total):
                        packet = self.pop_packet_t(flow_id, microservice)
                        self.total_queue_length -= 1
                        neighbor.isl.append(packet)
                    break
                else:
                    for _ in range(needs):
                        packet = self.pop_packet_t(flow_id, microservice)
                        self.total_queue_length -= 1
                        neighbor.isl.append(packet)
                    total -= needs

    def step_c_proportional(self):
        # 模拟c类虚拟节点的运行
        gradients = []
        all_flows = set(self.backlog_queue_c.keys())

        # 计算所有流和微服务的梯度
        for flow_id in all_flows:
            all_microservices = set(self.backlog_queue_c[flow_id].keys())
            for microservice in all_microservices:
                flow = self.env.get_flow_by_id(flow_id)
                next_microservices = flow.get_next_microservice(microservice)
                u_count = self.query_queue_c(flow_id, microservice)  # 当前微服务的积压
                v_count = 0  # 下游微服务的积压总和
                for next_ms in next_microservices:
                    v_count += self.query_queue_t(flow_id, next_ms)
                gradient = u_count - v_count  # 计算梯度
                if gradient > 0:  # 只考虑正梯度
                    gradients.append((flow_id, microservice, gradient))

        # 计算总梯度
        total_gradient = sum(g[2] for g in gradients)

        # 如果总梯度为0，直接返回，无需分配资源
        if total_gradient == 0:
            return

        # 计算每个流分配的资源量
        allocation = {}
        total = self.cal_throughput  # 虚拟节点的计算能力（资源总量）
        for flow_id, microservice, gradient in gradients:
            # 按梯度比例分配资源
            allocated = (gradient / total_gradient) * total
            needs = self.query_queue_c(flow_id, microservice)  # 当前微服务所需资源
            allocation[(flow_id, microservice)] = min(allocated, needs)  # 分配不能超过需求

        # TODO 当前c类节点资源分配方式：按比例分配
        for (flow_id, microservice), num in allocation.items():
            num = int(num)  # 确保资源分配为整数
            flow = self.env.get_flow_by_id(flow_id)
            next_microservices = flow.get_next_microservice(microservice)
            for _ in range(num):
                packet = self.pop_packet_c(flow_id, microservice)  # 从当前队列弹出数据包
                for next_ms in next_microservices:
                    dst = get_satellites_with_service(self.deployment, next_ms)
                    packet_copy = deepcopy(packet)  # 深拷贝数据包
                    packet_copy.dest_satellites = dst
                    packet_copy.next_microservice = next_ms
                    # self.buffer.append(packet_copy)
                    if self.id in dst:
                        self.receive_packet_c(flow_id, next_ms, packet_copy)
                    else:
                        self.receive_packet_t(flow_id, next_ms, packet_copy)
                del packet

    def step_t_proportional(self):
        # 模拟t类虚拟节点运行
        neighbors = [self.env.get_satellite_by_id(i) for i in list(self.constellation.successors(self.id))]
        all_flows = set(self.backlog_queue_t.keys())

        # TODO 当前t类节点资源分配方式：按比例分配，加入可行传播区域
        for neighbor in random.sample(neighbors, len(neighbors)):
            total = self.constellation.edges[(self.id, neighbor.id)]['weight']
            gradients = []

            for flow_id in all_flows:
                flow = self.env.get_flow_by_id(flow_id)
                all_microservices = set(self.backlog_queue_t[flow_id].keys())
                for microservice in all_microservices:
                    dest = get_satellites_with_service(self.deployment, microservice)
                    if neighbor.id in dest:
                        # t->c
                        u_count = self.query_queue_t(flow_id, microservice)
                        v_count = neighbor.query_queue_c(flow_id, microservice)
                        gradient = u_count - v_count
                        gradients.append((flow_id, microservice, gradient))
                    else:
                        # t->t
                        u_count = self.query_queue_t(flow_id, microservice)
                        v_count = neighbor.query_queue_t(flow_id, microservice)
                        gradient = u_count - v_count
                        if microservice == 'done' and neighbor.id == flow.dst:
                            gradient = 0x3f3f3f3f
                        if self.env.get_shortest_hop_distance(self.id, dest) < self.env.get_shortest_hop_distance(
                                neighbor.id, dest):
                            gradient = -1
                        gradients.append((flow_id, microservice, gradient))

            # 过滤正梯度并计算总和
            positive_gradients = [(flow_id, microservice, gradient) for flow_id, microservice, gradient in gradients if
                                  gradient > 0]
            total_positive_gradient = sum(gradient for _, _, gradient in positive_gradients)

            if total_positive_gradient == 0:  # 没有正梯度可分配
                continue

            # 按比例分配资源
            for flow_id, microservice, gradient in positive_gradients:
                proportion = gradient / total_positive_gradient
                allocated_resource = int(total * proportion)

                needs = self.query_queue_t(flow_id, microservice)
                allocated_packets = min(allocated_resource, needs)

                for _ in range(allocated_packets):
                    packet = self.pop_packet_t(flow_id, microservice)
                    neighbor.isl.append(packet)

                total -= allocated_packets
                if total <= 0:
                    break

    def step_c_rmpr(self):
        for _ in range(self.cal_throughput):
            if not self.queue_c:
                break
            packet = self.queue_c.pop()
            next_microservices = packet.flow.get_next_microservice(packet.next_microservice)
            for next_ms in next_microservices:
                dst = get_satellites_with_service(self.deployment, next_ms)
                packet_copy = deepcopy(packet)
                packet_copy.dest_satellites = dst
                packet_copy.next_microservice = next_ms
                self.queue_t.append(packet_copy)
            del packet

    def step_t_rmpr(self):
        neighbors = [self.env.get_satellite_by_id(i) for i in list(self.constellation.successors(self.id))]
        isl_constraints = defaultdict(int)
        for neighbor in neighbors:
            isl_constraints[neighbor.id] = self.constellation.edges[(self.id, neighbor.id)]['weight']
        temp_queue = []
        if len(self.queue_t) > 10000:
            self.queue_t = self.queue_t[9000:]
        self.queue_t = random.sample(self.queue_t, len(self.queue_t))
        while self.queue_t:
            packet = self.queue_t.pop()
            _, route = packet.info
            next_dst = route[0]
            if isl_constraints[next_dst] <= 0:
                temp_queue.append(packet)
                continue
            else:
                isl_constraints[next_dst] -= 1
                self.env.get_satellite_by_id(next_dst).isl.append(packet)
        self.queue_t.extend(temp_queue)

    def step_c_queueflower(self):
        # 模拟c类虚拟节点的运行
        gradients = []
        all_flows = set(self.backlog_queue_c.keys())

        # 当前c类节点资源分配方式：按 queueflower 方式计算 virtual queue
        for flow_id in all_flows:
            all_microservices = set(self.backlog_queue_c[flow_id].keys())
            for microservice in all_microservices:
                flow = self.env.get_flow_by_id(flow_id)
                next_microservices = flow.get_next_microservice(microservice)
                u_count = sum([packet.life_time for packet in self.backlog_queue_c[flow_id].get(microservice, [])])
                v_count = 0  # 下游微服务的积压总和
                for next_ms in next_microservices:
                    v_count += sum([packet.life_time for packet in self.backlog_queue_t[flow_id].get(next_ms, [])])
                gradient = u_count - v_count  # 计算梯度
                if gradient > 0:  # 只考虑正梯度
                    gradients.append((flow_id, microservice, gradient))

        # 计算总梯度
        total_gradient = sum(g[2] for g in gradients)

        # 如果总梯度为0，直接返回，无需分配资源
        if total_gradient == 0:
            return

        # 计算每个流分配的资源量
        allocation = {}
        total = self.cal_throughput  # 虚拟节点的计算能力（资源总量）
        for flow_id, microservice, gradient in gradients:
            # 按梯度比例分配资源
            allocated = (gradient / total_gradient) * total
            needs = self.query_queue_c(flow_id, microservice)  # 当前微服务所需资源
            allocation[(flow_id, microservice)] = min(allocated, needs)  # 分配不能超过需求

        for (flow_id, microservice), num in allocation.items():
            num = int(num)  # 确保资源分配为整数
            flow = self.env.get_flow_by_id(flow_id)
            next_microservices = flow.get_next_microservice(microservice)
            for _ in range(num):
                packet = self.pop_packet_c(flow_id, microservice)  # 从当前队列弹出数据包
                for next_ms in next_microservices:
                    dst = get_satellites_with_service(self.deployment, next_ms)
                    packet_copy = deepcopy(packet)
                    packet_copy.dest_satellites = dst
                    packet_copy.next_microservice = next_ms
                    if self.id in dst:
                        self.receive_packet_c(flow_id, next_ms, packet_copy)
                    else:
                        self.receive_packet_t(flow_id, next_ms, packet_copy)
                del packet

    def step_t_queueflower(self):
        # 模拟t类虚拟节点运行
        neighbors = [self.env.get_satellite_by_id(i) for i in list(self.constellation.successors(self.id))]
        all_flows = set(self.backlog_queue_t.keys())

        # 当前t类节点资源分配方式：按 queueflower 方式计算 virtual queue
        for neighbor in random.sample(neighbors, len(neighbors)):
            total = self.constellation.edges[(self.id, neighbor.id)]['weight']
            gradients = []

            for flow_id in all_flows:
                flow = self.env.get_flow_by_id(flow_id)
                all_microservices = set(self.backlog_queue_t[flow_id].keys())
                for microservice in all_microservices:
                    dest = get_satellites_with_service(self.deployment, microservice)
                    if neighbor.id in dest:
                        # t->c
                        u_count = sum([packet.life_time for packet in self.backlog_queue_t[flow_id].get(microservice, [])])
                        v_count = sum([packet.life_time for packet in neighbor.backlog_queue_c[flow_id].get(microservice, [])])
                        gradient = u_count - v_count
                        gradients.append((flow_id, microservice, gradient))
                    else:
                        # t->t
                        u_count = sum([packet.life_time for packet in self.backlog_queue_t[flow_id].get(microservice, [])])
                        v_count = sum([packet.life_time for packet in neighbor.backlog_queue_t[flow_id].get(microservice, [])])
                        gradient = u_count - v_count
                        # if microservice == 'done' and neighbor.id == flow.dst:
                        #     gradient = 0x3f3f3f3f
                        # if self.env.get_shortest_hop_distance(self.id, dest) < self.env.get_shortest_hop_distance(
                        #         neighbor.id, dest):
                        #     gradient = -1
                        gradients.append((flow_id, microservice, gradient))

            # 过滤正梯度并计算总和
            positive_gradients = [(flow_id, microservice, gradient) for flow_id, microservice, gradient in gradients if
                                  gradient > 0]
            total_positive_gradient = sum(gradient for _, _, gradient in positive_gradients)

            if total_positive_gradient == 0:  # 没有正梯度可分配
                continue

            # 按比例分配资源
            for flow_id, microservice, gradient in positive_gradients:
                proportion = gradient / total_positive_gradient
                allocated_resource = int(total * proportion)

                needs = self.query_queue_t(flow_id, microservice)
                allocated_packets = min(allocated_resource, needs)

                for _ in range(allocated_packets):
                    packet = self.pop_packet_t(flow_id, microservice)
                    neighbor.isl.append(packet)

                total -= allocated_packets
                if total <= 0:
                    break

    def step_c_lcra(self):
        for _ in range(self.cal_throughput):
            if not self.queue_c:
                break
            packet = self.queue_c.pop()
            next_microservices = packet.flow.get_next_microservice(packet.next_microservice)
            for next_ms in next_microservices:
                dst = get_satellites_with_service(self.deployment, next_ms)
                if next_ms == 'done':
                    dst = [packet.flow.dst]
                packet_copy = deepcopy(packet)
                packet_copy.dest_satellites = dst
                packet_copy.next_microservice = next_ms
                self.queue_t.append(packet_copy)
            del packet

    def step_t_lcra(self):
        neighbors = [self.env.get_satellite_by_id(i) for i in list(self.constellation.successors(self.id))]
        next_hop = defaultdict(lambda: defaultdict(int))
        isl_constraints = defaultdict(int)
        for neighbor in neighbors:
            isl_constraints[neighbor.id] = self.constellation.edges[(self.id, neighbor.id)]['weight']
        temp_queue = []
        # if len(self.queue_t) > 10000:
        #     self.queue_t = self.queue_t[9000:]
        # self.queue_t = random.sample(self.queue_t, len(self.queue_t))
        while self.queue_t:
            packet = self.queue_t.pop()
            if next_hop[packet.flow.id][packet.next_microservice] == 0:
                neighbors.sort(key=lambda x: (len(x.queue_t) / sum([data['weight'] for _, _, data in self.constellation.out_edges(x.id, data=True)])) if len(x.queue_t) > len(self.queue_t) else 0 + self.env.get_shortest_hop_distance(x.id, packet.dest_satellites))
                next_hop[packet.flow.id][packet.next_microservice] = neighbors[0].id
            next_dst = next_hop[packet.flow.id][packet.next_microservice]
            if isl_constraints[next_dst] <= 0:
                temp_queue.append(packet)
                continue
            else:
                isl_constraints[next_dst] -= 1
                self.env.get_satellite_by_id(next_dst).isl.append(packet)
        self.queue_t.extend(temp_queue)

    def increment_life_time(self):
        for nested_dict in self.backlog_queue_c.values():
            for item_list in nested_dict.values():
                for item in item_list:
                    item.life_time += 1

        for nested_dict in self.backlog_queue_t.values():
            for item_list in nested_dict.values():
                for item in item_list:
                    item.life_time += 1

    def increment_life_time_rmpr(self):
        for item in self.queue_c:
            item.life_time += 1
        for item in self.queue_t:
            item.life_time += 1

    def increment_life_time_lcra(self): self.increment_life_time_rmpr()
