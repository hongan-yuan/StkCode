from copy import deepcopy


class Packet:
    def __init__(self, flow, dest_satellites, next_microservice, life_time=0, info=None):
        self.flow = flow  # 所属的微服务流
        self.dest_satellites = dest_satellites  # 当前的目的地（一组c类虚拟节点或dst节点）
        self.next_microservice = next_microservice  # 下一个执行的微服务
        self.life_time = life_time  # 生存时间
        self.info = deepcopy(info)
