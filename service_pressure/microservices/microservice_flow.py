import networkx as nx


class Microservice_flow:
    def __init__(self, id, dependency_graph, root, src, dst):
        self.id = id
        self.dependency_graph = dependency_graph
        self.root = root
        self.src = src
        self.dst = dst

    def get_next_microservice(self, current_microservice=None):
        if not current_microservice:
            return [self.root]
        if isinstance(current_microservice, int):
            current_microservice = f'm{current_microservice}'
        next_microservices = list(self.dependency_graph.successors(current_microservice))
        if not next_microservices:
            next_microservices = ['done']
        return next_microservices

    def get_prev_microservice(self, current_microservice):
        if isinstance(current_microservice, int):
            current_microservice = f'm{current_microservice}'
        return list(self.dependency_graph.predecessors(current_microservice))[0]
