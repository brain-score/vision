import networkx as nx

from mkgu.metrics import Metric, Similarity

ventral_stream = nx.DiGraph()  # derived from Felleman & van Essen
ventral_stream.add_edge('input', 'V1')
ventral_stream.add_edge('V1', 'V2')
ventral_stream.add_edge('V1', 'V4')
ventral_stream.add_edge('V2', 'V1')
ventral_stream.add_edge('V2', 'V4')
ventral_stream.add_edge('V4', 'V1')
ventral_stream.add_edge('V4', 'V2')
ventral_stream.add_edge('V4', 'pIT')
ventral_stream.add_edge('V4', 'cIT')
ventral_stream.add_edge('V4', 'aIT')
ventral_stream.add_edge('pIT', 'V4')
ventral_stream.add_edge('pIT', 'cIT')
ventral_stream.add_edge('pIT', 'aIT')
ventral_stream.add_edge('cIT', 'V4')
ventral_stream.add_edge('cIT', 'pIT')
ventral_stream.add_edge('cIT', 'aIT')
ventral_stream.add_edge('aIT', 'V4')
ventral_stream.add_edge('aIT', 'pIT')
ventral_stream.add_edge('aIT', 'cIT')


class EdgeRatioMetric(Metric):
    def __init__(self):
        super(EdgeRatioMetric, self).__init__(similarity=EdgeRatio())


class EdgeRatio(Similarity):
    def __call__(self, source_graph, target_graph):
        unmatched = [edge for edge in target_graph.edges if edge not in source_graph.edges]
        return 1 - len(unmatched) / len(target_graph.edges)
