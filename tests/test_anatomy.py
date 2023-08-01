#!/usr/bin/env python
# -*- coding: utf-8 -*-
import networkx as nx

from brainscore_vision.metrics.anatomy import EdgeRatioMetric


class TestEdgeRatio:
    def test_2nodes_equal(self):
        graph = nx.DiGraph()
        graph.add_edge('A', 'B')
        assert 1 == EdgeRatioMetric()(graph, graph)

    def test_2nodes_different(self):
        graph1 = nx.DiGraph()
        graph1.add_edge('A', 'B')
        graph2 = nx.DiGraph()
        graph2.add_edge('B', 'A')
        assert 0 == EdgeRatioMetric()(graph1, graph2)

    def test_3nodes_equal(self):
        graph = nx.DiGraph()
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        assert 1 == EdgeRatioMetric()(graph, graph)

    def test_3nodes_1edge_missing(self):
        graph1 = nx.DiGraph()
        graph1.add_edge('A', 'B')
        graph1.add_edge('B', 'C')
        graph2 = nx.DiGraph()
        graph2.add_edge('A', 'B')
        graph2.add_edge('C', 'B')
        assert 0.5 == EdgeRatioMetric()(graph1, graph2)
