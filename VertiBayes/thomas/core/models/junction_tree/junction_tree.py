# -*- coding: utf-8 -*-
"""JunctionTree"""
from typing import List, Tuple, Set

import logging

import networkx as nx
from networkx.algorithms.shortest_paths.generic import shortest_path

from functools import reduce

from ... import error
from ...factors.factor import mul, Factor

from .tree_edge import TreeEdge
from .tree_node import TreeNode

log = logging.getLogger(__name__)


ClusterType = List[Set[str]]


class JunctionTree(object):
    """JunctionTree for a BayesianNetwork.

    The tree consists of TreeNodes and TreeEdges.
    """

    def __init__(self, bn, clusters: ClusterType = None):
        """Initialize a new JunctionTree.

        Args:
            bn (BayesianNetwork): associated BN.
        """
        self._bn = bn

        self.nodes = {}       # TreeNode, indexed by cluster.label
        self.edges = []
        self.indicators = {}  # evidence indicators; indexed by RV
        self._RVs = {}        # TreeNode, indexed by RV and

        # Create the structure.
        if clusters is None:
            self.clusters = self._get_elimination_clusters()
        else:
            self.clusters = clusters

        self._create_structure()

        # Assign factors & evidence indicators.
        self._assign_factors(bn)

    def _get_elimination_clusters(self):
        """Compute the clusters for the elimination tree.

        The cluster of node i in an elimination tree, is defined as the
        union of its variables and the variables that appear on its separators.

        A family in the Bayesian network is defined as a node and its parents.

        It is guaranteed that every family in the BN appears in one of the
        clusters (as stated in the definition of the junction tree).
        """
        bn = self._bn

        # Get the full set of clusters.
        edges = bn.moralize_graph()
        order = bn.get_node_elimination_order()
        clusters = self._get_elimination_clusters_rec(edges, order)

        # Merge clusters that are contained in other clusters by iterating over
        # the full set and replacing the smaller with the larger clusters.
        clusters = list(clusters)

        # We should merge later clusters into earlier clusters.
        # The reversion is undone just before the function returns.
        clusters.reverse()

        should_continue = len(clusters) > 1
        while should_continue:
            should_continue = False

            for idx_i in range(len(clusters)):
                modified = False
                C_i = clusters[idx_i]

                for idx_j in range(idx_i + 1, len(clusters)):
                    C_j = clusters[idx_j]

                    if C_i.issubset(C_j):
                        clusters[idx_i] = C_j
                        clusters.pop(idx_j)

                        modified = True
                        should_continue = len(clusters) > 1

                        # Break from the inner for-loop
                        break

                if modified:
                    # Break from the outer for-loop
                    break

        # Undo the earlier reversion.
        clusters.reverse()
        return clusters

    def _get_elimination_clusters_rec(self, edges: List[Tuple[str, str]],
                                      order: List[str]):
        """Recursively compute the clusters for the elimination tree.

        Args:
            edges (list): List of edges that make up the (remaining) Graph.
            order (list): Elimination order.

        Returns:
            list of clusters (i.e. sets of RVs)
        """
        # Create a copy to make sure we're not modifying the method argument.
        order = list(order)

        if not len(order):
            return []

        # Reconstruct the graph
        G = nx.Graph()
        G.add_nodes_from(order)
        G.add_edges_from(edges)

        node = order.pop(0)

        # Make sure the neighbors from `node` are connected by adding fill-in
        # edges.
        neighbors = list(G.neighbors(node))

        if len(neighbors) > 1:
            for outer_idx in range(len(neighbors)):
                n1 = neighbors[outer_idx]

                for inner_idx in range(outer_idx + 1, len(neighbors)):
                    n2 = neighbors[inner_idx]
                    G.add_edge(n1, n2)

        G.remove_node(node)
        cluster = set([node, ] + neighbors)

        return [cluster, ] + self._get_elimination_clusters_rec(G.edges, order)

    def _create_structure(self):
        """Create the tree's structure (i.e. add edges) using the clusters."""

        # Each cluster is added to a TreeNode by reference, meaning that any
        # changes to `node.cluster` are also reflected in `self.clusters`.
        for c in self.clusters:
            # Create a TreeNode and add it to self.nodes
            self.add_node(c)

        # The nodes contain a variable `cluster` that corresponds to one of the
        # tree's clusters. The order is equal to the order of `self.clusters`.
        nodes = list(self.nodes.values())

        # Iterate over the tree's nodes to find a neighbor for each node.
        # We need to cast the iterator to a list to use reversed()
        for idx, node_i in reversed(list(enumerate(nodes))):
            C_i = node_i.cluster

            # `remaining_nodes` will hold clusters C_i+1 , C_i+2, .... C_n
            remaining_nodes = nodes[idx + 1:]

            if remaining_nodes:
                remaining_clusters = [n.cluster for n in remaining_nodes]

                # We'll compute union over the remaining clusters and determine
                # the intersection with the current cluster.
                intersection = C_i.intersection(set.union(*remaining_clusters))

                # According to the running intersection property, there should
                # be a node/cluster that contains the above intersection.
                options = []

                for node_j in remaining_nodes:
                    C_j = node_j.cluster

                    if intersection.issubset(C_j):
                        options.append(node_j)

                        # self.add_edge(node_i, node_j)
                        # break from the for loop

                if len(options) > 1:
                    complexities = []

                    for n in options:
                        CPTs = [self._bn[RV].cpt for RV in n.cluster]
                        reduced = reduce(mul, CPTs)
                        complexities.append(len(reduced))

                    # sizes = [len(n.joint) for n in options]
                    # print(sizes)
                    # print(options)
                    option_idx = complexities.index(min(complexities))
                    self.add_edge(node_i, options[option_idx])
                else:

                    self.add_edge(node_i, options[0])

    def _assign_factors(self, bn):
        """Assign the BNs factors (nodes) to one of the clusters."""
        bn = self._bn

        # Iterate over all nodes in the BN to assign each BN node/CPT to the
        # first TreeNode that contains the BN node's RV. Also, assign an evidence
        # indicator for that variable to that JT node.
        for RV, bn_node in bn.nodes.items():
            # Iterate over the JT nodes/clusters
            for jt_node in self.nodes.values():
                # node.vars returns all variables in the node's factor
                if bn_node.vars.issubset(jt_node.cluster):
                    jt_node.add_bn_node(bn_node)
                    self.set_node_for_RV(RV, jt_node)

                    states = {RV: bn_node.states}
                    indicator = Factor(1, states=states)
                    self.add_indicator(indicator, jt_node)
                    break

        # Iterate over the JT nodes/clusters to make sure each cluster has
        # the correct factors assigned.
        for jt_node in self.nodes.values():
            for missing in (jt_node.cluster - jt_node.vars):
                bn_node = bn.nodes[missing]
                states = {bn_node.RV: bn_node.states}
                trivial = Factor(1, states=states)
                jt_node.add_factor(trivial)

    @property
    def width(self):
        """Return the width of the JT."""
        return max([len(c) for c in self.clusters]) -1

    def ensure_cluster(self, cluster):
        """Ensure cluster is contained in one of the nodes."""
        Q = set(cluster) if isinstance(cluster, list) else cluster

        if self.get_node_for_set(Q):
            return

        # Find the (first) node that has the maximum overlap with Q
        overlap = [len(Q.intersection(c)) for c in self.clusters]
        idx = overlap.index(max(overlap))
        cluster = self.clusters[idx]
        node = self.get_node_for_set(cluster)

        # Determine which variables are missing in the cluster
        missing = Q - node.cluster

        # Convert the JT to a NetworkX graph, so we can use it's implementation
        # of Dijkstra's shortest path later.
        G = self.as_networkx()

        for var in missing:
            # Find a path to the nearest node that contains `var`.
            # There may be multiple nodes that contain `var`, so first list all
            # targets.
            targets = [n for n in self.nodes.values() if var in n.cluster]
            paths = []

            for target in targets:
                paths.append(shortest_path(G, node, target))

            distances = [len(p) for p in paths]
            idx = distances.index(min(distances))
            path = paths[idx]

            states = self._bn.nodes[var].states

            for tree_node in path:
                # path includes the target node, which already has `var`
                # in scope
                if var not in tree_node.cluster:
                    f = Factor(1, states={var: states})
                    tree_node.add_factor(f)

    def get_node_for_RV(self, RV):
        """A[x] <==> A.__getitem__(x)"""
        return self._RVs[RV]

    def set_node_for_RV(self, RV, node):
        """A[x] = y <==> A.__setitem__(x, y)"""
        self._RVs[RV] = node

    def get_node_for_set(self, RVs):
        """Return the node that contains RVs or None.

        :param (set) RVs: set of RV names (strings)
        :return: TreeNode
        """
        for node in self.nodes.values():
            if RVs.issubset(node.cluster):
                return node

        return None

    # Alias
    get_node_for_family = get_node_for_set

    def get_marginals(self, RVs=None):
        """Return the probabilities for a set off/all RVs given set evidence."""
        if RVs is None:
            RVs = self._RVs

        return {RV: self.get_node_for_RV(RV).project(RV) for RV in RVs}

    def add_node(self, cluster):
        """Add a node to the junction tree."""
        node = TreeNode(cluster)
        self.nodes[node.label] = node
        return node

    def add_edge(self, node1, node2):
        """Add an edge between two nodes in the junction tree."""
        assert isinstance(node1, TreeNode), "node1 should be a TreeNode!"
        assert isinstance(node2, TreeNode), "node2 should be a TreeNode!"

        self.edges.append(TreeEdge(node1, node2))

    def add_indicator(self, factor, node):
        """Add an indicator for a random variable to a node."""
        assert isinstance(node, TreeNode), "node should be a TreeNode!"

        RV = list(factor.states.keys()).pop()
        self.indicators[RV] = factor
        node.indicators.append(factor)

    def reset_evidence(self, RVs=None):
        """Reset evidence."""
        if RVs is None:
            RVs = self.indicators

        for RV in RVs:
            indicator = self.indicators[RV]
            indicator.values[:] = 1.0

        self.invalidate_caches()

    def set_evidence_likelihood(self, RV, **kwargs):
        """Set likelihood evidence on a variable."""
        indicator = self.indicators[RV]

        # FIXME: it's not pretty to access Factor._data like this!
        data = indicator._data

        for state, value in kwargs.items():
            data[state] = value

        self.invalidate_caches()

    def set_evidence_hard(self, **kwargs):
        """Set hard evidence on a variable.

        This corresponds to setting the likelihood of the provided state to 1
        and the likelihood of all other states to 0.

        Kwargs:
            evidence (dict): dict with states, indexed by RV: {RV: state}
                             e.g. use as set_evidence_hard(G='g1')
        """
        for RV, state in kwargs.items():
            indicator = self.indicators[RV]

            # if state not in indicator.index.get_level_values(RV):
            if state not in indicator.states[RV]:
                raise error.InvalidStateError(RV, state, indicator)

            indicator.set(1, **{RV: state}, inplace=True)
            indicator.set_complement(0, **{RV: state}, inplace=True)

        self.invalidate_caches()

    def invalidate_caches(self, hard=False):
        """Invalidate the nodes' caches."""
        for n in self.nodes.values():
            n.invalidate_cache(hard)

    def as_networkx(self):
        """Return the JunctionTree as a networkx.Graph() instance."""
        G = nx.Graph()

        for e in self.edges:
            G.add_edge(e._left, e._right, label=','.join(e.separator))

        return G

    def draw(self, ax=None):
        """Draw the JunctionTree using networkx & matplotlib."""
        nx_tree = self.as_networkx()
        pos = nx.spring_layout(nx_tree)

        nx.draw(
            nx_tree,
            pos,
            ax,
            edge_color='black',
            width=1,
            linewidths=1,
            node_size=1500,
            node_color='pink',
            alpha=1.0,
            labels={node: node.label for node in nx_tree.nodes}
        )

        labels = {key: value['label'] for key, value in nx_tree.edges.items()}
        nx.draw_networkx_edge_labels(
            nx_tree,
            pos,
            edge_labels=labels,
            font_color='red'
        )
