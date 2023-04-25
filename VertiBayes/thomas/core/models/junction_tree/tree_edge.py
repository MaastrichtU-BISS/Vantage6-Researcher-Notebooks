"""TreeEdge: edge between TreeNodes in a JunctionTree."""

from .tree_node import TreeNode


class TreeEdge(object):
    """Edge in an elimination/junction tree."""

    def __init__(self, node1: TreeNode, node2: TreeNode):
        """Create a new (undirected) TreeEdge.

        Args:
            node1 (TreeNode): node 1
            node1 (TreeNode): node 2
        """
        # Left/right are arbitrary.
        self._left = node1
        self._right = node2
        self._separator = None

        node1.add_neighbor(self)
        node2.add_neighbor(self)

    def __repr__(self):
        """repr(x) <==> x.__repr__()"""
        return f'Edge: ({repr(self._left)} - {repr(self._right)})'

    @property
    def separator(self):
        """Return/compute the separator on this edge."""
        if self._separator is None:
            self.recompute_separator()

        return self._separator

    def get_neighbor(self, node):
        """Return the neighbor for `node`."""
        if node == self._left:
            return self._right

        if node == self._right:
            return self._left

        raise Exception('Supplied node is not connected to this edge!?')

    def recompute_separator(self):
        """(re)compute the separator for this Edge."""
        left_downstream = self._left.get_all_downstream_nodes(self)
        right_downstream = self._right.get_all_downstream_nodes(self)

        left_cluster = set.union(*[n.cluster for n in left_downstream])
        right_cluster = set.union(*[n.cluster for n in right_downstream])

        self._separator = set.intersection(left_cluster, right_cluster)
