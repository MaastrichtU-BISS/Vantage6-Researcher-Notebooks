"""TreeNode: Node in a JunctionTree."""
from functools import reduce
import logging

from ...factors.factor import Factor, mul

log = logging.getLogger(__name__)


class TreeNode(object):
    """Node in an elimination/junction tree."""

    def __init__(self, cluster):
        """Create a new node.

        Args:
            cluster (set): set of RV names (strings)
        """
        self.cluster = cluster
        self.indicators = []

        # dict of bayesiannetwork.Node instances, indexed by RV
        self._bn_nodes = {}
        self.__factors = []

        self._edges = []  # list: TreeEdge

        # cache
        self._cache = None
        self._factors_multiplied = None
        self._factor_index_cache = None

        # The cache is indexed by upstream node.
        self.invalidate_cache()  # sets self._cache = {}

    def __repr__(self):
        """x.__repr__() <==> repr(x)"""
        return f"TreeNode({self.cluster})"

    @property
    def label(self):
        """Return the Node's label."""
        return ','.join(self.cluster)

    @property
    def factors(self):
        """All factors, including indicators."""
        CPTs = [node.cpt for node in self._bn_nodes.values()]
        return CPTs + self.__factors + self.indicators

    @property
    def vars(self):
        """Scope of this TreeNode as a set."""
        v = [f.vars for f in self.factors]
        if v:
            return set.union(*v)
        return set()

    @property
    def joint(self):
        """Return the joint distribution over this TreeNode's cluster."""
        return self.pull().normalize()

    @property
    def factors_multiplied(self):
        """Compute the joint of the TreeNode's factors."""
        if self._factors_multiplied is None:
            factors = [n.cpt for n in self._bn_nodes.values()]
            self._factors_multiplied = reduce(mul, factors, Factor(1, {}))

            if isinstance(self._factors_multiplied, Factor):
                self._factors_multiplied.normalize(inplace=True)

        return self._factors_multiplied

    def add_neighbor(self, edge):
        if edge not in self._edges:
            self._edges.append(edge)

    def add_factor(self, factor):
        """Add a (trivial) factor to this TreeNode."""
        self.__factors.append(factor)

        for var in factor.vars:
            self.cluster.add(var)

        for edge in self._edges:
            edge.recompute_separator()

        self._factors_multiplied = None
        self._factor_index_cache = None

    def add_bn_node(self, node):
        """Add a Bayesian Network node."""
        self._bn_nodes[node.RV] = node

    def invalidate_cache(self, hard=False):
        """Invalidate the message cache."""
        self._cache = {}

        if hard:
            self._factors_multiplied = None

    def get_downstream_edges(self, upstream=None):
        return [e for e in self._edges if e is not upstream]

    def get_all_downstream_nodes(self, upstream):
        edges = self.get_downstream_edges(upstream)

        downstream = []

        for edge in edges:
            node = edge.get_neighbor(self)
            downstream.extend(node.get_all_downstream_nodes(edge))

        return [self, ] + downstream

    def pull(self, upstream=None):
        """Trigger pulling of messages towards this node.

        This entails:
         - calling pull() on each of the downstream edges
         - multiplying the results into this factor
         - if an upstream edge is specified, the result is projected onto the
           upstream edge's separator.

        :return: factor.Factor
        """
        log.debug(f"TreeNode '{self.label}' is pulling ...")
        # if 'age' in self.cluster:
        #     log.debug('--> This one has age! <--')
        #     log.debug(self.factors_multiplied.project('age'))

        downstream_edges = self.get_downstream_edges(upstream)
        result = self.factors_multiplied
        result = reduce(mul, [result, *self.indicators])

        if 'age' in result:
            log.debug('--> This node has age! <--')
            log.debug(result.project('age'))

        if downstream_edges:
            downstream_results = []

            for e in downstream_edges:
                if e not in self._cache:
                    n = e.get_neighbor(self)
                    self._cache[e] = n.pull(e)

                downstream_results.append(self._cache[e])

            # result = reduce(mul, downstream_results + [result])
            result = reduce(mul, downstream_results + [result])

            if 'age' in result:
                log.debug(f'{self.label} - Reduced result has age!')
                log.debug(result.project('age'))

        if upstream:
            log.debug('Projecting onto upstream separator ...')
            return result.project(upstream.separator)

        if 'age' in result:
            log.debug(f'{self.label} - Final result has age!')
            log.debug(result.project('age'))
            log.debug(f'{self.label} - Normalized ...')
            log.debug(result.project('age').normalize())

        return result

    def project(self, RV, normalize=True):
        """Trigger a pull and project the result onto RV.

        RV should be contained in the Node's cluster.
        Cave: note that TreeNode.project() has a different default behaviour
              compared to Factor.project(): it normalizes the result by default!

        Args:
            RV (str or set): ...
            normalize (bool): normalize the projection?

        Returns:
            ...
        """
        result = self.pull().project(RV)

        if normalize:
            return result.normalize()

        return result
