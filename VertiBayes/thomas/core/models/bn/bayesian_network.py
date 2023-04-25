"""BayesianNetwork"""
from __future__ import annotations

from functools import reduce

import numpy as np
import pandas as pd

import networkx as nx
import networkx.algorithms.moral

import json

from thomas.core import options
from ...factors.factor import Factor
from ...factors.cpt import CPT
from ...factors.jpt import JPT
from .node import Node
from .discrete_node import DiscreteNetworkNode

from ...learn.structure.greedy import greedy_network_learning, compute_cpts_network

from ..base import ProbabilisticModel
from ..bag import Bag
from ..jpt import JPTModel
from ..junction_tree import JunctionTree

from ... import error

import logging
log = logging.getLogger(__name__)


class BayesianNetwork(ProbabilisticModel):
    """A Bayesian Network (BN) consists of Nodes and directed Edges.

    A BN is essentially a Directed Acyclic Graph (DAG) where each Node
    represents a Random Variable (RV) and is associated with a conditional
    probability table (CPT). A CPT can only have a *single* conditioned
    variable; zero or more *conditioning*  variables are allowed. Conditioning
    variables are represented as the Node's parents.

    BNs can be used for inference. To do this efficiently, the BN first
    constructs a JunctionTree (or JoinTree).

    Because of the relation between the probability distribution (expressed as a
    set of CPTs) and the graph structure, it is possible to instantiate a BN
    from a list of CPTs.
    """

    def __init__(self, name, nodes, edges):
        """Instantiate a new BayesianNetwork.

        Args:
            name (str): Name of the Bayesian Network.
            nodes (list): List of Nodes.
            edges (list): List of Edges.
        """
        self.name = name

        # dictionaries, indexed by nodes' random variables
        self.nodes = {}
        self.evidence = {}

        # Process the nodes and edges.
        if nodes:
            self.add_nodes(nodes)

            if edges:
                self.add_edges(edges)

        self.elimination_order = None

        # Cached junction tree
        self._jt = None

        # Widget ...
        self.__widget = None

    def __getitem__(self, RV):
        """x[name] <==> x.nodes[name]"""
        return self.nodes[RV]

    def __getattr__(self, attr: str):
        """..."""
        try:
            return self.nodes[attr]
        except KeyError:
            msg = f"'BayesianNetwork' has no attribute '{attr}'"
            raise AttributeError(msg)

    def __repr__(self):
        """x.__repr__() <==> repr(x)"""
        s = f"<BayesianNetwork name='{self.name}'>\n"
        for RV in self.nodes:
            node = self.nodes[RV]
            s += f"  <Node RV='{RV}' description='{node.description}' states={self.nodes[RV].states} />\n"

        s += '</BayesianNetwork>'

        return s

    # --- properties ---
    @property
    def edges(self):
        edges = []
        for n in self.nodes.values():
            for c in n._children:
                edges.append((n.RV, c.RV))

        return edges

    @property
    def junction_tree(self):
        """Return the junction tree for this network."""
        if self._jt is None:
            self._jt = JunctionTree(self)

        return self._jt

    # Aliases ...
    jt = junction_tree
    junctiontree = junction_tree
    jointree = junction_tree

    @property
    def vars(self):
        """Return the variables in this BN (i.e. the scope) as a *set*."""
        return set(self.nodes.keys())

    @property
    def scope(self):
        """Return the variables in this BN (i.e. the scope) as a *list*."""
        return list(self.nodes.keys())

    @property
    def states(self):
        """Return a dict of states, indexed by random variable."""
        return {RV: self.nodes[RV].states for RV in self.nodes}

    @property
    def nodes_without_parents(self):
        """Return the nodes without parents."""
        return [n for n in self.nodes.values() if not n.has_parents()]

    @property
    def nodes_with_parents(self):
        """Return the of nodes that children of (an)other node(s)."""
        return [n for n in self.nodes.values() if n.has_parents()]

    def setWidget(self, widget):
        """Associate this BayesianNetwork with a BayesianNetworkWidget."""
        self.__widget = widget

    def complete_case(self, case, include_weights=True):
        """Complete a single case.

        Args:
            case (pandas.Series): case to complete.
            include_weights (bool): return multiple rows, including
                probabilities (weights) for each possible combination of missing
                values. If False, only the most likely case is returned.

        Return:
            pandas.Series or pandas.DataFrame
        """
        missing = list(case[case.isna()].index)
        evidence = {e: case[e] for e in case.index if e not in missing}

        if not missing:
            if include_weights:
                df = pd.DataFrame([case])
                df['weight'] = 1
                return df.reset_index(drop=True)

            return case

        try:
            jpt = self.compute_posterior(
                qd=missing,
                qv={},
                ed=[],
                ev=evidence
            )

        except error.InvalidStateError as e:
            print('WARNING - Could not complete case:', e)
            return

        # Create a dataframe by repeating the evicence multiple times: once for
        # each possible (combination of) value(s) of the missing variable(s).
        # The brackets enclosing the evidence transpose the matrix, because the
        # index is set, the row is broadcast.
        idx = jpt.get_pandas_index()
        imputed = pd.DataFrame([case[evidence]], index=idx)

        # The combinations of missing variables are in the index. Reset the
        # index to make them part of the dataframe.
        imputed = imputed.reset_index()

        # Add the computed probabilities as weights
        imputed.loc[:, 'weight'] = jpt.flat

        if include_weights:
            order = list(case.index) + ['weight']
            return imputed[order]

        return imputed.loc[imputed.weight.idxmax(), list(case.index)]

    def estimate_emperical(self, data):
        """Estimate the emperical distribution from data."""
        complete_case = lambda x: self.complete_case(x[1], include_weights=True)
        expanded = list(map(complete_case, data.iterrows()))

        summed = pd.concat(expanded).groupby(self.scope).sum()

        # Factor(0, ...) creates a Factor of all zeroes containing all possible
        # combinations of variable states. Adding to this Factor ensures the
        # JPT is complete
        return JPT(Factor(0, self.states) + (summed / summed.sum())['weight'])

    # --- graph manipulation ---
    def add_nodes(self, nodes: Node):
        """Add a Node to the network."""
        for node in nodes:
            self.nodes[node.RV] = node

        self._jt = None

    def add_edges(self, edges):
        """Recreate the edges using the nodes' CPTs."""
        for (parent_RV, child_RV) in edges:
            self.nodes[parent_RV].add_child(self.nodes[child_RV])

        self._jt = None

    def moralize_graph(self):
        """Return the moral graph for the DAG.

        A moral graph adds an edge between nodes that share a common child and
        then makes edges undirected.

        https://en.wikipedia.org/wiki/Moral_graph:
            > The name stems from the fact that, in a moral graph, two nodes
            > that have a common child are required to be married by sharing
            > an edge.
        """
        G = self.as_networkx()
        G_moral = nx.algorithms.moral.moral_graph(G)
        return list(G_moral.edges)

    # -- parameter estimation
    # FIXME: move this to thomas.core.learn
    def EM_learning(self, data, max_iterations=1, notify=True):
        """Perform parameter learning.
        Sources:
            * https://www.cse.ust.hk/bnbook/pdf/l07.h.pdf
            * https://www.youtube.com/watch?v=NDoHheP2ww4
        """
        # Ensure the data only contains states that are allowed.
        data = data.copy()

        for RV in self.scope:
            node = self.nodes[RV]
            data = data[data[RV].isin(node.states)]

        # Children (i.e. nodes with parents) identify the families in the BN.
        nodes_with_parents = self.nodes_with_parents
        nodes_without_parents = self.nodes_without_parents

        # Create a dataset with unique rows (& counts) ...
        overlapping_cols = list(set(data.columns).intersection(self.vars))
        counts = data.fillna('NaN')
        counts = counts.groupby(overlapping_cols, observed=True).size()
        counts.name = 'count'
        counts = pd.DataFrame(counts)
        counts = counts.reset_index()
        counts = counts[counts['count'] > 0]
        counts = counts.reset_index(drop=True)
        counts = counts.replace('NaN', np.nan)

        iterator = range(max_iterations)

        # If tqdm is available *and* we're not in quiet mode
        if not options.get('quiet', False):
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator)
            except Exception as e:
                print('Could not instantiate tqdm')
                print(e)

        for k in iterator:
            # print(f'--- iteration {k} ---')

            # dict of joint distributions, indexed by family index
            joints = {}

            # Iterate over the data: set a row as evidence and compute the
            # JPT for each family.
            # for idx, row in counts.iterrows():
            for row_idx, row in counts.iterrows():

                N = row.pop('count')
                evidence = row.dropna().to_dict()

                self.reset_evidence()
                self.junction_tree.set_evidence_hard(**evidence)

                # print(f'------ row: {row_idx} ------\n{row}\n')
                # print(f'------ row: {row_idx} ------')
                # print(f'Setting evidence: {evidence}')
                # print(f'Weight: {N}')
                # print()

                for node in nodes_with_parents:
                    # print(f'======== Processing node for {node.cpt.display_name} ========')
                    jt_node = self.junction_tree.get_node_for_family(node.vars)

                    # print(f'Using JT node with cluster {jt_node.label}')
                    jpt = jt_node.joint * N

                    # print()
                    # print(f'JPT:\n{jpt}\n')

                    joints[node] = joints[node] + jpt if node in joints else jpt

                # print()

            # Update CPTs for nodes *with* parents
            for node in nodes_with_parents:
                jpt = joints[node].project(node.vars)

                node.cpt = CPT(
                    data=jpt / jpt.project(node.conditioning),
                    conditioned=node.conditioned
                )

            # Update JPTs for nodes *without* parents
            for node in nodes_without_parents:
                for jpt in joints.values():
                    # Find a JPT that contains this node's RV
                    if node.vars.issubset(jpt.vars):
                        node.cpt = CPT(
                            data=jpt.project(node.vars).normalize(),
                            conditioned=node.conditioned
                        )

                        # Break from the *inner* for loop
                        break

            # Since the JT is linked to the BN's nodes' CPTs, we'll need to
            # invalidate the cache to recompute the probabilities. Setting
            # hard=True is needed to enforce recomputing joints.
            self.reset_evidence()
            self.jt.invalidate_caches(hard=True)

            # Update the widget after each iteration
            if self.__widget and notify:
                self.__widget.update()

    # FIXME: move this to thomas.core.learn
    def ML_estimation(self, df):
        """Perform Maximum Likelihood estimation of the BN parameters.

        *** Note: this will overwrite any CPTs already set. ***

        Args:
            df (pandas.Dataframe): dataset that contains columns with names
                corresponding to the variables in this BN's scope.
        """
        # Ensure the data only contains states that are allowed.
        data = df.copy()

        for RV in self.scope:
            node = self.nodes[RV]
            data = data[data[RV].isin(node.states)]

        # The empirical distribution may not contain all combinations of
        # variable states; `from_data` fixes that by setting all missing entries
        # to 0.
        jpt = JPTModel(JPT.from_data(df, cols=self.scope))

        for name, node in self.nodes.items():
            # print(name, node, node.cpt)
            cpt = jpt.compute_dist(node.conditioned, node.conditioning)
            node.cpt = cpt

        # CPTs have updated, so cache is no longer valid. JunctionTree will have
        # to be recreated.
        self._jt = None

        # Update the widget
        if self.__widget:
            self.__widget.update()

    # FIXME: move this to thomas.core.learn
    def likelihood(self, df, per_case=False):
        """Return the likelihood of the current network parameters given data.

        Warning: naive/slow implementation :-)
        """
        # Create a subset of the dataframe that only contains relevant columns
        vars = self.vars.intersection(df.columns)
        subset = df[vars]

        # Compute the posterior probability for each row of data
        func = lambda x: self.compute_posterior([], x.dropna().to_dict(), [], {})

        result = subset.apply(func, axis=1)

        if per_case:
            return result

        # Multiply the results with each other and return
        return reduce(lambda x, y: x*y, result)

    # --- inference ---
    # FIXME: this method probably belongs somewhere else
    def get_node_elimination_order(self):
        """Return a naïve elimination ordering, based on nodes' degree."""
        if self.elimination_order is None:
            G = nx.Graph()
            G.add_edges_from(self.moralize_graph())

            degrees = list(G.degree)
            degrees.sort(key=lambda x: x[1])

            self.elimination_order = [d[0] for d in degrees]

        return self.elimination_order

    def compute_joint_with_jt(self, RVs):
        """Compute the joint distribution over multiple variables.

        Args:
            RVs (list or set): Set of random variables to compute joint over.

        Returns:
            Factor:  marginal distribution over the RVs in Q.
        """
        Q = set(RVs) if isinstance(RVs, list) else RVs

        jt = JunctionTree(self)
        jt.ensure_cluster(Q)
        node = jt.get_node_for_set(Q)

        joint = node.project(RVs)

        if isinstance(RVs, list):
            joint = joint.reorder_scope(RVs)

        return CPT(joint, conditioned=RVs)

    def compute_marginals(self, qd=None, ev=None):
        """Compute the marginals of the query variables given the evidence.

        Note that calling this method will reset any evidence previously set
        using `set_evidence()`!

        Args:
            qd (list): Random variables to query
            ev (dict): dict of states, indexed by RV to use as
                evidence.

        Returns:
            dict of marginals, indexed by RV
        """
        if ev is None:
            ev = {}

        # Reset the tree and apply evidence
        self.junction_tree.reset_evidence()
        self.junction_tree.set_evidence_hard(**ev)

        return self.junction_tree.get_marginals(qd)

    def compute_posterior(self, qd, qv, ed, ev, use_VE=False):
        """Compute the (posterior) probability of query given evidence.

        The query P(I,G=g1|D,L=l0) would imply:
            qd = ['I']
            qv = {'G': 'g1'}
            ed = ['D']
            ev = {'L': 'l0'}

        Args:
            qd (list): query distributions: RVs to query
            qv (dict): query values: RV-values to extract
            ed (list): evidence distributions: coniditioning RVs to include
            ev (dict): evidence values: values to set as evidence.

        Returns:
            CPT or scalar (iff `qv` is specified)
        """
        # Evidence we can just set on the JT, but we'll need to compute the
        # joint over the other variables to answer the query.
        required_RVs = set(qd + list(qv.keys()) + ed)
        node = self.junction_tree.get_node_for_set(required_RVs)

        if node is None and use_VE is False:
            log.info('Cannot answer this query with the current junction tree.')
            use_VE = True

        if use_VE:
            log.debug('Using VE')
            return self.as_bag().compute_posterior(qd, qv, ed, ev)

        # Compute the answer to the query using the junction tree.
        log.debug(f'Found a node in the JT that contains {required_RVs}: {node.cluster}')
        self.junction_tree.reset_evidence()
        self.junction_tree.set_evidence_hard(**ev)

        # Process evidence distributions
        log.debug(f'  Projecting onto {required_RVs} without normalization')
        result = node.joint.project(set(required_RVs))
        result = result.normalize()

        if ed:
            result = result / result.project(set(ed))

        # RVs that are part of the query will need to be set as column names.
        query_vars = list(qv.keys()) + qd

        # If query values were specified we can extract them from the factor.
        if qv:
            result = result.get(**qv)

        if isinstance(result, Factor):
            return CPT(result, conditioned=query_vars)

        return result

    def reset_evidence(self, RVs=None, notify=True):
        """Reset evidence."""
        if isinstance(RVs, str):
            RVs = [RVs]

        self.junction_tree.reset_evidence(RVs)

        if RVs:
            for RV in RVs:
                del self.evidence[RV]
        else:
            self.evidence = {}

        if self.__widget and notify:
            self.__widget.update()

    def set_evidence_likelihood(self, RV, **kwargs):
        """Set likelihood evidence on a variable."""
        self.junction_tree.set_evidence_likelihood(RV, **kwargs)
        self.evidence[RV] = kwargs

    def set_evidence_hard(self, RV, state, notify=True):
        """Set hard evidence on a variable.

        This corresponds to setting the likelihood of the provided state to 1
        and the likelihood of all other states to 0.
        """
        log.debug(f"Setting evidence for '{RV}': {state}")
        self.junction_tree.set_evidence_hard(**{RV: state})
        self.evidence[RV] = state

        if self.__widget and notify:
            self.__widget.update()

    def get_marginals(self, RVs=None):
        """Return the probabilities for a set off/all RVs given set evidence."""
        return self.junction_tree.get_marginals(RVs)

    # --- constructors ---
    @classmethod
    def from_CPTs(cls, name, CPTs):
        """Create a Bayesian Network from a list of CPTs."""
        nodes = {}
        edges = []

        for cpt in CPTs:
            RV = cpt.conditioned[0]
            node = DiscreteNetworkNode(RV)

            if cpt.description:
                node.name = cpt.description

            for parent_RV in cpt.conditioning:
                edges.append((parent_RV, RV))

            nodes[RV] = node

        bn = BayesianNetwork(name, nodes.values(), edges)

        for cpt in CPTs:
            RV = cpt.conditioned[0]
            bn[RV].cpt = cpt

        return bn

    @classmethod
    def from_data(cls, name: str, df: pd.DataFrame,
                  degree_network: int = 2) -> BayesianNetwork:
        """Learn a BN structure and parameters from data.

        Uses greedy structure learning.
        """
        network = greedy_network_learning(df, degree_network)
        cpts = compute_cpts_network(df, network)
        bn = BayesianNetwork.from_CPTs(name, cpts.values())
        return bn

    # --- visualization ---
    def draw(self):
        """Draw the BN using networkx & matplotlib."""
        # nx.draw(self.as_networkx(), with_labels=True)

        nx_tree = self.as_networkx()
        pos = nx.spring_layout(nx_tree)

        nx.draw(
            nx_tree,
            pos,
            edge_color='black',
            font_color='white',
            width=1,
            linewidths=1,
            node_size=1500,
            node_color='purple',
            alpha=1.0,
            with_labels=True,
        )

    def print_cpts(self):
        """Print all CPTs to stdout."""
        for name, node in self.nodes.items():
            print('-' * 60)
            print(node.cpt)
            print()

    # --- (de)serialization and conversion ---
    def as_networkx(self):
        G = nx.DiGraph()
        G.add_edges_from(self.edges)
        return G

    def as_bag(self):
        return Bag(
            name=self.name,
            factors=[n.cpt for n in self.nodes.values()]
        )

    def as_dict(self):
        """Return a dict representation of this Bayesian Network."""
        return {
            'type': 'BayesianNetwork',
            'name': self.name,
            'nodes': [n.as_dict() for n in self.nodes.values()],
            'edges': self.edges,
        }

    def as_json(self, pretty=False):
        """Return a JSON representation (str) of this Bayesian Network."""
        indent = 4 if pretty else None
        return json.dumps(self.as_dict(), indent=indent)

    def save(self, filename):
        with open(filename, 'w') as fp:
            fp.write(self.as_json(True))

    def copy(self):
        """Return a copy of this BN."""
        return BayesianNetwork.from_dict(self.as_dict())

    @classmethod
    def from_dict(self, d):
        """Return a Bayesian Network initialized by its dict representation."""
        name = d.get('name')
        nodes = [Node.from_dict(n) for n in d['nodes']]
        edges = d.get('edges')
        bn = BayesianNetwork(name, nodes, edges)
        return bn

    @classmethod
    def from_json(cls, json_str):
        """Return a Bayesian Network initialized by its JSON representation."""
        d = json.loads(json_str)
        return cls.from_dict(d)

    @classmethod
    def open(cls, filename):
        with open(filename) as fp:
            data = fp.read()
            return cls.from_json(data)
