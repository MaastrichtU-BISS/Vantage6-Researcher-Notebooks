"""Functions for learning the structure of a bayesian network from an input dataset"""
import numpy as np

from collections import namedtuple
from itertools import combinations
from sklearn.metrics import mutual_info_score

from ...factors import Factor
from ...factors import CPT


def greedy_network_learning(df, degree_network=2):
    """Learns the network using a greedy approach based on the mutual
    information between variables

    Args:
        df (pandas.Dataframe): dataset that contains columns with names
            corresponding to the variables in this BN's scope.
        degree_network (int): maximum number of parents each node can
            have.
    """

    # init network
    network = []
    nodes = set(df.columns)
    nodes_selected = set()

    # define structure of NodeParentPair candidates
    NodeParentPair = namedtuple('NodeParentPair', ['node', 'parents'])

    # select random node as starting point
    root = np.random.choice(tuple(nodes))
    network.append(NodeParentPair(node=root, parents=None))
    nodes_selected.add(root)

    # select each remaining node iteratively that have the highest
    # mutual information with nodes that are already in the network
    for i in range(len(nodes_selected), len(nodes)):
        nodes_remaining = nodes - nodes_selected
        n_parents = min(degree_network, len(nodes_selected))

        node_parent_pairs = [
            NodeParentPair(n, tuple(p)) for n in nodes_remaining
            for p in combinations(nodes_selected, n_parents)
        ]

        # compute the mutual information for each note_parent pair candidate
        scores = _compute_scores(df, node_parent_pairs)

        # add best scoring candidate to the network
        sampled_pair = node_parent_pairs[np.argmax(scores)]
        nodes_selected.add(sampled_pair.node)
        network.append(sampled_pair)
    return network


def compute_cpts_network(df, network):
    """Computes the conditional probability distribution of each node
    in the Bayesian network
        Args:
        df (pandas.Dataframe): dataset that contains columns with names
            corresponding to the variables in this BN's scope.
        network (list): list of ordered NodeParentPairs
    """
    P = dict()
    for idx, pair in enumerate(network):
        if pair.parents is None:
            cpt = CPT.from_factor(Factor.from_data(df, cols=[pair.node])).normalize()
            # cpt = CPT(marginal_distribution, conditioned=[pair.node]).normalize()
        else:
            # todo: there should be a from_data at CPT
            cpt = CPT.from_factor(Factor.from_data(df, cols=[*pair.parents, pair.node])).normalize()
            # cpt = CPT(joint_distribution, conditioned=[pair.node]).normalize()

        # add conditional distribution to collection
        P[pair.node] = cpt
    return P


def _compute_scores(df, node_parent_pairs):
    """Computes mutual information for all NodeParentPair candidates"""
    scores = np.empty(len(node_parent_pairs))
    for idx, pair in enumerate(node_parent_pairs):
        scores[idx] = _compute_mutual_information(df, pair)
    return scores


def _compute_mutual_information(df, pair):
    node_values = df[pair.node].values
    if len(pair.parents) == 1:
        parent_values = df[pair.parents[0]].values
    else:
        # combine multiple parent columns into one string column
        parent_values = df.loc[:, pair.parents].astype(str).apply(lambda x: '-'.join(x.values), axis=1).values
    return mutual_info_score(node_values, parent_values)
