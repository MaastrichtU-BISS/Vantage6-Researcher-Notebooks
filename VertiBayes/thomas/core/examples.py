# -*- coding: utf-8 -*-
"""Example Bayesian networks.

Includes examples from ...
- Koller & Friedman's "Probabilistic Graphical Models"
- Adnan Darwiche's "Modeling and Reasoning with Bayesian Networks"
"""
import thomas.core
import thomas.core.reader.oobn

from .factors.factor import Factor
from .factors.cpt import CPT
from .factors.jpt import JPT

from .models.bn import BayesianNetwork
from .models.bn import DiscreteNetworkNode
from .models.jpt import JPTModel


def subset(full_dict, keys):
    """Return a subset of a dict."""
    return {k: full_dict[k] for k in keys}


def get_student_CPTs():
    """Return the CPTs for the Student Bayesian Network."""
    P = dict()
    states = {
        'I': ['i0', 'i1'],
        'S': ['s0', 's1'],
        'D': ['d0', 'd1'],
        'G': ['g1', 'g2', 'g3'],
        'L': ['l0', 'l1'],
    }

    P['I'] = CPT(
        [0.7, 0.3],
        states=subset(states, ['I']),
        description='Intelligence'
    )

    P['S'] = CPT(
        [0.95, 0.05,
         0.20, 0.80],
        states=subset(states, ['I', 'S']),
        description='SAT Score'
    )

    P['D'] = CPT(
        [0.6, 0.4],
        states=subset(states, ['D']),
        description='Difficulty'
    )

    P['G'] = CPT(
        [0.30, 0.40, 0.30,
         0.05, 0.25, 0.70,
         0.90, 0.08, 0.02,
         0.50, 0.30, 0.20],
        states=subset(states, ['I', 'D', 'G']),
        description='Grade'
    )

    P['L'] = CPT(
        [0.10, 0.90,
         0.40, 0.60,
         0.99, 0.01],
        states=subset(states, ['G', 'L']),
        description='Letter'
    )

    return P


def get_student_network_from_CPTs():
    """Return the Student Bayesian Network."""
    P = get_student_CPTs()
    return BayesianNetwork.from_CPTs('Student', P.values())


def get_student_network():
    """Return the Student Bayesian Network."""
    filename = thomas.core.get_pkg_filename('student.json')
    return BayesianNetwork.open(filename)


def get_sprinkler_factors():
    """Return the factors for the Sprinkler Bayesian Network.

    Data copied from "Modeling and Reasoning with Bayesian Networks"
    (page 127-128) by Adnan Darwiche (2009).

    Examples:
    >>> fA, fB_A, fC_A, fD_BC, fE_C = examples.get_sprinkler_factors()
    """
    states = {
        'A': ['a1', 'a0'],
        'B': ['b1', 'b0'],
        'C': ['c1', 'c0'],
        'D': ['d1', 'd0'],
        'E': ['e1', 'e0'],
    }

    # P(A)
    fA = Factor(
        [0.6, 0.4],
        subset(states, ['A'])
    )

    # P(B|A)
    fB_A = Factor(
        [0.2, 0.8, 0.75, 0.25],
        subset(states, ['A', 'B'])
    )

    # P(C|A)
    fC_A = Factor(
        [0.8, 0.2, 0.1, 0.9],
        subset(states, ['A', 'C'])
    )

    # Define a factor that holds the *conditional* distribution P(D|BC)
    fD_BC = Factor(
        [0.95, 0.05, 0.9, 0.1, 0.8, 0.2, 0.0, 1.0],
        subset(states, ['B', 'C', 'D'])
    )

    # P(E|C)
    fE_C = Factor(
        [0.7, 0.3, 0.0, 1.0],
        subset(states, ['C', 'E'])
    )

    return [fA, fB_A, fC_A, fD_BC, fE_C]


def get_sprinkler_jpt():
    """Return the JPT for the Sprinkler network.

    Data copied from "Modeling and Reasoning with Bayesian Networks"
    (page 127-128) by Adnan Darwiche (2009).
    """
    states = {
        'A': ['a1', 'a0'],
        'B': ['b1', 'b0'],
        'C': ['c1', 'c0'],
        'D': ['d1', 'd0'],
        'E': ['e1', 'e0'],
    }

    data = [0.06384, 0.02736, 0.00336, 0.00144, 0, 0.02160,
            0, 0.00240, 0.21504, 0.09216, 0.05376, 0.02304,
            0, 0, 0, 0.09600, 0.01995, 0.00855, 0.00105,
            0.00045, 0, 0.24300, 0, 0.02700, 0.00560, 0.00240,
            0.00140, 0.00060, 0, 0, 0, 0.090]

    return JPTModel(JPT(data, states))


def get_sprinkler_network_from_factors():
    """Return the Sprinkler Bayesian Network."""
    factors = get_sprinkler_factors()
    CPTs = [CPT(f) for f in factors]
    return BayesianNetwork.from_CPTs('Sprinkler', CPTs)


def get_sprinkler_network():
    """Return the Sprinkler Network.
    """
    filename = thomas.core.get_pkg_filename('sprinkler.json')
    return BayesianNetwork.open(filename)


def get_example7_factors():
    """Return the factors for a very simple BN.

    Data copied from "Modeling and Reasoning with Bayesian Networks"
    (page 154) by Adnan Darwiche (2009).
    """
    states = {
        'A': ['a1', 'a0'],
        'B': ['b1', 'b0'],
        'C': ['c1', 'c0'],
    }

    # P(A)
    fA = Factor(
        [0.6, 0.4],
        subset(states, ['A'])
    )

    # P(B|A)
    fB_A = Factor(
        [0.9, 0.1, 0.2, 0.8],
        subset(states, ['A', 'B'])
    )

    # P(C|A)
    fC_B = Factor(
        [0.3, 0.7, 0.5, 0.5],
        subset(states, ['B', 'C'])
    )

    return fA, fB_A, fC_B


def get_example7_network():
    """Return a very simple BN.

    Data copied from "Modeling and Reasoning with Bayesian Networks"
    (page 154) by Adnan Darwiche (2009).
    """
    fA, fB_A, fC_B = get_example7_factors()
    nA = DiscreteNetworkNode('A', cpt=CPT(fA))
    nB = DiscreteNetworkNode('B', cpt=CPT(fB_A))
    nC = DiscreteNetworkNode('C', cpt=CPT(fC_B))

    bn = BayesianNetwork(
        'class',
        [nA, nB, nC],
        [('A', 'B'), ('B', 'C')]
    )

    return bn


def get_example17_2_factors():
    """Return the factors for a very simple BN.

    Data copied from "Modeling and Reasoning with Bayesian Networks"
    (Figure 17.2, page 441) by Adnan Darwiche (2009).
    """
    fH = Factor.from_dict({
        'type': 'Factor',
        'scope': ['H'],
        'states': {'H': ['h0', 'h1']},
        'data': [0.25, 0.75]
     })

    fS_H = Factor.from_dict({
        'type': 'Factor',
        'scope': ['H', 'S'],
        'states': {'H': ['h0', 'h1'], 'S': ['s0', 's1']},
        'data': [0.75, 0.25, 0.84, 0.16]
    })

    fE_H = Factor.from_dict({
        'type': 'Factor',
        'scope': ['H', 'E'],
        'states': {'H': ['h0', 'h1'], 'E': ['e0', 'e1']},
        'data': [0.5, 0.5, 0.084, 0.916]
    })

    return fH, fS_H, fE_H


def get_example17_2_network():
    """Return the network Darwiche's chapter 17.2."""
    fH, fS_H, fE_H = get_example17_2_factors()

    nH = DiscreteNetworkNode('H', name='Health aware', cpt=CPT(fH), position=[165, 29])
    nS = DiscreteNetworkNode('S', name='Smokes', cpt=CPT(fS_H), position=[66,141])
    nE = DiscreteNetworkNode('E', name='Exercices', cpt=CPT(fE_H), position=[288,154])

    # nH = DiscreteNetworkNode('H', cpt=CPT(fH), position=[165, 29])
    # nS = DiscreteNetworkNode('S', cpt=CPT(fS_H), position=[66,141])
    # nE = DiscreteNetworkNode('E', cpt=CPT(fE_H), position=[288,154])

    bn = BayesianNetwork(
        'Health',
        [nH, nS, nE],
        [('H', 'S'), ('H', 'E')]
    )

    return bn


def get_example17_3_factors():
    """..."""
    states = {
        'A': ['a1', 'a2'],
        'B': ['b1', 'b2'],
        'C': ['c1', 'c2'],
        'D': ['d1', 'd2'],
    }

    # P(A)
    fA = Factor(
        [0.2, 0.8],
        subset(states, ['A'])
    )

    # P(B|A)
    fB_A = Factor(
        [0.75, 0.25, 0.10, 0.90],
        subset(states, ['A', 'B'])
    )

    # P(C|A)
    fC_A = Factor(
        [0.5, 0.5, 0.25, 0.75],
        subset(states, ['A', 'C'])
    )

    # P(D|B)
    fD_B = Factor(
        [0.2, 0.8, 0.7, 0.3],
        subset(states, ['B', 'D'])
    )

    return fA, fB_A, fC_A, fD_B


def get_example17_3_network():
    """..."""
    fA, fB_A, fC_A, fD_B = get_example17_3_factors()

    positions = {
        'A': [184, 23],
        'B': [58, 117],
        'C': [288, 117],
        'D': [58, 222]
    }

    nA = DiscreteNetworkNode('A', cpt=CPT(fA), position=positions['A'])
    nB = DiscreteNetworkNode('B', cpt=CPT(fB_A), position=positions['B'])
    nC = DiscreteNetworkNode('C', cpt=CPT(fC_A), position=positions['C'])
    nD = DiscreteNetworkNode('D', cpt=CPT(fD_B), position=positions['D'])

    bn = BayesianNetwork(
        'Example 17.3',
        [nA, nB, nC, nD],
        [('A', 'B'), ('A', 'C'), ('B', 'D')]
    )

    return bn


def get_lungcancer_network():
    """Load 'lungcancer.oobn'."""
    # filename = thomas.core.get_pkg_filename('lungcancer.oobn')
    # return thomas.core.reader.oobn.read(filename)
    filename = thomas.core.get_pkg_filename('lungcancer.json')
    return BayesianNetwork.open(filename)
