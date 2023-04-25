# -*- coding: utf-8 -*-
"""
OOBN reader
"""
import numpy as np
import pandas as pd
import lark

from ..util import flatten

from ..factors.cpt import CPT
from ..models import bn

GRAMMAR = r"""
    start: group+

    group: net | node | potential

    net: "net" properties [comment]
    node: "node" name properties
    potential: "potential" "(" name ["|" parents] ")" properties

    name: CNAME

    properties: "{" property* "}"
    property: name "=" value ";" [comment]

    parents: name*

    ?value: string
          | number
          | "boolean"
          | tuple

    string: ESCAPED_STRING
    number: SIGNED_NUMBER
    tuple: "(" value* (comment value)* ")" [comment]

    comment: "%" /.+/ NEWLINE

    %import common.CNAME
    %import common.ESCAPED_STRING
    %import common.SIGNED_NUMBER
    %import common.NEWLINE
    %import common.WS
    %ignore WS
"""


class BasicTransformer(lark.Transformer):
    """Transform lark.Tree into basic, Python native, objects."""

    def start(self, items):
        """Starting point."""
        # print('--- START ---')
        # print(items)
        # print()

        flattened = flatten(items)
        net = [f for f in flattened if f['type'] == 'net'][0]
        nodes = [f for f in flattened if f['type'] == 'node']
        potentials = [f for f in flattened if f['type'] == 'potential']

        net['nodes'] = {n['name']: n for n in nodes}
        net['potentials'] = {p['name']: p for p in potentials}

        return net

    def group(self, items):
        """Starting point."""
        # print('--- GROUP ---')
        # print(items)
        # print()

        return items

    def net(self, items):
        """A group is either a 'net', 'node' or 'potential'"""
        # print('--- NET ---')
        # print(items)
        # print()

        properties = items[0]

        net_obj = {
            'type': 'net',
            'name': None,
            'node_size': [80, 40],
        }

        net_obj.update(properties)

        return net_obj

    def node(self, items):
        """Subclass of the root element.

        Contains a list of states for a particular node.
        """
        # print('--- NODE ---')
        # print(items)
        # print()

        name, properties = items
        node = {
            'type': 'node',
            'name': name,
            'class': 'DiscreteNetworkNode',
            'position': [0, 0],
        }
        node.update(properties)

        return node

    def potential(self, items):
        """Subclass of the root element.

        Contains a node's
         * the prior/conditional probability table (CPT)
         * experience table
        """
        # print('--- POTENTIAL ---')
        # print(items)
        # print('--- END of POTENTIAL ---')
        # print()

        parents = []

        if len(items) == 2:
            name, properties = items
        elif len(items) == 3:
            name, parents, properties = items

        potential = {
            'type': 'potential',
            'name': name,
            'parents': parents,
        }

        potential.update(properties)

        return potential

    def name(self, tokens):
        """A name is essentially an unquoted string."""
        # tokens: list of Token; Token is a subclass of string
        return str(tokens[0])

    def properties(self, items):
        """
        Properties of the root element or one of its contained subclassess.
        """
        properties = []
        nodes = {}
        potentials = {}

        for i in items:
            if isinstance(i, tuple):
                # Every property is a (key, value) tuple
                properties.append(i)

        return dict(properties)

    def property(self, items):
        """A property consists of a simple key, value pair."""
        name, value = items[:2]
        return (name, value)

    def parents(self, items):
        """Represents the parents of a node in the DAG."""
        return list(items)

    def string(self, items):
        """Quoted string."""
        # Remove the quotes ...
        return str(items[0][1:-1])

    def number(self, items):
        """Number."""
        return float(items[0])

    def tuple(self, items):
        """Tuple."""
        return [i for i in items if i is not None]

    def comment(self, items):
        """Comment. Ignored."""
        return None


def _parse(filename):
    with open(filename, 'r') as fp:
        net_data = fp.read()

    parser = lark.Lark(GRAMMAR, parser='lalr')
    tree = parser.parse(net_data)

    return tree

def _parseFromString(string):

    parser = lark.Lark(GRAMMAR, parser='lalr')
    tree = parser.parse(string)

    return tree


def _transform(tree):
    transformer = BasicTransformer()
    return transformer.transform(tree)


def _create_structure(tree):
    # dict, indexed by node name
    nodes = {}

    # list of tuples
    edges = []

    # Iterate over the list of nodes
    for name in tree['nodes']:
        node = tree['nodes'][name]
        potential = tree['potentials'][name]
        node_parents = potential['parents']

        if node_parents is not None:
            for parent in node_parents:
                edges.append((parent, name))

        node_states = node['states']
        node_position = node['position']
        parent_states = {}
        if node_parents is not None:
            for parent in node_parents:
                parent_states[parent] = tree['nodes'][parent]['states']

        states = {
            name: node_states
        }

        states.update(parent_states)

        # Get the data for the prior/conditional probability distribution.
        # In BNs that do not have a CPT yet, the 'data' key will be missing.
        data = potential.get('data', 1)

        if not isinstance(data, int):
            data = np.array(data)

        # If there are parents, it's a CPT
        if node_parents is not None and len(node_parents) > 0:
            # Prepare the indces for the dataframe
            index = pd.MultiIndex.from_product(
                parent_states.values(),
                names=parent_states.keys()
            )
            columns = pd.Index(node_states, name=name)
            data = data.reshape(-1, len(columns))
            df = pd.DataFrame(data, index=index, columns=columns)
            stacked = df.stack()

            # This keeps the index order
            cpt = CPT(
                stacked,
                states={n: states[n] for n in stacked.index.names},
                conditioned=[name],
            )

        # Else, it's a probability table
        else:
            try:
                cpt = CPT(
                    data,
                    conditioned=[name],
                    states=states
                )
            except Exception:
                raise

        # Get the data for the dataframe
        nodes[name] = {
            'class': node['class'],
            'RV': name,
            'name': name,
            'states': node_states,
            'parents': node_parents,
            'position': node_position,
            'CPT': cpt
        }

    network = {
        'name': tree['name'],
        'node_size': tree['node_size'],
        'nodes': nodes,
        'edges': edges,
    }

    return network


def _create_bn(structure):
    """Create a BayesianNetwork from a previously created structure."""
    nodes = []

    for name, node_properties in structure['nodes'].items():
        RV = node_properties['RV']
        states = node_properties['states']
        position = node_properties['position']
        description = ''

        cpt = node_properties['CPT']

        constructor = getattr(bn, node_properties['class'])

        n = constructor(RV, name, states, description, cpt)
        n.position = position
        nodes.append(n)

    edges = structure['edges']
    return bn.BayesianNetwork(structure['name'], nodes, edges)


def read(filename):
    """Parse the OOBN file and transform it into a sensible dictionary."""
    # Parse the OOBN file
    tree = _parse(filename)

    # Transform the parsed tree into native objects
    transformed = _transform(tree)

    # The transformed tree still has separate keys for nodes' states and their
    # parents and PDs/CPDs. Let's merge the structure to have all relevant data
    # together and let's transform the PDs/CPDs into Pandas DataFrames.
    structure = _create_structure(transformed)

    return _create_bn(structure)

def readFromString(string):
    """Parse the OOBN file and transform it into a sensible dictionary."""
    # Parse the OOBN file
    tree = _parseFromString(string)

    # Transform the parsed tree into native objects
    transformed = _transform(tree)

    # The transformed tree still has separate keys for nodes' states and their
    # parents and PDs/CPDs. Let's merge the structure to have all relevant data
    # together and let's transform the PDs/CPDs into Pandas DataFrames.
    structure = _create_structure(transformed)

    return _create_bn(structure)
