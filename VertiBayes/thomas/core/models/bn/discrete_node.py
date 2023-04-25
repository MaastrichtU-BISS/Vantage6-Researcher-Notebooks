"""DiscreteNetworkNode."""
from ...factors.cpt import CPT
from .node import Node


class DiscreteNetworkNode(Node):
    """Node in a Bayesian Network with discrete values."""

    def __init__(self, RV, name=None, states=None, description='', cpt=None, position=None):
        """Initialize a new discrete Node.

        A Node represents a random variable (RV) in a Bayesian Network. For
        this purpose, it keeps track of a conditional probability distribution
        (CPT).

        Args:
            name (str): Name of the Node. Should correspond to the name of a
                conditioned variable in the CPT.
            states (list): List of states (strings)
            description (str): Name of the Node
        """
        super().__init__(RV, name, description)

        self.states = states or []
        self.position = position if position is not None else [0, 0]

        if cpt is not None:
            self.cpt = cpt

            if self.description == '':
                self.description = cpt.description
        else:
            self._cpt = None

    def __repr__(self):
        """x.__repr__() <==> repr(x)"""
        components = [f"DiscreteNetworkNode('{self.RV}'"]

        if self.name:
            components.append(f"name='{self.name}'")

        if self.states:
            states = ', '.join([f"'{s}'" for s in self.states])
            components.append(f"states=[{states}]")

        if self.description:
            components.append(f"description='{self.description}'")

        return ', '.join(components) + ')'

    @property
    def parents(self):
        if self._cpt:
            parents = dict([(p.RV, p) for p in self._parents])
            sort_order = list(self._cpt.scope[:-1])

            return [parents[p] for p in sort_order]

        return self._parents

    @property
    def conditioned(self):
        """Return the conditioned variable(s)."""
        return self.cpt.conditioned

    @property
    def conditioning(self):
        """Return the conditioning variable(s)."""
        return self.cpt.conditioning

    @property
    def cpt(self):
        """Return the Node's CPT."""
        if self._cpt is None:
            # Create a new, uniform,CPT
            vs = {}

            for p in self.parents:
                vs[p.RV] = p.states

            vs[self.RV] = self.states

            self._cpt = CPT(1, states=vs).normalize()

        return self._cpt

    @cpt.setter
    def cpt(self, cpt):
        """
        Set the Node's CPT.

        This method should only be called *after* the node's parents are known!

        Args:
            cpt (CPT, Factor, pandas.Series): CPT for this node. Can be one of
                CPT, Factor or pandas.Series. Factor or Series require an
                appropriately set Index/MultiIndex.
        """

        # Do a sanity check and ensure that the CPTs has no more then a
        # single conditioned variable. This is only useful if cpt is an
        # actual CPT: for Factor/Series the last level in the index
        # will be assumed to be the conditioned variable.
        if not isinstance(cpt, CPT):
            e = "Argument should be a CPT"
            raise Exception(e)

        elif len(cpt.conditioned) != 1:
            e = "CPT should only have a single conditioned variable"
            raise Exception(e)

        elif cpt.conditioned[0] != self.RV:
            c = cpt.conditioned[0]
            RV = self.RV
            e = f"Conditioned variable '{c}' should correspond to '{RV}'"
            raise Exception(e)

        if not self.states:
            self.states = cpt.states[self.RV]

        # Looking good :-)
        self._cpt = cpt

    @property
    def vars(self):
        """Return the variables in this node (i.e. the scope) as a set."""
        if self._cpt:
            return self._cpt.vars

        return []

    def reset(self):
        """Create a default CPT.

        Throws an Exception if states is not set on this Node or one of its
        parents.
        """
        states = {}

        # Iterate over the parents (all DiscreteNodes) and the node itself to
        # create a dict of states. In Python ≥ 3.6 these dicts are ordered!
        for p in (self._parents + [self, ]):
            if not p.states:
                msg = 'Cannot reset the values of Node (with a parent) without'
                msg += ' states!'
                raise Exception(msg)

            states[p.RV] = p.states

        # Assume a uniform distribution
        self.cpt = CPT(1, states=states).normalize()

    def add_parent(self, parent, **kwargs):
        """Add a parent to the Node.

        Discrete nodes can only have other discrete nodes as parents. If
        succesful, the Node's CPT will be reset.

        Return:
            True iff the parent was added.
        """
        e = "Parent of a DiscreteNetworkNode should be a DiscreteNetworkNode."
        e += f" Not a {type(parent)}"
        assert isinstance(parent, DiscreteNetworkNode), e

        if super().add_parent(parent, **kwargs):
            return True

        return False

    def remove_parent(self, parent):
        """Remove a parent from the Node.

        If succesful, the Node's CPT will be reset.

        Return:
            True iff the parent was removed.
        """
        if super().remove_parent(parent):
            self.reset()
            return True

        return False

    def validate(self):
        """Validate the probability parameters for this Node."""
        if self.cpt.conditioning != [p.RV for p in self._parents]:
            e = "Conditioning variables in CPT should correspond to Node's"
            e += " parents. Order is important!"
            raise Exception(e)

    # --- (de)serialization ---
    def as_dict(self):
        """Return a dict representation of this Node."""
        cpt = self.cpt.as_dict() if self.cpt else None

        d = {
            'type': 'DiscreteNetworkNode',
            'RV': self.RV,
            'name': self.name,
            'states': self.states,
            'description': self.description,
            'cpt': cpt,
            'position': self.position,
        }

        return d

    @classmethod
    def from_dict(cls, d):
        """Return a DiscreteNetworkNode initialized by its dict representation."""
        cpt = CPT.from_dict(d['cpt'])

        node = DiscreteNetworkNode(
            RV=d['RV'],
            name=d['name'],
            states=d['states'],
            description=d['description']
        )
        node.position = d.get('position', (0, 0))
        node.cpt = cpt

        return node
