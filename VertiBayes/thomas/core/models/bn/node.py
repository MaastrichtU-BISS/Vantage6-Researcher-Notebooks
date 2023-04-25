"""Node."""
from typing import List
import sys


class Node(object):
    """Base class for discrete and continuous nodes in a Bayesian Network.

    In Hugin, discrete nodes can only have other discrete nodes as parents.
    Continous nodes can have either continuous or discrete nodes as parents.

    BayesiaLab does allow discrete nodes to have continous nodes as parents by
    associating discrete states with the continous value.
    """

    def __init__(self, RV, name=None, description=''):
        """Initialize a new Node.

        Args:
            RV (str): Name of the (conditioned) random variable
            name (str): Name of the Node.
            description (str): Name of the Node
        """
        self.RV = RV
        self.name = name or RV
        self.description = description

        # A node needs to know its parents in order to determine the shape of
        # its CPT. This should be a list of Nodes.
        self._parents: List[Node] = []

        # For purposes of message passing, a node also needs to know its
        # children.
        self._children: List[Node] = []

    @property
    def parents(self):
        return self._parents

    def has_parents(self):
        """Return True iff this node has a parent."""
        return len(self._parents) > 0

    def has_children(self):
        """Return True iff this node has children."""
        return len(self._children) > 0

    def add_parent(self, parent, add_child=True):
        """Add a parent to the Node.

        If succesful, the Node's distribution's parameters (ContinousNode) or
        CPT (DiscreteNode) should be reset.

        Args:
            parent (Node): parent to add.
            add_child (bool): iff true, this node is also added as a child to
                the parent.

        Return:
            True iff the parent was added.
        """
        if parent not in self._parents:
            self._parents.append(parent)

            if add_child:
                parent.add_child(self, add_parent=False)

            return True

        return False

    def add_child(self, child: object, add_parent=True) -> bool:
        """Add a child to the Node.

        Args:
            child (Node): child to add.
            add_child (bool): iff true, this node is also added as a parent to
                the child.

        Return:
            True iff the child was added.
        """
        if child not in self._children:
            self._children.append(child)

            if add_parent:
                child.add_parent(self, add_child=False)

            return True

        return False

    def remove_parent(self, parent: object, remove_child=True) -> bool:
        """Remove a parent from the Node.

        If succesful, the Node's distribution's parameters (ContinousNode) or
        CPT (DiscreteNode) should be reset.

        Return:
            True iff the parent was removed.
        """
        if parent in self._parents:
            self._parents.remove(parent)

            if remove_child:
                parent._children.remove(self)

            return True

        return False

    def remove_child(self, child: object, remove_parent=True) -> bool:
        """Remove a child from the Node.

        Return:
            True iff the parent was removed.
        """
        if child in self._children:
            self._children.remove(child)

            if remove_parent:
                child._parents.remove(self)

            return True

        return False

    def validate(self):
        """Validate the probability parameters for this Node."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d):
        """Return a Node (subclass) initialized by its dict representation."""
        clsname = d['type']

        if clsname == cls.__name__:
            raise Exception('Cannot instantiate abstract class "Node"')

        clstype = getattr(sys.modules[__name__], clsname)
        return clstype.from_dict(d)


# I'm sure there's a more elegant way to deal with finding subclasses in
# Node.from_dict() ...
from .discrete_node import DiscreteNetworkNode  # noqa
