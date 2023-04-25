"""Factor: the basis for all reasoning."""
from __future__ import annotations

from typing import List, Dict, Union

from functools import reduce
from itertools import product
import warnings
import logging

import numpy as np
import pandas as pd

from thomas.core import options
from thomas.core import error
# from thomas.core.util import isiterable

log = logging.getLogger(__name__)


# FIXME: refactor to Factor.multiply
def mul(x1, x2):
    """Multiply two Factors (or scalars) with each other.

    Helper function for functools.reduce().
    """
    if isinstance(x1, Factor):
        result = x1.mul(x2)

    elif isinstance(x2, Factor):
        result = x2.mul(x1)

    else:
        result = x1 * x2

    return result


class FactorIndex(object):
    """Index for Factors."""

    def __init__(self, states):
        """Initialize a new FactorIndex."""
        self._index = self.get_index_tuples(states)

    def __getitem__(self, RV):
        """index[x] <==> index.__getitem__(x)"""
        pass

    @staticmethod
    def get_index_tuples(states):
        """Return an index as a list of tuples."""
        return list(product(*states.values()))


class Factor(object):
    """Factor for discrete variables.

    Code is heavily inspired (not to say partially copied from) by pgmpy's
    DiscreteFactor.

    See https://github.com/pgmpy/pgmpy/blob/dev/pgmpy/factors/discrete/DiscreteFactor.py
    """

    def __init__(self, data: Union[int, float, np.array],
                 states: Dict[str, str] = None):
        """Initialize a new Factor.

        Args:
            data (list): any iterable
            states (dict): dictionary of random variables and their
                corresponding states.
        """
        # msg = f'data ({type(data)}) is not iterable? states: {states}'
        # assert isiterable(data), msg

        msg = f"'states should be a dict, but got {type(states)} instead?"
        msg += f" type(data): {type(data)}"
        assert isinstance(states, dict), msg

        if np.isscalar(data):
            total_size = np.product([len(s) for s in states.values()])
            data = np.repeat(data, total_size)

        # Copy & make sure we're dealing with a numpy array
        data = np.array(data, dtype=float)
        # data = np.array(data, dtype=np.longdouble)

        cardinality = [len(i) for i in states.values()]
        expected_size = np.product(cardinality)

        if data.size != expected_size:
            msg = f"Trying to create Factor with states {states}. "
            msg += f"'data' has size {data.size}, which should be: {expected_size}"
            raise ValueError(msg)

        # Set self.states and create indices/mappings.
        self._set_states(states)

        # Storing the data as a multidimensional array helps with addition,
        # subtraction, multiplication and division.
        self.values = data.reshape(cardinality)

    def __repr__(self):
        """repr(f) <==> f.__repr__()"""
        precision = options.get('precision', 4)

        if self.states:
            with pd.option_context('precision', precision):
                s = f'{self.display_name}\n{repr(self.as_series())}'
            return s

        tpl = '{self.display_name}: {' + 'self.values:.{}f'.format(precision) + '}'
        return tpl.format(**locals())

    def __eq__(self, other):
        """f1 == f2 <==> f1.__eq__(f2)"""
        return self.equals(other)

    def __len__(self):
        """len(factor) <==> factor.__len__()"""
        return self.width

    def __contains__(self, var):
        """x in y <--> y.__contains__(x)"""
        return var in self.scope

    def __getitem__(self, keys):
        """factor[x] <==> factor.__getitem__(x)"""

        if isinstance(keys, list):
            indices = [self._states_to_indices(idx) for idx in keys]
            return [self.values[idx] for idx in indices]

        return self.values[self._states_to_indices(keys)]

    def __setitem__(self, keys, value):
        """factor[x] = y <==> factor.__setitem__(x, y)"""
        self.values[self._states_to_indices(keys)] = value

    def __add__(self, other):
        """A + B <=> A.__add__(B)"""
        return self.add(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        """A * B <=> A.mul(B)"""
        return self.mul(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """A / B <=> A.div(B)"""
        return self.div(other)

    def _set_states(self, states):
        """Set self.states and create indices/mappings."""
        self.states = states.copy()

        # create two dicts of dicts to map variable state names to index numbers
        # and back
        # TODO: refactor `name_to_number` to `name_to_position`
        self.name_to_number = {}
        self.number_to_name = {}

        for RV, values in self.states.items():
            self.name_to_number[RV] = {name: nr for nr, name in enumerate(self.states[RV])}
            self.number_to_name[RV] = {nr: name for nr, name in enumerate(self.states[RV])}

    def _states_to_indices(self, states):
        """Return the indices for states.

        Args:
            states (str, list, slice): state values, provided in the same order
                as self.scope.
        """
        if isinstance(states, (str, slice)):
            states = (states, )

        # Create (RV, state) tuples from the provided states
        states = [(self.variables[i], state) for i, state in enumerate(states)]
        indices = [self.get_state_index(RV, state) for RV, state in states]
        return tuple(indices)

    def _get_index_tuples(self, RVs=None):
        """Return an index as a list of tuples.

        Args:
            states (dict): dict of states, indexed by RV

        Return:
            list of tuples, making up all combinations of states.
        """
        # return np.array(list(product(*states.values())))
        if RVs is None:
            states = self.states.values()
        else:
            states = [self.states[RV] for RV in RVs]

        return list(product(*states))

    def _get_state_idx(self, RV):
        """Return ..."""

        # Return the column that corresponds to the position of 'RV'
        idx_cols = np.array(self._get_index_tuples())
        return idx_cols[:, self.variables.index(RV)]

    def _get_bool_idx(self, **states):
        """Return ..."""
        idx_cols = np.array(self._get_index_tuples())

        # Select all entries by default
        trues = np.ones(idx_cols.shape) == 1

        # Filter on kwargs
        for idx, RV in enumerate(self.scope):
            if RV in states:
                trues[:, idx] = idx_cols[:, idx] == states[RV]

        return trues.all(axis=1)

    @property
    def display_name(self):
        names = list(self.states.keys())
        names = ','.join(names)

        return f'factor({names})'

    @property
    def cardinality(self):
        """Return the size of the dimensions of this Factor."""
        return self.values.shape

    @property
    def scope(self):
        """Return the scope of this factor."""
        return list(self.states.keys())

    # Alias
    variables = scope

    @property
    def vars(self):
        """Return the variables in this factor (i.e. the scope) as a *set*."""
        return set(self.scope)

    @property
    def width(self):
        """Return the width of this factor."""
        return len(self.scope)

    @property
    def flat(self):
        """Return the values as a flat list."""
        return self.values.reshape(-1)

    def reorder_scope(self, order, inplace=False):
        """Reorder the scope."""
        factor = self if inplace else Factor.copy(self)

        # rearranging the axes of 'factor' to match 'order'
        variables = factor.variables

        # extend `order` to facilitate setting only the first n values.
        if len(order) < len(variables):
            order = order + list(set(variables) - set(order))

        for axis in range(factor.values.ndim):
            exchange_index = variables.index(order[axis])

            variables[axis], variables[exchange_index] = (
                variables[exchange_index],
                variables[axis],
            )

            factor.values = factor.values.swapaxes(axis, exchange_index)

        # factor.states = {key: factor.states[key] for key in order}
        factor._set_states({key: factor.states[key] for key in order})

        if not inplace:
            return factor

    def align_index(self, other, inplace=False):
        """Align the index to conform to `other`.

        Note: this requires the scope of the two factors to overlap!
        """
        factor = self if inplace else Factor.copy(self)
        original_scope = factor.scope

        # We need at least one overlapping variable to be able to align anything
        shared_vars = list(set(other.variables).intersection(factor.variables))
        shared_states = {RV: other.states[RV] for RV in shared_vars}

        if not len(shared_vars):
            raise error.IncompatibleScopeError(factor.scope, other.scope)

        # Make sure that the shared_vars are first in scope
        factor.reorder_scope(shared_vars, inplace=True)

        # get a list of tuples; each tuple is ordered according to the scope
        shared_indices = other._get_index_tuples(shared_vars)

        remaining_vars = factor.variables[len(shared_vars):]

        if remaining_vars:
            # if there are other variables in scope, we'll need to extend the
            # indices in include theirs
            remaining_indices = factor._get_index_tuples(remaining_vars)

            # `product` creates a list of tuples: combinations of shared_indices
            # and remaining_indices. by adding those we get the full index.
            # _combine_indices = lambda x: x[0] + x[1]
            def _combine_indices(x):
                return x[0] + x[1]

            indices = list(
                map(
                    _combine_indices,
                    product(shared_indices, remaining_indices)
                )
            )

            states = {
                **shared_states,
                **{RV: factor.states[RV] for RV in remaining_vars}
            }

        else:
            # if there are no other variables in scope, we can just use the
            # shared_indices to rearrange the values
            indices = shared_indices
            states = shared_states

        # sort factor.values according to the indices
        cardinality = [len(i) for i in states.values()]
        values = np.array(factor[indices]).reshape(cardinality)

        factor._set_states(states)
        factor.values = values
        factor.reorder_scope(original_scope, inplace=True)

        if not inplace:
            return factor

    def extend_with(self, other, inplace=False):
        """Extend this factor with the variables & states of another."""
        factor = self if inplace else Factor.copy(self)

        shared_vars = set(other.variables).intersection(factor.variables)

        if shared_vars:
            # Make sure that the *shared* variables have the same states / the
            # indices are aligned.
            factor.align_index(other, inplace=True)

            f_states = {RV: factor.states[RV] for RV in shared_vars}
            o_states = {RV: other.states[RV] for RV in shared_vars}

            if f_states != o_states:
                # ---> move alignment here when it works
                raise error.StatesNotAlignedError(f_states, o_states)

        # Note: the order of 'extra_vars' holds no importance ..
        extra_vars = set(other.variables) - set(factor.variables)

        if extra_vars:
            # Create as many new dimensions in the array as there are
            # additional variables.
            slice_ = [slice(None)] * len(factor.variables)
            slice_.extend([np.newaxis] * len(extra_vars))
            factor.values = factor.values[tuple(slice_)]

            factor.states.update(other.states)
            factor.name_to_number.update(other.name_to_number)
            factor.number_to_name.update(other.number_to_name)

        if not inplace:
            return factor

    @staticmethod
    def extend_and_reorder(factor, other):
        """Extend factors factor and other to be over the same scope and
        reorder their axes to match.
        """
        # Assuming 'other' is another Factor.
        other = Factor.copy(other)

        # modifying 'factor' (self) to add new variables
        factor.extend_with(other, inplace=True)

        # modifying 'other' to add new variables
        other.extend_with(factor, inplace=True)

        # rearranging the axes of 'other' to match 'factor'
        other.reorder_scope(factor.variables, inplace=True)

        # Since factor was modified in place, returning it is technically
        # unnecessary   .
        return factor, other

    @staticmethod
    def multiply(self, factors: List[Factor]) -> Factor:
        """Multiply a set of factors ..."""
        msg = "Argument 'factors' should be a List[Factor]"
        assert all([isinstance(f, Factor) for f in factors]), msg
        return reduce(mul, factors)

    def copy(self):
        """Return a copy of this Factor."""
        return Factor(self.values, self.states)

    def sum(self):
        """Sum all values of the factor."""
        return self.values.sum()

    def add(self, other, inplace=False):
        """A + B <=> A.add(B)"""
        factor = self if inplace else Factor.copy(self)

        if isinstance(other, (int, float)):
            factor.values += other

        else:
            # # Assuming 'other' is another Factor.
            factor, other = self.extend_and_reorder(factor, other)
            factor.values = factor.values + other.values

        if not inplace:
            return factor

    def mul(self, other, inplace=False):
        """A * B <=> A.mul(B)"""
        factor = self if inplace else Factor.copy(self)

        if isinstance(other, (int, float)):
            factor.values *= other

        else:
            # # Assuming 'other' is another Factor.
            factor, other = self.extend_and_reorder(factor, other)
            factor.values = factor.values * other.values

        if not inplace:
            return factor

    def div(self, other, inplace=False):
        """A / B <=> A.div(B)"""
        factor = self if inplace else Factor.copy(self)

        if isinstance(other, (int, float)):
            factor.values /= other

        else:
            # # Assuming 'other' is another Factor.
            factor, other = self.extend_and_reorder(factor, other)

            with warnings.catch_warnings(record=True):
                # Cause all warnings to always be triggered.
                warnings.simplefilter("always")

                factor.values = factor.values / other.values
                factor.values[np.isnan(factor.values)] = 0

        if not inplace:
            return factor

    def get_state_names(self, RV, idx):
        """Return the state for RV at idx."""
        return self.number_to_name[RV][idx]

    def del_state_names(self, RVs):
        """Deletes the state names for variables in RVs."""
        for RV in RVs:
            del self.states[RV]
            del self.name_to_number[RV]
            del self.number_to_name[RV]

    def get_state_index(self, RV, state):
        """Return the index for RV with state."""
        if isinstance(state, slice):
            return state

        if isinstance(state, (tuple, list)):
            return [self.name_to_number[RV][s] for s in state]

        try:
            return self.name_to_number[RV][state]
        except Exception:
            # print(f'self.name_to_number: {self.name_to_number}')
            # print(f'RV: {RV}')
            # print(f'state: {state}')
            raise

    def get(self, **kwargs):
        """Return the cells identified by kwargs.

        Examples
        --------
        >>> factor = Factor([1, 1], {'A': ['a0', 'a1']})
        >>> print(factor)
        factor(A)
        A
        a0    1.0
        a1    1.0
        dtype: float64
        >>> factor.get(A='a0')
        array([1.])
        """
        return self.flat[self._get_bool_idx(**kwargs)]

    def set(self, value, inplace=False, **kwargs):
        """Set a value to cells identified by **kwargs.

        Examples
        --------
        >>> factor = Factor([1, 1], {'A': ['a0', 'a1']})
        >>> print(factor)
        factor(A)
        A
        a0    1.0
        a1    1.0
        dtype: float64
        >>> factor.set(0, A='a0')
        >>> print(factor)
        factor(A)
        A
        a0    0.0
        a1    1.0
        dtype: float64
        """
        factor = self if inplace else Factor.copy(self)

        factor.flat[factor._get_bool_idx(**kwargs)] = value

        if not inplace:
            return factor

    def set_complement(self, value, inplace=False, **kwargs):
        """Set a value to cells *not* identified by **kwargs.

        Examples
        --------
        >>> factor = Factor([1, 1], {'A': ['a0', 'a1']})
        >>> print(factor)
        factor(A)
        A
        a0    1.0
        a1    1.0
        dtype: float64
        >>> factor.set_complement(0, A='a0')
        >>> print(factor)
        factor(A)
        A
        a0    1.0
        a1    0.0
        dtype: float64
        """
        factor = self if inplace else Factor.copy(self)

        idx = np.invert(factor._get_bool_idx(**kwargs))
        factor.flat[idx] = value

        if not inplace:
            return factor

    def equals(self, other):
        """Return True iff two Factors have (roughly) the same values."""
        if not isinstance(other, Factor):
            return False

        if self.values.size != other.values.size:
            return False

        if set(self.scope) != set(other.scope):
            return False

        reordered = other.reorder_scope(self.scope).align_index(self)
        return np.allclose(self.values, reordered.values)

    def normalize(self, inplace=False):
        """Normalize the Factor so the sum of all values is 1."""
        factor = self if inplace else self.copy()

        factor.values = factor.values / factor.values.sum()

        if not inplace:
            return factor

    def sum_out(self, variables, inplace=False):
        """Sum-out (marginalize) a variable (or list of variables) from the
        factor.

        Args:
            variables (str, list): Name or list of names of variables to sum out.

        Returns:
            Factor: factor with the specified variable removed.
        """
        factor = self if inplace else Factor.copy(self)

        if isinstance(variables, (str, tuple)):
            variable_set = set([variables])
        else:
            variable_set = set(variables)

        if len(variable_set) == 0:
            # Nothing to sum out ...
            return factor

        scope_set = set(factor.scope)

        if not variable_set.issubset(scope_set):
            raise error.NotInScopeError(variable_set, scope_set)

        # Find the indices of the variables to sum out
        var_indexes = [factor.scope.index(var) for var in variable_set]

        # Remove the variables from the factor
        factor.del_state_names(variable_set)

        # Sum over the variables we just deleted. This can reduce the result
        # to a scalar.
        factor.values = np.sum(factor.values, axis=tuple(var_indexes))

        if not inplace:
            return factor

    def project(self, Q, inplace=False):
        """Project the current factor on Q.

        Args:
            Q (set or str): variable(s) to compute marginal over.

        Returns:
            Factor:  marginal distribution over the RVs in Q.
        """
        if isinstance(Q, (list, tuple)):
            Q = set(Q)

        assert isinstance(Q, (set, str)), "Q should be a set or a string!"

        if isinstance(Q, str):
            Q = {Q}

        factor = self if inplace else Factor.copy(self)
        vars_to_sum_out = list(set(factor.scope) - Q)

        factor.sum_out(vars_to_sum_out, inplace=True)

        if not inplace:
            return factor

    def zipped(self):
        """Return a dict with data, indexed by tuples."""
        tuples = self._get_index_tuples()

        if self.width == 1:
            tuples = [t[0] for t in tuples]

        return dict(zip(tuples, self.values))

    @classmethod
    def from_series(cls, series):
        """Create a Factor from a (properly indexed) pandas.Series."""
        states = cls.index_to_states(series.index)

        # Sanity check
        full_idx = cls.states_to_index(states)
        if len(full_idx) != len(series):
            # Create a Factor with all zeros and update it with the values of
            # series.
            factor = Factor(0, states)
            for idx in factor._get_index_tuples():
                factor[idx] = series.get(idx, 0)

        else:
            factor = Factor(series[full_idx].values, states)

        return factor

    def as_series(self):
        """Return a pandas.Series."""
        log.debug("Factor.as_series()")

        idx = pd.MultiIndex.from_product(
            self.states.values(),
            names=self.states.keys()
        )

        data = self.values.reshape(-1)

        log.debug(f"  idx: {idx}")
        log.debug(f"  data: {data}")

        return pd.Series(data, index=idx)

    @staticmethod
    def index_to_states(idx: Union[pd.Index, pd.MultiIndex]) -> dict:
        """Create a state dict from a pandas.Index or MultiIndex.

        Note that this may return the states in different order since
        pandas doesn't provide a way to order level values.
        """
        if isinstance(idx, pd.MultiIndex):
            states = {i.name: list(i) for i in idx.levels}
        else:
            states = {idx.name: list(idx)}

        return states

    @staticmethod
    def states_to_index(states):
        """Create a pandas.Index or MultiIndex from a state dict."""
        idx = pd.MultiIndex.from_product(
            states.values(),
            names=states.keys()
        )

        return idx

    def get_pandas_index(self):
        """Return a pandas.Index or pandas.MultiIndex."""
        return self.states_to_index(self.states)

    @classmethod
    def from_dict(cls, d):
        """Return a Factor initialized by its dict representation."""

        # Scope is guaranteed to be ordered in JSON
        scope = d['scope']

        states = {RV: d['states'][RV] for RV in scope}
        return Factor(d['data'], states)

    def as_dict(self):
        """Return a dict representation of this Factor."""

        return {
            'type': 'Factor',
            'scope': self.scope,
            'states': self.states,
            'data': self.values.tolist(),
        }

    @classmethod
    def from_data(cls, df, cols=None, states=None, complete_value=0):
        """Create a full Factor from data (using Maximum Likelihood Estimation).

        Determine the empirical distribution by counting the occurrences of
        combinations of variable states.

        Note that the this will *drop* any NAs in the data.

        Args:
            df (pandas.DataFrame): data
            cols (list): columns in the data frame to use. If `None`, all
                columns are used.
            states (dict): list of allowed states for each random
                variable, indexed by name. If `states` is None it will be
                determined from the data.
            complete_value (int): Base (count) value to use for combinations of
                variable states in the dataset.

        Return:
            Factor (unnormalized)
        """
        cols = cols if cols else list(df.columns)
        subset = df[cols]
        counts = subset.groupby(cols).size()

        if states is None:
            # We'll need to try to determine states from the df
            states = cls.index_to_states(counts.index)

        if len(cols) > 1:
            # Create a factor containing *all* combinations set to `complete_value`.
            f2 = Factor(complete_value, states)

            # By summing the Factor with the Series all combinations not in the
            # data are set to `complete_value`.
            total = f2.as_series() + counts
        else:
            total = counts
        values = np.nan_to_num(total.values, nan=complete_value)
        return Factor(values, states=states)
