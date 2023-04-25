"""CPT: Conditional Probability Table."""
import logging

import pandas as pd

from thomas.core import options
from .factor import Factor

log = logging.getLogger(__name__)


class CPT(Factor):
    """Conditional Probability Distribution.

    A CPT is essentially a Factor that knows which variables in its scope are
    the conditioning variables. We also *display* the CPT differently:
      - the random variable states make up the columns
      - the conditioning variable states make up the rows.
    """

    def __init__(self, data, states=None, conditioned=None, description=''):
        """Initialize a new CPT.

        Args:
            data (list, pandas.Series, Factor): array of values.
            conditioned (list): list of conditioned variables
            states (dict): list of allowed states for each random
                variable, indexed by name. If states is None, `data`
                should be a pandas.Series (or Factor) with a proper
                Index/MultiIndex.
            description (str): An optional description of the random variables'
                meaning.
        """
        if isinstance(data, Factor):
            super().__init__(data.values, data.states)
        else:
            super().__init__(data, states)

        # Each name in the index corresponds to a random variable. We'll assume
        # the last variable of the index is being conditioned if not explicitly
        # specified.
        if conditioned is None:
            conditioned = [self.scope[-1]]
        else:
            conditioned = list(conditioned)

        # The remaining variables must be the conditioning variables
        conditioning = [i for i in self.scope if i not in conditioned]

        # Make sure the conditioned variable appears rightmost in the index.
        if self.width > 1:
            order = conditioning + conditioned
            self.reorder_scope(order, inplace=True)

        # Make sure that the CPTs rows sum to 1. We'll need to add a level
        # to row_sum for broadcasting to work properly.
        if len(conditioning):
            log.debug(f"Conditioning on {len(conditioning)} variable(s), so normalizing rows!")
            sum_idx = (slice(None), ) * len(conditioning) + (None, ) * len(conditioned)
            axes = tuple(range(-1, -len(conditioned) - 1, -1))

            log.debug(f"  sum_idx: {sum_idx}")
            log.debug(f"  axes: {axes}")
            row_sum = self.values.sum(axis=axes)[sum_idx]
            self.values = self.values / row_sum

        else:
            log.debug("Keeping the rows as is, we're not conditioning on anything.")

        # Set remaining attributes
        self.conditioned = conditioned
        self.conditioning = conditioning
        self.description = description

    @classmethod
    def _short_query_str(cls, sep1, sep2, conditioned, conditioning):
        """Return a short query string."""
        conditioned = sep1.join(conditioned)
        conditioning = sep1.join(conditioning)

        if conditioning:
            return f'{conditioned}{sep2}{conditioning}'

        return f'{conditioned}'

    def short_query_str(self, sep1=',', sep2='|'):
        """Return a short version of the query string."""
        return self._short_query_str(
            sep1,
            sep2,
            self.conditioned,
            self.conditioning
        )

    @property
    def display_name(self):
        """Return a short version of the query string."""
        return f'P({self.short_query_str()})'

    def _repr_html_(self):
        """Return an HTML representation of this CPT.

        Note that the order of the index may differ as pandas sorts it when
        performing `unstack()`.
        """
        precision = options.get('precision', 4)
        data = self.as_series()

        with pd.option_context('precision', precision):
            if self.conditioning:
                html = data.unstack(self.conditioned)._repr_html_()
            else:
                df = pd.DataFrame(data, columns=['']).transpose()
                html = df._repr_html_()

        return f"""
            <div>
                <div style="margin-top:6px">
                    <span><b>{self.display_name}</b></span>
                    <span style="font-style: italic;">{self.description}</span>
                    {html}
                </div>
            </div>
        """

    def copy(self):
        """Return a copy of this CPT."""
        return CPT(
            self.values,
            self.states,
            self.conditioned,
            self.description
        )

    def as_factor(self):
        """Return a copy this CPT as a Factor."""
        return Factor(self.values, self.states)

    def as_dataframe(self):
        """Return the CPT as a pandas.DataFrame."""
        data = self.as_series()

        if self.conditioning:
            data = data.unstack(self.conditioned)

        return data

    @classmethod
    def from_factor(cls, factor):
        """Create a CPT from a Factor.

        This is equivalent to calling CPT(factor) and is provided merely for
        consistency.
        """
        return cls(factor)

    def as_dict(self):
        """Return a dict representation of this CPT."""
        d = super().as_dict()
        d.update({
            'type': 'CPT',
            'description': self.description,
            'conditioned': self.conditioned,
            'conditioning': self.conditioning
        })

        return d

    @classmethod
    def from_dict(cls, d):
        """Return a CPT initialized by its dict representation."""
        factor = super().from_dict(d)

        return CPT(
            factor,
            conditioned=d.get('conditioned'),
            description=d.get('description')
        )


