"""JPT: the Joint Probability Table"""
from typing import Union, Dict

import logging
import numpy as np

# from ..models.base import ProbabilisticModel
from .factor import Factor

log = logging.getLogger(__name__)


class JPT(Factor):
    """Joint Probability Table."""

    def __init__(self, data: Union[int, float, np.array, Factor],
                 states: Dict[str, str] = None):
        """Initialize a new JPT.

        A JPT is simply a Factor over all combinations of RV states that
        sums to 1.

        Args:
            data (list): any iterable
            states (dict): dictionary of random variables and their
                corresponding states.
        """
        if isinstance(data, Factor):
            super().__init__(data.values, data.states)
        else:
            super().__init__(data, states)

        # Ensure the Factor sums to 1
        self.values = self.values / self.values.sum()

    @property
    def display_name(self):
        names = ','.join(self.scope)
        return f'JPT({names})'

    @classmethod
    def from_data(cls, df, cols=None, states=None, complete_value=0):
        """Create a full JPT from data (using Maximum Likelihood Estimation).

        Determine the empirical distribution by ..
          1. counting the occurrences of combinations of variable states; the
             heavy lifting is done by Factor.from_data().
          2. normalizing the result

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
            JPT (normalized)
        """
        factor = Factor.from_data(df, cols, states, complete_value)
        return JPT(factor.normalize().values, states=factor.states)
