"""JPTModel: probilistic model based upon a Joint Probability Distribution."""
from typing import Union, Dict

import logging

import numpy as np

from .base import ProbabilisticModel

from ..factors.factor import Factor
from ..factors.cpt import CPT
from ..factors.jpt import JPT

log = logging.getLogger(__name__)


class JPTModel(JPT, ProbabilisticModel):
    """Probilistic model based upon a Joint Probability Distribution."""

    def __init__(self, data: Union[int, float, np.array, Factor, JPT],
                 states: Dict[str, str] = None):
        """Create a new JPT instance."""
        if isinstance(data, (Factor, JPT)):
            super().__init__(data.values, data.states)
        else:
            super().__init__(data, states)

    def compute_dist(self, qd, ed=None):
        """Compute a (conditional) distribution.

        This is short for self.compute_posterior(qd, {}, ed, {})
        """
        if ed is None:
            ed = []

        return self.compute_posterior(qd, {}, ed, {})

    def compute_posterior(self, qd, qv, ed, ev):
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
            CPT

        """
        log.debug(f"JPT.compute_posterior({qd}, {qv}, {ed}, {ev})")
        # Get a list of *all* variables to query
        query_vars = list(qv.keys()) + qd
        evidence_vars = list(ev.keys()) + ed

        result = self.project(query_vars + evidence_vars)

        if evidence_vars:
            result = result / result.sum_out(query_vars)

        # If query values were specified we can extract them from the factor.
        if qv:
            result = result.get(**qv)

        if isinstance(result, Factor):
            log.debug(f"result.values: {result.values}")
            return CPT(
                result.values,
                states=result.states,
                conditioned=query_vars
            )

        elif isinstance(result, np.ndarray) and len(result) == 1:
            result = result[0]

        return result
