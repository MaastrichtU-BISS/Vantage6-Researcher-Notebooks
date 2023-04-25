"""Base for probabilistic models."""

class ProbabilisticModel(object):

    @classmethod
    def parse_query_string(cls, query_string):
        """Parse a query string into a tuple of query_dist, query_values,
        evidence_dist, evidence_values.

        The query P(I,G=g1|D,L=l0) would imply:
            query_dist = ('I',)
            query_values = {'G': 'g1'}
            evidence_dist = ('D',)
            evidence_values = {'L': 'l0'}
        """
        def split(s):
            dist, values = [], {}
            params = []

            if s:
                params = s.split(',')

            for p in params:
                if '=' in p:
                    key, value = p.split('=')
                    values[key] = value
                else:
                    dist.append(p)

            return dist, values

        query_str, given_str = query_string, ''

        if '|' in query_str:
            query_str, given_str = query_string.split('|')

        return split(query_str) + split(given_str)

    @classmethod
    def create_query_string(cls, qd=None, qv=None, ed=None, ev=None):
        """Generate a query string."""
        qd_str = ','.join(qd) if qd else ''
        qv_str = ','.join([f'{k}={v}' for k, v in qv.items()]) if qv else ''

        ed_str = ','.join(ed) if ed else ''
        ev_str = ','.join([f'{k}={v}' for k, v in ev.items()]) if ev else ''

        Q = ','.join([q for q in [qd_str, qv_str] if q])
        E = ','.join([e for e in [ed_str, ev_str] if e])

        return '|'.join([p for p in [Q, E] if p])

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
        raise NotImplementedError

    def P(self, query_string):
        """Return the probability as queried by query_string.

        P('I,G=g1|D,L=l0') is equivalent to calling compute_posterior with:
            query_dist = ('I',)
            query_values = {'G': 'g1'}
            evidence_dist = ('D',)
            evidence_values = {'L': 'l0'}
        """
        qd, qv, gd, gv = self.parse_query_string(query_string)
        return self.compute_posterior(qd, qv, gd, gv)

    # def MAP(self, query_dist, evidence_values, include_probability=True):
    #     """Perform a Maximum a Posteriori query."""
    #     d = self.compute_posterior(query_dist, {}, [], evidence_values)
    #     evidence_vars = [e for  e in evidence_values.keys() if e in d.scope]
    #
    #     d = d.droplevel(evidence_vars)
    #
    #     if include_probability:
    #         return d.idxmax(), d.max()
    #
    #     return d.idxmax()
