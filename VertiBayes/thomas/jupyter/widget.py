import ipywidgets as widgets
from traitlets import Integer, Unicode, Any, observe

from thomas.core import BayesianNetwork

# See js/lib/widget.js for the frontend counterpart to this file.


@widgets.register
class BayesianNetworkWidget(widgets.DOMWidget):
    """Widget displaying a Bayesian Network."""

    # Name of the widget view class in front-end
    _view_name = Unicode('View').tag(sync=True)

    # Name of the widget model class in front-end
    _model_name = Unicode('Model').tag(sync=True)

    # Name of the front-end module containing widget view
    _view_module = Unicode('thomas-jupyter-widget').tag(sync=True)

    # Name of the front-end module containing widget model
    _model_module = Unicode('thomas-jupyter-widget').tag(sync=True)

    # Version of the front-end module containing widget view
    _view_module_version = Unicode('^0.1.0').tag(sync=True)
    # Version of the front-end module containing widget model
    _model_module_version = Unicode('^0.1.0').tag(sync=True)

    # Widget properties are defined as traitlets. Any property tagged with `
    # sync=True`is automatically synced to the frontend *any* time it changes
    # in Python. It is synced back to Python from the frontend *any* time the
    # model is touched.
    value = Any().tag(sync=True)
    marginals_and_evidence = Any().tag(sync=True)
    evidence_sink = Any().tag(sync=True)
    height = Integer(300).tag(sync=True)

    def __init__(self, bn: BayesianNetwork, height=300, **kwargs):
        """Create a new instance.

        Args:
            bn (BayesianNetwork): BN to display.
        """
        super().__init__(**kwargs)
        # print(f'Hi! This is widget #{id(self)}')

        self.bn = bn
        self.height = height

        # Associate the widget with the BN
        bn.setWidget(self)

    @property
    def bn(self) -> BayesianNetwork:
        return self._bn

    @bn.setter
    def bn(self, bn: BayesianNetwork):
        """Set the BN on display."""
        self._bn = bn

        self.value = bn.as_dict()
        self.update()

    @property
    def marginals(self):
        if 'marginals' in self.marginals_and_evidence:
            return self.marginals_and_evidence['marginals']

        return {}

    @property
    def evidence(self):
        if 'evidence' in self.marginals_and_evidence:
            return self.marginals_and_evidence['evidence']

        return {}

    def getPositions(self):
        """Return the positions of the nodes."""
        return {n['RV']: n['position'] for n in self.value['nodes']}

    def update(self):
        """Update the marginals using the evidence set on the BN."""
        probs = self.bn.get_marginals()
        evidence = self.bn.evidence
        marginals = {key: value.zipped() for key, value in probs.items()}

        self.marginals_and_evidence = {
            'marginals': marginals,
            'evidence': evidence,
        }

    @observe('evidence_sink')
    def sink_observer(self, value):
        evidence = value['new']
        # print('sink_observer checking in!', evidence)

        for RV, state in evidence.items():
            if state:
                self.bn.set_evidence_hard(RV, state, notify=False)
            else:
                self.bn.reset_evidence(RV, notify=False)

        self.update()

