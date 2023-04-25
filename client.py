import vantage6.client

IMAGE = 'harbor.carrier-mu.src.surf-hosted.nl/carrier/vertibayes:3.0'
NAME = 'vertibayes from client'

# Example network VertiBayesAsia
INITIAL_NETWORK = [{
    "parents": [],
    "name": "asia",
    "type": "string",
    "probabilities": [],
    "bins": []
}, {
    "parents": ["asia"],
    "name": "tub",
    "type": "string",
    "probabilities": [],
    "bins": []
}, {
    "parents": [],
    "name": "smoke",
    "type": "string",
    "probabilities": [],
    "bins": []
}, {
    "parents": ["smoke"],
    "name": "lung",
    "type": "string",
    "probabilities": [],
    "bins": []
}, {
    "parents": ["smoke"],
    "name": "bronc",
    "type": "string",
    "probabilities": [],
    "bins": []
}, {
    "parents": ["tub", "lung"],
    "name": "either",
    "type": "string",
    "probabilities": [],
    "bins": []
}, {
    "parents": ["either"],
    "name": "xray",
    "type": "string",
    "probabilities": [],
    "bins": []
}, {
    "parents": ["either", "bronc"],
    "name": "dysp",
    "type": "string",
    "probabilities": [],
    "bins": []
}]


class VertibayesClient:

    def __init__(self, client: vantage6.client.Client):
        """

        :param client: Vantage6 client
        """
        self.client = client

    def vertibayes(self, collaboration, commodity_node, nodes, targetVariable, minPercentage, folds):
        return self.client.task.create(collaboration=collaboration,
                                       organizations=[commodity_node],
                                       name=NAME, image=IMAGE, description=NAME,
                                       input={'method': 'vertibayes', 'master': True,
                                              'args': [nodes, INITIAL_NETWORK, targetVariable, minPercentage, folds,
                                                       INITIAL_NETWORK]})

    def vertibayesNoInitialNetwork(self, collaboration, commodity_node, nodes, targetVariable, minPercentage, folds):
        return self.client.task.create(collaboration=collaboration,
                                       organizations=[commodity_node],
                                       name=NAME, image=IMAGE, description=NAME,
                                       input={'method': 'vertibayes', 'master': True,
                                              'args': [nodes, None, targetVariable, minPercentage, folds]})
