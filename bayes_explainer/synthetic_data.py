import os
import random
import numpy as np
from pgmpy.models import LinearGaussianBayesianNetwork

class SyntheticData:

    '''
    Objective: contains methods to create Bayesian Network and create synthetic data from network,
               also contains methods to save network structure information and dataset.
    '''
    
    def __init__(self, name : str):

        self.name = name
        self.model = self.create_network()
        
        pass

    def create_network(self) -> LinearGaussianBayesianNetwork:

        '''
        Objective: creates a Linear Gaussian Bayesian Network with random structure and parameters.
        '''

        num_nodes = random.randint(3, 5)
        edge_probability = random.random()
        model = LinearGaussianBayesianNetwork.get_random(n_nodes=num_nodes, edge_prob=edge_probability, latents=False)

        return model

    def save_edges(self) -> None:
        
        '''
        Objective: saves edges of model in a text file under the structures directory.
        '''

        filename = os.path.join("structures", f"{self.name}.txt")

        with open(filename, "w") as file:
            for parent, child in self.model.edges():
                file.write(f"{parent}:{child}\n")

    def forward_sample(self) -> dict:

        '''
        Objective: performs forward sampling to generate a single particle.
        '''

        order = self.model.nodes()
        particle = {}
        
        for node in order:
            parents = list(self.model.get_parents(node))
            parent_values = [particle[key] for key in parents]
            parent_values = [1] + parent_values
            cpd = self.model.get_cpds(node)
            weights = cpd.beta
            std = cpd.std
            mean = weights.dot(np.array(parent_values))
            value = np.random.normal(mean, std)
            particle[node] = value
        
        return particle

    def create_dataset(self) -> None:

        '''
        Objective: uses forward sampling to create dataset representative of the network.
        '''

        pass