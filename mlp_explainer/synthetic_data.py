from typing import Dict

import os
import random
import numpy as np
import pandas as pd
import networkx as nx
from pgmpy.models import LinearGaussianBayesianNetwork

class SyntheticData:

    '''
    Objective: contains methods to create Bayesian Network and create synthetic data from network,
               also contains methods to save network structure information and dataset.
    '''
    
    def __init__(self, identifier : str):

        self.identifier = identifier
        self.model = self.__create_network()
        self.dataset = [] # rows are instances, columns are variables

    def __create_network(self) -> LinearGaussianBayesianNetwork:

        '''
        Objective: creates a Linear Gaussian Bayesian Network with random structure and parameters.
        '''

        num_nodes = random.randint(8, 20)
        edge_probability = random.uniform(0, 0.5) # plausible way to regulate in-degree per node
        model = LinearGaussianBayesianNetwork.get_random(n_nodes=num_nodes, edge_prob=edge_probability, latents=False)
        return model

    def __forward_sample(self) -> Dict:

        '''
        Objective: performs forward sampling to generate a single particle.
        '''

        order = nx.topological_sort(self.model)
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

        N = 2 ** (len(self.model.nodes()) + 2)

        for n in range(N):
            sample = self.__forward_sample()
            datapoint = np.array([sample[key] for key in sample])
            self.dataset.append(datapoint)
    
    def save(self) -> None:
        
        '''
        Objetive: saves Linear Gaussian Bayesian Network model and dataset.
        '''

        data_filename = os.path.join("data", f"{self.identifier}_dataset.csv")
        data_frame = pd.DataFrame(np.array(self.dataset), columns=list(self.model.nodes()))
        data_frame.to_csv(data_filename, index=False)
    
        structure_filename = os.path.join("structures", f"{self.identifier}_ground.txt")

        with open(structure_filename, "w") as file:
            for node in self.model.nodes():
                file.write(f"{node}:{self.model.get_children(node)}\n")