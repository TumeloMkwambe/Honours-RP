from typing import Dict

import os
import pickle
import random
import numpy as np
import pandas as pd
import networkx as nx
from pgmpy.sampling import BayesianModelSampling
from pgmpy.models import DiscreteBayesianNetwork, LinearGaussianBayesianNetwork


class SyntheticData:

    '''
    Objective: contains methods to create Bayesian Network and create synthetic data from network,
               also contains methods to save network structure information and dataset.
    '''
    
    def __init__(self, identifier, type_):

        self.identifier = identifier
        self.__type = type_
        self.dataset = []
        self.model = self.__create_network()

    def __create_network(self) -> LinearGaussianBayesianNetwork:

        '''
        Objective: creates a Linear Gaussian Bayesian Network with random structure and parameters.
        '''

        network_filename = os.path.join("networks", f"{self.identifier}_ground_network.pkl")

        num_nodes = random.randint(5, 25)
        edge_probability = random.uniform(0, 0.5) # plausible way to regulate in-degree per node
        
        if os.path.exists(network_filename):
            with open(network_filename, 'rb') as file:
                return pickle.load(file)
    
        elif self.__type == 'continuous':
            return LinearGaussianBayesianNetwork.get_random(n_nodes = num_nodes, edge_prob = edge_probability, latents = False)
        
        elif self.__type == 'discrete':
            return DiscreteBayesianNetwork.get_random(n_nodes = num_nodes, edge_prob = edge_probability, latents = False)
    
    def __forward_sample_d(self, N) -> Dict:
        
        inference = BayesianModelSampling(self.model)
        samples = inference.forward_sample(size = N)
        self.dataset = samples.to_numpy()

    def __forward_sample_c(self) -> Dict:

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

        if self.__type == 'continous':
            for n in range(N):
                sample = self.__forward_sample_c()
                datapoint = np.array([sample[key] for key in sample])
                self.dataset.append(datapoint)
        
        elif self.__type == 'discrete':
            self.__forward_sample_d(N)
    
    def save(self) -> None:
        
        '''
        Objetive: saves Linear Gaussian Bayesian Network model and dataset.
        '''

        data_filename = os.path.join("data", f"{self.identifier}_ground_data.csv")
        network_filename = os.path.join("networks", f"{self.identifier}_ground_network.pkl")

        if os.path.exists(data_filename):
            print(f"Ground-truth for {self.identifier} dataset already exists.")
        
        else:
            data_frame = pd.DataFrame(np.array(self.dataset), columns=list(self.model.nodes()))
            data_frame.to_csv(data_filename, index=False)

        if os.path.exists(network_filename):
            print(f"Ground-truth {self.identifier} ground truth network already exists.")
        
        else:
            with open(network_filename, "wb") as f:
                pickle.dump(self.model, f)


def get_target(model):
    target_node = None
    max_in_degree = -1
    leaf_nodes = list(model.get_leaves())
    in_degree_iterator = model.in_degree_iter()
    
    for node, in_degree in in_degree_iterator:
        if node in leaf_nodes and in_degree > max_in_degree:
            target_node = node
            max_in_degree = in_degree

    return target_node

def split_data(model, data):

    target_node = get_target(model)

    X_columns = [column for column in data.columns.tolist() if column != target_node]
    
    X = data[X_columns].values
    y = data[target_node].values
    
    return X, y, target_node

def sample_datapoints(X, n):
    
    num_rows = X.shape[0]
    random_indices = np.random.choice(num_rows, size = n, replace = False)
    explain_X = X[random_indices]

    return explain_X
