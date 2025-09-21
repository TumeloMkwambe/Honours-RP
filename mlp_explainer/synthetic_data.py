from typing import Dict

import os
import pickle
import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import LinearGaussianBayesianNetwork
from networkx.drawing.nx_agraph import graphviz_layout


class SyntheticData:

    '''
    Objective: contains methods to create Bayesian Network and create synthetic data from network,
               also contains methods to save network structure information and dataset.
    '''
    
    def __init__(self, identifier):

        self.identifier = identifier
        self.model = self.__create_network()
        self.dataset = [] 

    def __create_network(self) -> LinearGaussianBayesianNetwork:

        '''
        Objective: creates a Linear Gaussian Bayesian Network with random structure and parameters.
        '''

        network_filename = os.path.join("networks", f"{self.identifier}_ground_network.pkl")

        if os.path.exists(network_filename):
            with open(network_filename, 'rb') as file:
                return pickle.load(file)

        else:
            num_nodes = random.randint(5, 25)
            edge_probability = random.uniform(0, 0.5) # plausible way to regulate in-degree per node
            return LinearGaussianBayesianNetwork.get_random(n_nodes = num_nodes, edge_prob = edge_probability, latents = False)

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
