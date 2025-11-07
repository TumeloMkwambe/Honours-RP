from typing import Dict

import os
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
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

    def __create_network(self):

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


    def create_dataset(self, n) -> None:

        '''
        Objective: uses forward sampling to create dataset representative of the network.
        '''

        self.dataset = self.model.simulate(n)
    
    def get_target(self) -> None:

        target = None
        score = 0
        
        for candidate_target in list(self.model.nodes()):
            X = self.dataset.drop(columns = [candidate_target])
            y = self.dataset[candidate_target]
            new_score = mutual_info_classif(X, y).sum()
            if new_score > score:
                target = candidate_target
                score = new_score
        
        return target
    
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
