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

    def draw_network(self) -> None:

        DAG = nx.DiGraph()
        DAG.add_nodes_from(self.model.nodes())
        DAG.add_edges_from(self.model.edges())

        pos = graphviz_layout(DAG, prog="dot")
        
        nx.draw(
            DAG,
            pos,
            with_labels = True,
            node_size = 2000,
            node_color = "lightblue",
            arrowsize = 20,
            font_size = 12,
            font_weight = "bold"
        )

        structure_filename = os.path.join("networks", f"{self.identifier}_ground_network.png")
        plt.savefig(structure_filename, bbox_inches = "tight")
        plt.show()
    
    def save(self) -> None:
        
        '''
        Objetive: saves Linear Gaussian Bayesian Network model and dataset.
        '''

        data_filename = os.path.join("data", f"{self.identifier}_ground_data.csv")
        network_filename = os.path.join("networks", f"{self.identifier}_ground_network.pkl")

        if os.path.exists(data_filename):
            print(f"Aborting: {self.identifier} dataset already exists.")
        
        else:
            data_frame = pd.DataFrame(np.array(self.dataset), columns=list(self.model.nodes()))
            data_frame.to_csv(data_filename, index=False)

        if os.path.exists(network_filename):
            print(f"Aborting: {self.identifier} ground truth network already exists.")
        
        else:
            with open(network_filename, "wb") as f:
                pickle.dump(self.model, f)