import random
import numpy as np
from pgmpy.models import LinearGaussianBayesianNetwork

class SyntheticData:
    def __init__(self, num_datasets):
        '''
        Agrs:
            num_networks (int): denotes the number of datasets to create.
        '''
        
        self.num_datasets = num_datasets

        pass

    def forward_sampling(self, network):
        pass


    def make_datasets(self):

        '''
        Objective: creates a set of networks and uses forward sampling to create datasets representative of the networks.
        '''

        for i in range(self.num_datasets):

            num_nodes = random.randint(8, 64)
            
            edge_probability = random.random()
            
            model = LinearGaussianBayesianNetwork.get_random(n_nodes=num_nodes, edge_prob=edge_probability, latents=False)
            
            # sampling + data creation + model saving
            
            graphviz_ = model.to_graphviz()
            
            graphviz_.draw(f'model_{i}.png', prog='dot')