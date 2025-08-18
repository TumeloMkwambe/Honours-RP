import os
import random
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

        num_nodes = random.randint(8, 64)
            
        edge_probability = random.random()

        model = LinearGaussianBayesianNetwork.get_random(n_nodes=num_nodes, edge_prob=edge_probability, latents=False)

        return model

    def save_edges(self) -> None:
        
        '''
        Objective: saves edges of model in a text file under the structures directory.

        Args:
            model (LinearGaussianBayesianNetwork): model whose edges we intend to save.
            filename (str): name of the file where network edge information will be saved.
        '''

        filename = os.path.join("structures", f"{self.name}.txt")

        with open(filename, "w") as file:

            for parent, child in self.model.edges():
            
                file.write(f"{parent}:{child}\n")

    def create_dataset(self) -> None:

        '''
        Objective: uses forward sampling to create dataset representative of the network.
        '''

        pass