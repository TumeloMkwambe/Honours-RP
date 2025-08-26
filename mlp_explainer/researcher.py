import os

class Researcher:
    '''
    The Researcher serves the sole purpose of guiding the experiment. It performs checks to assess if data for an experiment exists, 
    and notifies programmer of the stage of the experiment. Returns results of experiment if experiment is complete.

    Args:
        identifier (str): identifier of data in storage, identifies data for datasets, models, bayesian network structure information, and metrics information.
    '''
    
    
    def __init__(self, identifier : str):
        self.identifier = identifier

    def __checkSynData(self) -> None:

        '''
        __checkSynData checks if the SyntheticData stage of an experiment is complete by checking for the files corresponding with given identifier.
        '''

        dataset_path = os.path.join("data", f"{self.identifier}_dataset.csv")
        structure_path = os.path.join("structures", f"{self.identifier}_ground.txt")
        condition = os.path.isfile(dataset_path) and os.path.isfile(structure_path)

        if condition:
            return True
        
        raise FileNotFoundError(f'Dataset and structure information not in storage for {self.identifier} identifier... \n Consult SyntheticData module for creation')



    def init(self) -> None:

        self.__checkSynData()