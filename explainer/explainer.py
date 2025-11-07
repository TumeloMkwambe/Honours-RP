import numpy as np
import pandas as pd
from explainer.metrics import markov_blanket
from pgmpy.estimators import PC, HillClimbSearch, ExpertKnowledge, GES

class Explainer:
    
    def __init__(self, model, training_data, feature_names, target_name, n_samples = 100, rep_prob = 0.5):

        self.model = model
        self.training_data = training_data
        self.feature_names = feature_names
        self.target_name = target_name
        self.n_samples = n_samples
        self.rep_prob = rep_prob

        self.local_data = None
        
        self.pc_blanket = None
        self.pc_mb = None
        
        self.bic_blanket = None
        self.bic_mb = None
        
    def data_generation(self, instance):
        
        instance_prediction = np.argmax(self.model.predict(instance.reshape(1, -1)).squeeze(0))
        
        random_indices = np.random.randint(len(self.training_data), size = self.n_samples)
        random_datapoints = self.training_data[random_indices]
        
        replacement_mask = np.random.rand(self.n_samples, random_datapoints.shape[1]) < self.rep_prob

        samples = np.where(replacement_mask, instance, random_datapoints)
                
        sample_predictions = np.argmax(self.model.predict(samples), axis = 1)
                
        sample_prediction_masks = sample_predictions != instance_prediction
        
        sample_masks = samples != instance
        
        masked_samples = sample_masks.astype(str)
        masked_sample_predictions = sample_prediction_masks.astype(str)
        
        self.local_data = pd.DataFrame(masked_samples, columns = self.feature_names)
        self.local_data[self.target_name] = masked_sample_predictions
    
    def get_structures(self):

        data = self.local_data
        
        #pc_est = PC(data) # constraint-based
        bic_est = HillClimbSearch(data) # score-based
        
        #pc_dag = pc_est.estimate(ci_test = "chi_square", return_type = "dag")
        bic_dag = bic_est.estimate(scoring_method = "bic-d")

        #self.pc_mb = markov_blanket(pc_dag, self.target_name)
        self.bic_mb = markov_blanket(bic_dag, self.target_name)

        #self.pc_blanket = self.pc_mb.get_markov_blanket(self.target_name)
        self.bic_blanket = self.bic_mb.get_markov_blanket(self.target_name)
        
    def log_data(self, instance):

        self.data_generation(instance)