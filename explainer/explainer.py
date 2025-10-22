import itertools
import numpy as np
import pandas as pd
from pgmpy.base import PDAG
from pgmpy.estimators import PC, HillClimbSearch, ExpertKnowledge
from mlxtend.frequent_patterns import fpgrowth, association_rules

class Explainer:
    
    def __init__(self, model, X, target, preprocessor, n_samples = 100, rep_prob = 0.5):
        
        self.X = X.to_numpy()
        
        self.x_cols = X.columns
        self.y_col = target
        
        self.model = model
        self.preprocessor = preprocessor
        
        self.n_samples = n_samples
        self.rep_prob = rep_prob
        
        
        self.associations = None
        self.data = None
        
        self.dag = None
        self.pdag = None

        self.relevance_dict = {col: 0 for col in self.x_cols}

    def init_structures(self):

        self.associations = None
        self.data = None
        
        self.dag = None
        self.pdag = None
        
    def data_generation(self, x):
        
        y = self.model.predict(self.preprocessor(x.reshape(1, -1)), verbose = 0).squeeze(0)
        y_argmax = y.argmax()
        
        random_indices = np.random.randint(len(self.X), size = self.n_samples)
        samples_X_base = self.X[random_indices]
        
        replace_mask = np.random.rand(self.n_samples, samples_X_base.shape[1]) < self.rep_prob
        
        samples_X_generated = np.where(replace_mask, x, samples_X_base)
        
        preprocessed_batch = self.preprocessor(samples_X_generated)
        
        samples_Y_raw = self.model.predict(preprocessed_batch, verbose = 0)
        
        samples_Y_argmax = samples_Y_raw.argmax(axis = 1)
        
        samples_Y_bool = samples_Y_argmax != y_argmax
        
        samples_X_bool = samples_X_generated != x
        
        samples_X_int = samples_X_bool.astype(int)
        samples_Y_int = samples_Y_bool.astype(int)
        
        self.data = pd.DataFrame(samples_X_int, columns = self.x_cols)
        self.data[self.y_col] = samples_Y_int

    def fp_growth(self, min_support, min_threshold):
        
        data = self.data.astype(bool)
        
        patterns = fpgrowth(data, min_support = min_support, use_colnames = True)
        
        self.associations = association_rules(patterns, metric = "confidence", min_threshold = min_threshold)

    def statistical_relevance(self):
        
        associations = self.associations[self.associations['consequents'] == frozenset({self.y_col})]
        
        associations = associations.sort_values(by = 'confidence', ascending = False).reset_index(drop = True)
        
        redundant = set()
        '''
        for i in range(len(associations)):
            
            if i in redundant:
                continue
            
            super_ant = associations.loc[i, 'antecedents']
            super_conf = associations.loc[i, 'confidence']
            
            for j in range(len(associations)):
                
                if i == j or j in redundant:
                    continue
                
                sub_ant = associations.loc[j, 'antecedents']
                sub_conf = associations.loc[j, 'confidence']
                
                if sub_ant.issubset(super_ant) and sub_ant != super_ant and sub_conf >= super_conf:
                    
                    redundant.add(i)
                    break
        '''
        associations = associations.drop(index = list(redundant)).reset_index(drop = True)
        
        self.associations = associations[['antecedents', 'consequents', 'support', 'confidence']]

    def relevance_ranking(self):

        for i, row in self.associations.iterrows():
            
            features = tuple(row['antecedents'])
            
            confidence = row['confidence']
            
            n_features = len(features)
                    
            for feature in features:
                
                self.relevance_dict[feature] += confidence / n_features
        
    '''
    def structures(self):
        
        nodes = set()
        edges = set()
        
        for i in range(len(self.associations)):
            
            ant_nodes = list(self.associations.loc[i, 'antecedents'])
            ant_edges = list(itertools.combinations(ant_nodes + [self.y_col], 2))
            
            nodes.update(ant_nodes)
            edges.update(ant_edges)
        
        self.pdag = PDAG(undirected_ebunch = list(edges))
    '''
        
    def explain(self, x, min_support, min_threshold):

        self.init_structures()
        self.data_generation(x)
        self.fp_growth(min_support, min_threshold)
        self.statistical_relevance()
        self.relevance_ranking()
        #self.structures()
        self.relevance_dict = dict(sorted(self.relevance_dict.items(), key = lambda item: item[1], reverse = True))
        
        return self.relevance_dict