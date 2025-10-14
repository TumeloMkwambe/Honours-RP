import numpy as np
import pandas as pd
from pgmpy.estimators import PC
from mlxtend.frequent_patterns import fpgrowth, association_rules

class Explainer:
    
    def __init__(self, model, X, preprocessor, n_samples = 100, rep_prob = 0.5):

        self.model = model
        self.X = X.to_numpy()
        self.preprocessor = preprocessor
        self.x_cols = X.columns
        self.y_col = 'target'
        self.n_samples = n_samples
        self.rep_prob = rep_prob

        self.bn = None
        self.data = None
        self.structure_data = None
        self.patterns = []
        self.count = 0
        self.relevance_dict = {col: 0 for col in self.x_cols}

    def __init_structures(self):
        
        self.data = None
        self.patterns = []
        
    def __data_generation(self, x: np.ndarray):
        
        y = self.model.predict(self.preprocessor(x.reshape(1, -1)), verbose=0).squeeze(0)
        y_argmax = y.argmax()
        
        random_indices = np.random.randint(len(self.X), size = self.n_samples)
        samples_X_base = self.X[random_indices]
        
        replace_mask = np.random.rand(self.n_samples, samples_X_base.shape[1]) < self.rep_prob
        
        samples_X_generated = np.where(replace_mask, x, samples_X_base)
        
        preprocessed_batch = self.preprocessor(samples_X_generated)
        
        samples_Y_raw = self.model.predict(preprocessed_batch, verbose=0)
        
        samples_Y_argmax = samples_Y_raw.argmax(axis=1)
        
        samples_Y_bool = samples_Y_argmax != y_argmax
        
        samples_X_bool = samples_X_generated != x
        
        samples_X_int = samples_X_bool.astype(int)
        samples_Y_int = samples_Y_bool.astype(int)
        
        self.data = pd.DataFrame(samples_X_int, columns = self.x_cols)
        self.data[self.y_col] = samples_Y_int
        
        self.structure_data = pd.concat([self.structure_data, self.data], ignore_index = True)

    def fp_growth(self, data, class_):
        
        if class_ == 0:
            data = 1 - data
        
        data = data.astype(bool)
        
        class_patterns = fpgrowth(data, min_support = 0.3, use_colnames = True)
        
        self.patterns.append(class_patterns)

    def __harmonic_merge(self):
        
        self.patterns[0] = self.patterns[0].rename(columns = {'support': 'support_stable'})
        self.patterns[1] = self.patterns[1].rename(columns = {'support': 'support_unstable'})
        
        patterns_merged = pd.merge(
            self.patterns[0],
            self.patterns[1],
            on = 'itemsets',
            how = 'outer'
        )
        
        patterns_merged = patterns_merged.fillna(0)
        
        support_0 = patterns_merged['support_stable']
        support_1 = patterns_merged['support_unstable']
        
        denominator = support_0 + support_1
        patterns_merged['Harmonic Mean'] = (2 * support_0 * support_1 / denominator).mask(denominator == 0, 0)
        
        harmonic_rank = patterns_merged[['itemsets', 'support_stable', 'support_unstable', 'Harmonic Mean']].sort_values(
            by = 'Harmonic Mean', 
            ascending = False
        )
        
        return harmonic_rank[harmonic_rank['Harmonic Mean'] > 0.0]

    def __relevance_rank(self):
        
        rank = self.__harmonic_merge()
        
        for row, idx in rank.iterrows():
            
            features = tuple(idx['itemsets'])
            h_mean = idx['Harmonic Mean']
            n_features = len(features)
            
            if n_features == 1:
                for feature in features:
                    self.relevance_dict[feature] += h_mean

    def __structure_learning(self, data):

        est = PC(data = data)
        
        self.bn = est.estimate(ci_test = "chi_square", return_type = 'dag')

    def setup(self, x) -> None:

        self.count += 1
        self.__init_structures()
        self.__data_generation(x)
        
        for class_ in [0, 1]:
            
            class_data = self.data.loc[self.data[self.y_col] == class_]
            class_data = class_data.drop(self.y_col, axis = 1)
            self.fp_growth(class_data, class_)
        
        self.__relevance_rank()

    def output(self, threshold):

        self.relevance_dict = {key: self.relevance_dict[key] / self.count for key in self.relevance_dict}
        self.relevance_dict = {key: self.relevance_dict[key] for key in self.relevance_dict if self.relevance_dict[key] > threshold}
        data = self.structure_data[list(self.relevance_dict) + [self.y_col]].astype(str)
        self.relevance_dict = dict(sorted(self.relevance_dict.items(), key = lambda item: item[1], reverse = True))
        self.__structure_learning(data)