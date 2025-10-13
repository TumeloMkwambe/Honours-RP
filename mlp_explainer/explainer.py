import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

class Explainer:
    def __init__(self, model, X, preprocessor, n_samples = 100, rep_prob = 0.5):

        '''
        Args:
            model: model
            X: dataframe of X data (training, testing or all)
            preprocessor: function used to preprocess data prior to model forward pass
            n_samples: number of samples to generate for a single prediction explanation
            rep_prob: probability at which each feature value of x datapoint should be replaced with a value from a sample
        '''

        self.model = model
        self.X = X.to_numpy()
        self.preprocessor = preprocessor
        self.x_cols = X.columns
        self.y_col = 'target'
        self.n_samples = n_samples
        self.rep_prob = rep_prob
        
        self.data = None
        self.patterns = []
        self.relevance_dict = {col: 0 for col in self.x_cols}

    def __init_structures(self):
        self.data = None
        self.patterns = []
        
    def __data_generation(self, x: np.ndarray):

        y = self.model.predict(self.preprocessor(x.reshape(1, -1)), verbose = 0).squeeze(0)

        samples_X = []
        samples_Y = []
        
        for i in range(self.n_samples):
            
            sample_x = self.X[np.random.randint(len(self.X))]
            sample_x = np.where(np.random.rand(len(sample_x)) < self.rep_prob, x, sample_x)

            sample_y = self.model.predict(self.preprocessor(sample_x.reshape(1, -1)), verbose = 0).squeeze(0)
            
            sample_y = sample_y.argmax() != y.argmax()
            sample_x = sample_x != x
            
            samples_X.append(sample_x.astype(int))
            samples_Y.append(sample_y.astype(int))
            
        self.data = pd.DataFrame(samples_X, columns = self.x_cols)
        self.data[self.y_col] = samples_Y

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
            
            for feature in features:
                self.relevance_dict[feature] += h_mean / n_features

    def forward(self, x) -> None:

        self.__init_structures()
        
        self.__data_generation(x)
        
        for class_ in [0, 1]:
            
            class_data = self.data.loc[self.data[self.y_col] == class_]
            class_data = class_data.drop(self.y_col, axis = 1)
            self.fp_growth(class_data, class_)
        
        self.__relevance_rank()