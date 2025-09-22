import numpy as np
import pandas as pd
from pgmpy.estimators import PC, HillClimbSearch, GES

class MLPExplainer:
    def __init__(self, mlp, target_node):
        self.__mlp = mlp
        self.target_node = target_node

        self.bn = None
        self.__X = None
        self.__Y = None

    def dg_c(self, X, explain_X, n_samples = 1000, replacement_prob = 0.5):
    
        '''
        emp_cov = np.cov(X, rowvar = False)
    
        explain_X = np.atleast_2d(explain_X)
        n_features = explain_X.shape[1]
    
        diff_X_all = []
        diff_Y_all = []
    
        for explain_x in explain_X:
    
            samples = np.random.multivariate_normal(mean = explain_x, cov = emp_cov, size = n_samples)
    
            mask = np.random.rand(n_samples, n_features) < replacement_prob
            samples = np.where(mask, explain_x, samples)
    
            proc_explain_x = preprocessor(explain_x.reshape(1, -1))
            explain_y = self.__mlp.predict(proc_explain_x)
    
            proc_samples = preprocessor(samples)
            samples_y = self.__mlp.predict(proc_samples)
    
            diff_X = explain_x - samples
            diff_Y = explain_y - samples_y.reshape(-1, 1)
    
            diff_X_all.append(diff_X)
            diff_Y_all.append(diff_Y)
    
        diff_X_all = np.vstack(diff_X_all)
        diff_Y_all = np.vstack(diff_Y_all)
    
        self.__X = diff_X_all
        self.__Y = diff_Y_all
        '''

    def dg_d(self, X, explain_x, N = 10, replacement_prob = 0.5):

        samples_X = []
        samples_Y = []

        column_values = [np.unique(X[:,i]) for i in range(len(X[0]))]

        explain_y = self.__mlp.predict(explain_x.reshape(1, -1))

        for i in range(N):
            
            sample = np.array([np.random.choice(column) for column in column_values])

            mask = np.random.rand(len(sample)) < replacement_prob
            sample = np.where(mask, explain_x, sample)

            pred = self.__mlp.predict(sample.reshape(1, -1))

            target_mask = pred.argmax() == explain_y.argmax()
            sample = sample == explain_x

            samples_X.append(sample.astype(int))
            samples_Y.append(target_mask.astype(int))

        self.__X = samples_X
        self.__Y = samples_Y

    def variable_selection(self):
        '''
        Variable selection.
        '''

    def structure_learning(self, x_columns):

        dataframe = pd.DataFrame(self.__X, columns = x_columns)
        dataframe[self.target_node] = self.__Y

        est = HillClimbSearch(data = dataframe)
        self.bn = est.estimate(scoring_method = "bic-g", max_indegree = 5, max_iter = int(1e4))
    
    def run(self, X, explain_x, x_columns, n_samples, replacement_prob):
        
        if type(explain_x[0]) == float:
            self.dg_c(X, explain_x, n_samples, replacement_prob)
        if type(explain_x[0]) == int:
            self.dg_d(X, explain_x, n_samples, replacement_prob)

        # self.variable_selection()
        
        self.structure_learning(x_columns)
