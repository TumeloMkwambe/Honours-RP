import numpy as np
import pandas as pd
from itertools import combinations
from pgmpy.estimators import HillClimbSearch, BIC, CITests

class Explainer:
    def __init__(self, model):

        self.bn = None
        self.model = model
        self.data = None

    def __data_generation(self, x: np.ndarray, X_train: np.ndarray, preproc, x_cols: list, y_col: str, n: int = 10, prob: float = 0.5):

        '''
        Args:
            x: a single local datapoint forming the base reference from which we sample perturbed instances.
            X_train: an array containing data used to train the model prior to preprocessing.
            preproc: the function used to preprocess input data prior to a forward pass to the model.
            x_cols: list of names of features.
            y_col: name of target.
            n: number of samples to draw from base reference local datapoint using perturbation scheme.
            prob: probability of replacing a randomly drawn datapoint from the training data with values from the base reference local datapoint.
        '''

        X = []
        Y = []

        proc_x = preproc.transform(x.reshape(1, -1))
        y = self.model.predict(proc_x, verbose = 0)
        y = y.reshape(len(y[0]),)

        for i in range(n):

            # sample a random datapoint x_prime from the training set
            idx = np.random.randint(len(X_train))
            x_prime = X_train[idx]

            # replace random values in the sample with values from the original datapoint x
            mask = np.random.rand(len(x_prime)) < prob
            x_prime = np.where(mask, x, x_prime)

            # preprocess x_prime the same way as the training set and get a prediction
            proc_x_prime = preproc.transform(x_prime.reshape(1, -1))
            y_prime = self.model.predict(proc_x_prime, verbose = 0)
            y_prime = y_prime.reshape(len(y_prime[0]),)

            # model returns probabilities, so we have to find the class with the maximum
            y_prime = y_prime.argmax() == y.argmax() # 0 if target changed, else 1
            x_prime = x_prime == x # 0 if feature value changed, else 1

            # add masked sample and prediction to respective collections
            X.append(x_prime.astype(int))
            Y.append(y_prime.astype(int))

        X = [X[i].astype(str) for i in range(len(X))]
        Y = [Y[i].astype(str) for i in range(len(Y))]

        self.data = pd.DataFrame(X, columns = x_cols)
        self.data[y_col] = Y

    def __variable_selection(self, target: str):

        variables = list(self.data.columns)
        S = {v: set() for v in variables}

        for var1, var2 in combinations(variables, 2):
            test = CITests.chi_square(X = var1, Y = var2, Z = [], data = self.data, boolean = True, significance_level = 0.05)

            if not test:
                S[var1].add(var2)
                S[var2].add(var1)
        
        U = {target}
        
        for node in S[target]:
            U |= S[node]
        
        self.data = self.data[list(U)]

    def __structure_learning(self) -> None:

        score = BIC(self.data)
        est = HillClimbSearch(data = self.data)
        
        self.bn = est.estimate(scoring_method = score, max_indegree = 5, max_iter = int(1e3))

    def explain(self, x, X_train, preproc, x_cols, y_col, n, prob) -> None:

        x = np.asarray(x)

        self.__data_generation(x, X_train, preproc, x_cols, y_col, n, prob)
        self.__variable_selection(y_col)
        self.__structure_learning()