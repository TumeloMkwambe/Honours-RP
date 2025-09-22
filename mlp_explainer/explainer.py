import numpy as np
import pandas as pd
from pgmpy.estimators import PC, HillClimbSearch, GES

class MLPExplainer:
    def __init__(self, mlp):

        self.bn = None
        self.__mlp = mlp
        self.__X = None
        self.__Y = None

    def __dg_d(self, x, X_train, preprocessor, n = 10, prob = 0.5):

        samples_X = []
        samples_Y = []

        processed_x = preprocessor(x_prime.reshape(1, -1))
        y = self.__mlp.predict(processed_x)

        for i in range(n):
            
            # sample a random datapoint x_prime from the training set
            idx = np.random.randint(len(X_train))
            x_prime = X_train[idx]

            # replace random values in the sample with values from the original datapoint x
            mask = np.random.rand(len(x_prime)) < prob
            x_prime = np.where(mask, x, x_prime)

            # preprocess x_prime the same way as the training set and get a prediction
            processed_x_prime = preprocessor(x_prime.reshape(1, -1))
            y_prime = self.__mlp.predict(processed_x_prime)

            # model returns probabilities, so we have to find the class with the maximum
            # if prediction changed mask target value as 1 and 0 otherwise
            # mask values in x_prime that are different from x's as 1 and 0 otherwise
            y_prime = y_prime.argmax() == y.argmax()
            x_prime = x_prime == x

            # add masked sample and prediction to respective collections
            samples_X.append(x_prime.astype(int))
            samples_Y.append(y_prime.astype(int))

        self.__X = samples_X
        self.__Y = samples_Y

    def __vs(self):
        '''
        Variable selection.
        '''

    def __sl(self, x_cols, y_col) -> None:

        dataframe = pd.DataFrame(self.__X, columns = x_cols)
        dataframe[y_col] = self.__Y

        est = HillClimbSearch(data = dataframe)
        self.bn = est.estimate(scoring_method = "bic-g", max_indegree = 5, max_iter = int(1e4))

    def explain(self, x, X_train, preprocessor, x_cols, y_col, n, prob) -> None:

        if type(x[0]) == int:
            self.__dg_d(x, X_train, preprocessor, n, prob)

        # self.__vs()
        
        self.__sl(x_cols, y_col)
