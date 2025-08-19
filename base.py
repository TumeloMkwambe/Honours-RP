import numpy as np
from bayes_explainer.synthetic_data import SyntheticData


if __name__ == "__main__":
    print('Here!')
    training = SyntheticData("training")
    training.create_dataset()
    print(f'DATASET: \n {training.dataset}')