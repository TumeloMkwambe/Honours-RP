import numpy as np
from mlp_explainer.synthetic_data import SyntheticData


if __name__ == "__main__":
    print('Open!')

    training = SyntheticData("training")
    training.create_dataset()
    training.save()

    print(f'Nodes: {training.model.nodes()}')
    print(f'Edegs: {training.model.edges()}')

    print('Close.')