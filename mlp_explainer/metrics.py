import numpy as np
import networkx as nx
from sklearn.metrics import f1_score

def get_f1_score(estimated_model, true_model):
    
    nodes = estimated_model.nodes()
    
    est_adj = nx.to_numpy_array(
        estimated_model.to_undirected(), nodelist = nodes, weight = None
    )
    true_adj = nx.to_numpy_array(
        true_model.to_undirected(), nodelist = nodes, weight = None
    )

    f1 = f1_score(np.ravel(true_adj), np.ravel(est_adj))
    print("F1-score for the model skeleton: ", f1)