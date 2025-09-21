import copy
import numpy as np
import networkx as nx
from pgmpy.metrics import SHD
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from networkx.drawing.nx_agraph import graphviz_layout

def draw_network(model) -> None:

    '''
    Draws network.
    '''

    DAG = nx.DiGraph()
    DAG.add_nodes_from(model.nodes())
    DAG.add_edges_from(model.edges())

    pos = graphviz_layout(DAG, prog="dot")
        
    nx.draw(
        DAG,
        pos,
        with_labels = True,
        node_size = 2000,
        node_color = "lightblue",
        arrowsize = 20,
        font_size = 12,
        font_weight = "bold"
    )

    plt.show()

def markov_blanket(model, target):
    
    model_ = copy.deepcopy(model)
    blanket = model_.get_markov_blanket(target)

    for node in list(model_.nodes()):
        if node not in blanket and node != target:
            model_.remove_node(node)

    return model_

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

def metrics(ground_bn, explainer_bn, target_node):

    '''
    Notes: cannot adapt SHD to Markov blankets only due to different nodes in blankets.
    '''
    
    ground_mb = markov_blanket(ground_bn, target_node)
    explainer_mb = markov_blanket(explainer_bn, target_node)
    
    intersection = len(np.intersect1d(ground_mb, explainer_mb, assume_unique = False, return_indices = False))
    union = len(np.union1d(ground_mb, explainer_mb))
    
    print(f'Ground Markov Blanket: {ground_mb.get_markov_blanket(target_node)}')
    print(f'Explainer Markov Blanket: {explainer_mb.get_markov_blanket(target_node)} \n')

    print(f'Markov Blanket Accuracy: {intersection} / {union}')

    print(f'Ground Markov Blanket: \n')
    draw_network(ground_mb)

    print(f'Explainer Markov Blanket: \n')
    draw_network(explainer_mb)
