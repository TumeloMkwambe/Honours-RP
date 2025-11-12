import copy
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

def draw_network(dag):

    '''
    Draws given dag.
    '''

    DAG = nx.DiGraph()
    DAG.add_nodes_from(dag.nodes())
    DAG.add_edges_from(dag.edges())

    pos = nx.drawing.nx_agraph.graphviz_layout(
        DAG, 
        prog = "dot", 
        args = '-Gsep=1.5 -Gnodesep=0.5 -Granksep=1.0 -Gsize="10,8!"'
    )

    plt.figure(figsize = (12, 8)) 

    nx.draw(
        DAG,
        pos,
        with_labels = True,
        node_size = 1500,
        node_color="skyblue",
        arrowsize = 5,
        font_size = 6
    )

    plt.show()


def markov_blanket(network, variable):

    '''
    Removes edges in network and returns structure only showing markov blanket of specified variable.
    '''
    
    network_ = copy.deepcopy(network)
    blanket = network_.get_markov_blanket(variable)

    for node in list(network_.nodes()):
        if node not in blanket and node != variable:
            network_.remove_node(node)

    return network_


def adjacency_matrix(markov_blanket, variables):

    '''
    Given a markov blanket structure, constructs an adjacency matrix over all variables showing edges.
    Symmetric, so edges are shown in the lower and upper triangular matrix.
    '''

    blanket_nodes = list(markov_blanket.nodes())

    nodes_to_idx = {node: variables.index(node) for node in blanket_nodes}

    blanket_edges = list(markov_blanket.edges())

    blanket_edges = [(nodes_to_idx[edge[0]], nodes_to_idx[edge[1]]) for edge in blanket_edges]

    matrix = np.zeros((len(variables), len(variables)))

    for edge in blanket_edges:

        matrix[edge[0]][edge[1]] = 1
        matrix[edge[1]][edge[0]] = 1

    return matrix


def structure_metrics(ground, estimate, variables):

    '''
    Metrics used to evaluate the method's ability to retrieve the ground-truth network structure influencing the model's decision making.
    '''

    ground_matrix = adjacency_matrix(ground, variables)
    estimate_matrix = adjacency_matrix(estimate, variables)
    
    involved = np.where((ground_matrix.sum(axis = 0) + ground_matrix.sum(axis = 1) + 
                         estimate_matrix.sum(axis = 0) + estimate_matrix.sum(axis = 1)
                        ) > 0)[0]
    
    ground_sub = ground_matrix[np.ix_(involved, involved)].copy()
    estimate_sub = estimate_matrix[np.ix_(involved, involved)].copy()
    
    np.fill_diagonal(ground_sub, 0)
    np.fill_diagonal(estimate_sub, 0)
    
    ground_sub = np.maximum(ground_sub, ground_sub.T)
    estimate_sub = np.maximum(estimate_sub, estimate_sub.T)
    
    precision = precision_score(ground_sub.flatten(), estimate_sub.flatten(), zero_division = 0)
    recall = recall_score(ground_sub.flatten(), estimate_sub.flatten(), zero_division = 0)
    f1 = f1_score(ground_sub.flatten(), estimate_sub.flatten(), zero_division = 0)
    accuracy = accuracy_score(ground_sub.flatten(), estimate_sub.flatten())
    confusion_matrix_ = confusion_matrix(ground_sub.flatten(), estimate_sub.flatten())

    return precision, recall, f1, accuracy, confusion_matrix_


def in_distribution(instance, training_data, indices, radius = 0):
    
    '''
    Looks for datapoints in the training dataset which have the same values for the frozen features (indicated by indices) as the instance.
    Radius allows for a certain number of features in the frozen set to be different, in case we cannot generate enough data to test the distribution drift.
    '''
    
    diffs = np.sum(training_data[:, indices] != instance[indices], axis = 1)
    
    mask = diffs <= radius
    
    if not np.any(mask):

        return None

    candidates = np.unique(training_data[mask], axis = 0)
    
    return candidates


def distribution_drift(instance, training_data, feature_set, feature_names, model, n_trials):

    '''
    Idea: freeze values of features in feature_set, sample datapoints with same values for frozen features but different unconstrained values
          for other features (in-distribution randomization), then obtain distribution over classes from model to get a sense of how much the frozen
          features anchor the prediction. Iteratively increases radius if we cannot generate required number of trials.
    '''

    og_distro = model(instance.reshape(1, -1)).numpy()[0]

    feature_indices = [feature_names.index(feature) for feature in feature_set]

    candidates = None
    
    radius = 0

    while True:

        candidates = in_distribution(instance, training_data, feature_indices, radius)

        if candidates is not None and candidates.shape[0] >= n_trials:

            break

        radius += 1

        '''
        if radius > len(feature_indices):

            raise RuntimeError(f"Not enough matching in distribution candidates even up to radius {radius}, reduce n_trials.")
        '''

    candidates = candidates[:n_trials].copy()

    candidates[:, feature_indices] = instance[feature_indices][None, :]

    new_distros = model(candidates).numpy()

    return new_distros, og_distro


def average_distribution_drift(og_distro, new_distros, eps = 1e-12):

    '''
    Quantifies the average distribution drift using the jensen-shannon divergence metric* (kl is not a metric).
    '''

    og_distro = np.asarray(og_distro, dtype = np.float64)
    og_distro = (og_distro + eps) / (og_distro.sum() + eps * len(og_distro))

    js_values = []
    
    for new_d in new_distros:

        new_d = np.asarray(new_d, dtype = np.float64)
        new_d = (new_d + eps) / (new_d.sum() + eps * len(new_d))
        
        js_values.append(jensenshannon(og_distro, new_d) ** 2)

    avg_js = np.mean(js_values)

    return avg_js, js_values


def divergence_plot(js_values, method_names):

    '''
    Box and whisker plots of jensen-shannon divergence values.
    '''

    data = pd.DataFrame(js_values.T, columns = [f'{name}' for name in method_names])
    data_melt = data.melt(var_name = 'Method', value_name = 'Jensen-Shannon Divergence')
    
    fig = px.box(data_melt, x = 'Method', y = 'Jensen-Shannon Divergence', title = 'Jensen-Shannon Divergence per Method')
    
    fig.show()


def fidelity_plot(new_distros, og_distro, title):

    '''
    Shows divergence of distributions over classes.
    '''
    
    fig = go.Figure()
    
    for distro in new_distros:
    
        fig.add_trace(go.Scatter(
            x = np.arange(new_distros.shape[1]),
            y = distro,
            mode = 'lines',
            line = dict(color = 'rgba(150, 150, 150, 0.5)'),
            showlegend = False
        ))
    
    fig.add_trace(go.Scatter(
        x = np.arange(new_distros.shape[1]),
        y = og_distro,
        mode = 'lines+markers',
        name = 'Original Distribution',
        line = dict(color = '#004aad', width = 3)
    ))
    
    fig.update_layout(
        title = f'{title} Class Distribution Drift',
        xaxis_title = 'Class Index',
        yaxis_title = 'Predicted Probability',
        template = 'simple_white'
    )

    fig.show()

