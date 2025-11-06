import copy
import wandb
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.spatial.distance import jensenshannon

def draw_network(model):

    DAG = nx.DiGraph()
    DAG.add_nodes_from(model.nodes())
    DAG.add_edges_from(model.edges())

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


def markov_blanket(model, target):
    
    model_ = copy.deepcopy(model)
    blanket = model_.get_markov_blanket(target)

    for node in list(model_.nodes()):
        if node not in blanket and node != target:
            model_.remove_node(node)

    return model_


def metrics(ground_mb, explainer_mb, target_node):
    
    ground_features = ground_mb.get_markov_blanket(target_node)
    explainer_features = explainer_mb.get_markov_blanket(target_node)
    
    intersection = len(np.intersect1d(ground_features, explainer_features, assume_unique = False, return_indices = False))
    union = len(np.union1d(ground_features, explainer_features))
    
    print(f'Ground Markov Blanket: {ground_features} \n')
    print(f'Explainer Markov Blanket: {explainer_features} \n')
    
    print(f'Markov Blanket Accuracy: {intersection} / {union}')
    
    print(f'Ground Markov Blanket: \n')
    draw_network(ground_mb)
    
    print(f'Explainer Markov Blanket: \n')
    draw_network(explainer_mb)


def in_distribution(instance, training_data, indices, radius = 0):

    diffs = np.sum(training_data[:, indices] != instance[indices], axis = 1)

    mask = diffs <= radius

    if not np.any(mask):

        return None

    candidates = np.unique(training_data[mask], axis = 0)
    
    return candidates


def distribution_drift(instance, training_data, feature_set, feature_names, model, n_trials):

    og_distro = model(instance.reshape(1, -1)).numpy()[0]

    feature_indices = [feature_names.index(feature) for feature in feature_set]

    candidates = None
    
    radius = 0

    while True:

        candidates = in_distribution(instance, training_data, feature_indices, radius)

        if candidates is not None and candidates.shape[0] >= n_trials:

            break

        radius += 1

        if radius > len(feature_indices) / 2:

            raise RuntimeError(f"Not enough matching in distribution candidates even up to radius {radius}, reduce n_trials.")

    candidates = candidates[:n_trials].copy()

    candidates[:, feature_indices] = instance[feature_indices][None, :]

    new_distros = model(candidates).numpy()

    return new_distros, og_distro


def average_distribution_drift(og_distro, new_distros):

    js_values = [jensenshannon(og_distro, new_d) ** 2 for new_d in new_distros]

    avg_js = np.mean(js_values)

    return avg_js, js_values


def fidelity_plot(new_distros, og_distro, title):
    
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

