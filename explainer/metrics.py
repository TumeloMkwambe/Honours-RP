import copy
import wandb
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def draw_network(model) -> None:

    '''
    Draws network.
    '''

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


def metrics(ground_bn, explainer_bn, target_node):
    
    ground_mb = markov_blanket(ground_bn, target_node)
    explainer_mb = markov_blanket(explainer_bn, target_node)
    
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


def local_fidelity(instance, training_data, feature_set, model, n_trials):

    og_distro = model.predict_proba(instance.reshape(1, -1))[0]

    feature_indices = [feature_names.index(feature) for feature in feature_set]

    feature_values = [np.unique(training_data[:, feature_index]) for feature_index in feature_indices]

    distributions = []

    for i in range(n_trials):

        perturbed_instance = instance.copy()
        
        perturbed_instance[feature_indices] = [np.random.choice(values) for values in feature_values]

        new_distro = model.predict_proba(perturbed_instance.reshape(1, -1))[0]

        distributions.append(new_distro)

    return np.array(distributions), og_distro


def fidelity_plot(distributions, og_distro, title):

    wandb.init(entity = "mlp-e", project = "local-fidelity")
    
    fig = go.Figure()
    
    for distro in distributions:
    
        fig.add_trace(go.Scatter(
            x = np.arange(distributions.shape[1]),
            y = distro,
            mode = 'lines',
            line = dict(color = 'rgba(150, 150, 150, 0.5)'),
            showlegend = False
        ))
    
    fig.add_trace(go.Scatter(
        x = np.arange(distributions.shape[1]),
        y = og_distro,
        mode = 'lines+markers',
        name = 'Original Distribution',
        line = dict(color = 'rgba(0, 200, 55, 1.0)', width = 3)
    ))
    
    fig.update_layout(
        title = 'Perturbed Instance vs Original Instance Prediction Distributions',
        xaxis_title = 'Class Index',
        yaxis_title = 'Predicted Probability',
        template = 'plotly_dark'
    )
    
    wandb.log({f'{title} Local Fidelity Distributions': fig})

