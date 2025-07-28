import networkx as nx
import numpy as np
import community.community_louvain as community_louvain  # Assuming you have python-louvain installed

def abs_modularity(G, weight_attr='weight'):
    G_abs = G.copy()
    for u, v, d in G_abs.edges(data=True):
        d[weight_attr] = abs(float(d.get(weight_attr, 1)))

    edges_to_remove = [(u, v) for u, v, d in G_abs.edges(data=True) if d[weight_attr] == 0]
    G_abs.remove_edges_from(edges_to_remove)
    if G_abs.number_of_edges() == 0:
        return 0.0
    partition = community_louvain.best_partition(G_abs, weight=weight_attr)
    return community_louvain.modularity(partition, G_abs, weight=weight_attr)


def calculate_natural_connectivity(G):
    if len(G) < 2:
        return 0.0
    try:
        eigenvalues = nx.adjacency_spectrum(G, weight=None)
        scaled_sum = sum(np.exp(e.real) for e in eigenvalues) / len(G)
        return float(np.log(scaled_sum)) if scaled_sum > 0 else 0.0
    except:
        return 0.0

def calculate_global_efficiency(G):
    if len(G) < 2:
        return 0.0
    return nx.global_efficiency(G)

def get_lcc_size(G):
    if not G:
        return 0
    components = list(nx.connected_components(G))
    return len(max(components, key=len)) if components else 0

def normalize_dict(d):
    if not d or all(v == 0 for v in d.values()):
        return {k: 0 for k in d}
    vals = np.array(list(d.values()))
    mean = vals.mean()
    std = vals.std() if vals.std() > 0 else 1
    return {k: (v - mean) / std for k, v in d.items()}

def use_absolute_weights(G):
    G_abs = G.copy()
    for u, v, data in G_abs.edges(data=True):
        if 'weight' in data:
            data['weight'] = abs(data['weight'])
    return G_abs
