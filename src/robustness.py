# Modified robustness.py
import os
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm
from pathlib import Path
from src.utils import calculate_natural_connectivity, calculate_global_efficiency, get_lcc_size, normalize_dict

def calculate_influence_score(G):
    deg = normalize_dict(nx.degree_centrality(G))
    clo = normalize_dict(nx.closeness_centrality(G))
    bet = normalize_dict(nx.betweenness_centrality(G))
    return {node: deg[node] + clo[node] - bet[node] for node in G.nodes()}

def _single_robustness_run(G, removal_fraction=1.0, strategy='forward', seed=None):
    np.random.seed(seed)
    G_copy = G.copy()
    original_size = len(G_copy)
    original_edges = G_copy.number_of_edges()
    initial_nat_conn = calculate_natural_connectivity(G_copy)
    nodes_to_remove = int(original_size * removal_fraction)

    importance = calculate_influence_score(G_copy) if strategy == 'influence' else {
        node: random.random() for node in G_copy.nodes()
    }
    sorted_nodes = sorted(importance.items(), key=lambda x: x[1], reverse=(strategy == 'influence'))
    if strategy != 'influence':
        random.shuffle(sorted_nodes)

    lcc_sizes = [get_lcc_size(G_copy)]
    nat_conns = [1.0 if initial_nat_conn > 0 else 0.0]
    edge_fractions = [1.0]
    removed_fractions = [0]

    step_size = max(1, nodes_to_remove // 5)
    for i in range(0, nodes_to_remove, step_size):
        batch_end = min(i + step_size, nodes_to_remove)
        nodes_to_remove_batch = [n for n, _ in sorted_nodes[i:batch_end] if n in G_copy]
        G_copy.remove_nodes_from(nodes_to_remove_batch)

        lcc_sizes.append(get_lcc_size(G_copy))
        current_nat_conn = calculate_natural_connectivity(G_copy)
        nat_conns.append(current_nat_conn / initial_nat_conn if initial_nat_conn > 0 else 0.0)
        current_edges = G_copy.number_of_edges()
        edge_fractions.append(current_edges / original_edges if original_edges > 0 else 0.0)
        removed_fractions.append(batch_end / original_size)

    return removed_fractions, nat_conns, edge_fractions, lcc_sizes

def robustness_analysis(G, removal_fraction=0.7, strategy='influence', num_runs=1, day=None, group=None):
    max_workers = min(os.cpu_count() or 4, 8)
    seeds = [random.randint(0, 1000000) for _ in range(num_runs)]
    tasks = [(G, removal_fraction, strategy, seed) for seed in seeds]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        all_results = list(tqdm(executor.map(lambda args: _single_robustness_run(*args), tasks), 
                                total=num_runs, 
                                desc=f"Running {strategy} strategy for {group} on day {day}",
                                leave=False))
    
    return all_results

def compute_robustness_data(graphs, strategies=['influence', 'random'], num_random_runs=100, 
                            removal_fraction=1.0, save_path: Path = None, day=None):
    results = {}
    for graph_name, G in graphs.items():
        print(f"Processing {graph_name} (Nodes: {len(G)}, Edges: {G.number_of_edges()})")
        results[graph_name] = {}
        for strategy in tqdm(strategies, leave=False):
            num_runs = num_random_runs if strategy == 'random' else 1
            results[graph_name][strategy] = robustness_analysis(
                G, removal_fraction, strategy, num_runs, day=day, group=graph_name
            )

    if save_path:
        print(f"Saving results to {save_path}")
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
    return results

def load_robustness_data(file_path: Path):
    print(f"Loading results from {file_path}")
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def analyze_graphs(title, graphs, strategies=['influence', 'random'], num_random_runs=100, 
                   removal_fraction=1.0, show_properties=True, save_data_path: Path = None, 
                   save_figure_path: Path = None, day=None):
    if save_data_path and save_data_path.exists():
        results = load_robustness_data(save_data_path)
        graph_sizes = {name: get_lcc_size(graphs[name]) for name in graphs}
    else:
        print(f"--- Day {day} ---")
        if show_properties:
            for name, G in graphs.items():
                print(f"{name}: Nodes={len(G)}, Edges={G.number_of_edges()}, "
                      f"Avg Degree={2 * G.number_of_edges() / len(G):.2f}, "
                      f"Global Eff={calculate_global_efficiency(G):.4f}, "
                      f"Nat Conn={calculate_natural_connectivity(G):.4f}")
        
        results = compute_robustness_data(graphs, strategies, num_random_runs, removal_fraction, save_data_path, day=day)
        graph_sizes = {name: len(G) for name, G in graphs.items()}
    
    from src.plotting import plot_robustness_comparison
    plot_robustness_comparison(title, list(graphs.keys()), strategies, results, graph_sizes, save_figure_path)
    return results
