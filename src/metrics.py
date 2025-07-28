import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils import abs_modularity, get_lcc_size

def single_network_metrics(graphml_path: Path, directed=False, weight_attr='weight'):
    G = nx.read_graphml(graphml_path)
    if not directed:
        G = G.to_undirected()
    print(G) 
    num_components = nx.number_connected_components(G)
    clustering_coeff = nx.average_clustering(G, weight=weight_attr)
    modularity = abs_modularity(G, weight_attr=weight_attr)
    
    num_edges = G.number_of_edges()
    num_nodes = G.number_of_nodes()
    
    num_positive_edges = sum(1 for u, v, d in G.edges(data=True) if float(d.get(weight_attr, 1)) > 0)
    positive_edge_pct = num_positive_edges / num_edges if num_edges > 0 else 0
    edge_density = nx.density(G)
    
    relative_lcc_size = get_lcc_size(G) / num_nodes if num_nodes > 0 else 0
    
    mean_degree = (num_edges * 2 / num_nodes) if num_nodes > 0 else 0
    A = nx.to_numpy_array(G, weight=weight_attr)
    eigenvalues = np.linalg.eigvalsh(A)
    natural_connectivity = np.log(np.mean(np.exp(eigenvalues))) if eigenvalues.size > 0 else 0
    
    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'mean_degree': mean_degree,
        'num_components': num_components,
        'relative_lcc_size': relative_lcc_size,
        'clustering_coefficient': clustering_coeff,
        'modularity': modularity,
        'positive_edge_percentage': positive_edge_pct,
        'edge_density': edge_density,
        'natural_connectivity': natural_connectivity
    }

def compute_network_metrics(graph_dir: Path, days, groups, output_dir: Path):
    metrics = {}
    file_path = output_dir / 'network_topology_metrics.xlsx'
    if file_path.exists():
        print(f"Network topology metrics were already computed and saved to {file_path}.")
    else:
        for day in days:
            metrics[day] = {}
            for group in groups:
                graphml_path = graph_dir / f'{group}_{day}.graphml'
                if graphml_path.exists():
                    metrics[day][group] = single_network_metrics(graphml_path)
        metrics_df = pd.DataFrame.from_dict({(day, group): metrics[day][group] for day in metrics for group in metrics[day]}, orient='index')
        metrics_df.to_excel(output_dir / 'network_topology_metrics.xlsx')
        print(f"Network topology metrics are being computed and saved to {file_path}.")
