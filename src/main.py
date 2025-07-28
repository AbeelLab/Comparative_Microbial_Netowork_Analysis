from pathlib import Path
import networkx as nx
import sys
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.metrics import compute_network_metrics
from src.plotting import plot_degree_distribution, plot_keystone_venn, plot_upset_keystone
from src.robustness import analyze_graphs
from src.keystone import load_taxonomy_and_data, compute_keystone, prepare_keystone_groups
from src.rf import load_data, run_nested_cv

ROOT_DIR = Path(__file__).parent.parent
GRAPH_DIR = ROOT_DIR / 'data' / 'processed' / 'graphs'
RAW_DIR = ROOT_DIR / 'data' / 'raw'
OUTPUT_DIR = ROOT_DIR / 'output'
FIGURES_DIR = ROOT_DIR / 'figures'

DAYS = ['14', '21', '35']
GROUPS = ['CTR', 'PFA', 'AGP']

def load_graphs(days=DAYS, groups=GROUPS):
    graphs_by_day = {}
    for day in days:
        graphs = {}
        for group in GROUPS:
            path = GRAPH_DIR / f'{group}_{day}.graphml'
            if path.exists():
                graphs[group] = nx.read_graphml(path)
        if graphs:
            graphs_by_day[day] = graphs
    return graphs_by_day

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    
    # Network Metrics Comparison
    print("Step 1: Computing network metrics...")
    compute_network_metrics(GRAPH_DIR, DAYS, GROUPS, OUTPUT_DIR)
    
    # Degree Distribution Plots
    print("Step 2: Generating degree distribution plots...")
    plot_degree_distribution(GRAPH_DIR, DAYS, GROUPS, FIGURES_DIR)
    
    # Robustness Analysis (computes data and plots for each day)
    print("Step 3: Performing robustness analysis...")
    graphs_by_day = load_graphs()
    for day, graphs in graphs_by_day.items():
        analyze_graphs(
            f'Day {day}',
            graphs=graphs,
            num_random_runs=100,
            save_data_path=OUTPUT_DIR / f'day{day}_node_removal.pkl',
            save_figure_path=FIGURES_DIR / f'day{day}_node_removal.png',
            day = day
        )
    
    # Keystone Taxa
    print("Step 4: Computing keystone taxa...")
    GTDB_dict, species_genus_dict, species_family_dict, species_order_dict, species_class_dict = load_taxonomy_and_data(RAW_DIR)
    keystone = compute_keystone(GRAPH_DIR, DAYS, GROUPS,  GTDB_dict, OUTPUT_DIR)
    print("Step 5: Generating keystone Venn diagrams and Upset Plot")
    plot_keystone_venn(keystone, DAYS, FIGURES_DIR)
    keystone_groups = prepare_keystone_groups(keystone)
    plot_upset_keystone(keystone_groups, species_genus_dict, species_family_dict, species_order_dict, species_class_dict, output_dir=OUTPUT_DIR, figures_dir=FIGURES_DIR) 
    
    # RF Classification based on keystone
    print("Step 6: Running RF classification based on keystone taxa abundance")
    X, y_encoded, le, feature_names = load_data(RAW_DIR, OUTPUT_DIR)
    run_nested_cv(X, y_encoded, le, FIGURES_DIR, OUTPUT_DIR)

if __name__ == "__main__":
    main()
