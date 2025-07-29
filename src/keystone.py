import pickle
import pandas as pd
import networkx as nx
import re
from pathlib import Path
from collections import defaultdict
from src.utils import normalize_dict, use_absolute_weights
import community as community_louvain

def load_taxonomy_and_data(raw_dir: Path):
    GTDB_species_id_name_dict_file = raw_dir / 'GTDB_species_id_name_dict.pkl'
    bac_taxonomy_file = raw_dir / 'bac120_taxonomy_r207.tsv'

    with open(GTDB_species_id_name_dict_file, 'rb') as f:
        GTDB_species_id_name_dict = pickle.load(f)
    GTDB_species_id_name_dict = {str(k): v for k, v in GTDB_species_id_name_dict.items()}

    species_genus_dict = {}
    species_family_dict = {}
    species_order_dict = {}
    species_class_dict = {}

    bac_taxonomy = pd.read_csv(bac_taxonomy_file, sep='\t', names=['region', 'taxon'])
    bac_taxonomy_list = bac_taxonomy.taxon.tolist()
    for taxa in bac_taxonomy_list:
        pattern = '|'.join(map(re.escape, ['d__', ';p__', ';c__', ';o__', ';f__', ';g__', ';s__']))
        parts = re.split(pattern, taxa)
        species = parts[-1]
        genus = parts[-2]
        family = parts[-3]
        order = parts[-4]
        class_taxa = parts[-5]
        species_genus_dict[species] = genus
        species_family_dict[species] = family
        species_order_dict[species] = order
        species_class_dict[species] = class_taxa

    return GTDB_species_id_name_dict, species_genus_dict, species_family_dict, species_order_dict, species_class_dict

def get_keystone_taxa(graph, N=50):
    deg = nx.degree_centrality(graph)
    clo = nx.closeness_centrality(graph)
    bet = nx.betweenness_centrality(graph)
    clu = nx.clustering(graph)

    deg_norm = normalize_dict(deg)
    clo_norm = normalize_dict(clo)
    bet_norm = normalize_dict(bet)
    clu_norm = normalize_dict(clu)

    influence_score = {
        node: deg_norm[node] + clo_norm[node] + clu_norm[node] - bet_norm[node]
        for node in graph.nodes()
    }

    top_nodes = sorted(influence_score, key=influence_score.get, reverse=True)[:N]
    return top_nodes

def compute_keystone(graph_dir: Path, days, groups, GTDB_dict, output_dir: Path, seed=242):
    keystone_path = output_dir / 'keystone.pkl'
    if keystone_path.exists():
        print(f"Loading existing keystone data from {keystone_path}")
        with open(keystone_path, 'rb') as file:
            keystone = pickle.load(file)
    else:
        print("Computing keystone taxa...")
        keystone = {}
        for day in days:
            keystone[day] = {}
            for group in groups :
                graph_path = graph_dir / f'{group}_{day}.graphml'
                if graph_path.exists():
                    G = nx.read_graphml(graph_path)
                    G = nx.relabel_nodes(G, GTDB_dict)
                    G_abs = use_absolute_weights(G)
                    community_louvain.best_partition(G_abs, weight='weight', random_state=seed)
                    keystone_nodes = get_keystone_taxa(G_abs, 50)
                    keystone[day][group] = keystone_nodes

        with open(keystone_path, 'wb') as file:
            pickle.dump(keystone, file)
            print(f"Keystone taxa are being saved to {keystone_path}")
    
    return keystone

def get_all_species(nested_dict):
    species = []
    for value in nested_dict.values():
        if isinstance(value, dict):
            species.extend(get_all_species(value))
        elif isinstance(value, list):
            species.extend(value)
    return list(set(species))

def prepare_keystone_groups(keystone):
    keystone_groups = defaultdict(list)
    for day in keystone:
        for group in keystone[day]:
            keystone_groups[group].extend(keystone[day][group])
    return dict(keystone_groups)
