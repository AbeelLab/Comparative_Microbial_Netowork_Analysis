import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
from pathlib import Path
from matplotlib_venn import venn3
import numpy as np
from upsetplot import from_memberships, UpSet
import distinctipy
from collections import defaultdict


def plot_degree_distribution(graph_dir: Path, days, groups, figures_dir: Path):
    for day in days:
        degree_data = []
        for group in groups:
            graphml_path = graph_dir / f'{group}_{day}.graphml'
            if graphml_path.exists():
                G = nx.read_graphml(graphml_path)
                degrees = [d for n, d in G.degree()]
                df = pd.DataFrame({'degree': degrees, 'group': group})
                degree_data.append(df)
        
        if degree_data:
            combined_df = pd.concat(degree_data)
            
            plt.figure(figsize=(8, 6))
            sns.kdeplot(data=combined_df, x='degree', hue='group', 
                    hue_order=['CTR', 'PFA', 'AGP'], palette={'CTR':'blue', 'PFA':'gold', 'AGP':'red'}, common_norm=False)
            plt.title(f'Node degree distribution across treatment groups at day {day}')
            plt.xlim(0, 350)
            plt.ylim(0, 0.015)
            plt.xlabel('Node Degree')
            plt.ylabel('Density')
            plt.tight_layout()
            plt.savefig(figures_dir / f'node_degree_distribution_day{day}.png', dpi=300, bbox_inches='tight')
            plt.close()

def plot_robustness_comparison(day, graphs_keys, strategies, results, graph_sizes, save_path: Path = None, figsize=(18, 10), dpi=300):
    fig, axes = plt.subplots(3, len(graphs_keys), figsize=figsize)
    colors = {'influence': 'purple', 'random': 'black'}
    axes = np.array(axes).reshape(3, -1) if len(graphs_keys) > 1 else axes.reshape(3, 1)
    
    for col, graph_name in enumerate(graphs_keys):
        graph_results = results[graph_name]
        graph_size = graph_sizes[graph_name]
        
        ax_edge, ax_lcc, ax_nat = axes[:, col]
        
        for ax, ylabel, title in zip(
            [ax_edge, ax_lcc, ax_nat],
            ['Remaining Edge Fraction', 'LCC Relative Size', 'Normalized Natural Connectivity'],
            [f'{day} - {graph_name}' for _ in range(3)]
        ):
            ax.set_title(title)
            ax.set_xlabel('Fraction of Nodes Removed')
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, 1.1)

        for strategy in strategies:
            strategy_results = graph_results[strategy]
            
            if strategy == 'random' and len(strategy_results) > 1:
                all_fractions = [r[0] for r in strategy_results]
                all_lcc_sizes = np.array([r[3] for r in strategy_results])
                all_nat_conns = np.array([r[1] for r in strategy_results])
                all_edge_fractions = np.array([r[2] for r in strategy_results])
                
                min_length = min(len(f) for f in all_fractions)
                mean_fractions = all_fractions[0][:min_length]
                mean_lcc_sizes = np.mean(all_lcc_sizes[:, :min_length], axis=0)
                mean_nat_conns = np.mean(all_nat_conns[:, :min_length], axis=0)
                mean_edge_fractions = np.mean(all_edge_fractions[:, :min_length], axis=0)
                std_lcc_sizes = np.std(all_lcc_sizes[:, :min_length], axis=0)
                std_nat_conns = np.std(all_nat_conns[:, :min_length], axis=0)
                std_edge_fractions = np.std(all_edge_fractions[:, :min_length], axis=0)
                
                normalized_mean_lcc = mean_lcc_sizes / graph_size
                
                for ax, data, std in zip(
                    [ax_edge, ax_lcc, ax_nat],
                    [mean_edge_fractions, normalized_mean_lcc, mean_nat_conns],
                    [std_edge_fractions, std_lcc_sizes / graph_size, std_nat_conns]
                ):
                    ax.plot(mean_fractions, data, color=colors[strategy], linewidth=3,
                            label=f'{strategy.capitalize()} (n={len(strategy_results)})')
                    ax.fill_between(mean_fractions, data - std, data + std,
                                    color=colors[strategy], alpha=0.2)
            else:
                fractions, nat_conns, edge_fractions, lcc_sizes = strategy_results[0]
                normalized_lcc = np.array(lcc_sizes) / graph_size
                
                ax_edge.plot(fractions, edge_fractions, color=colors[strategy], linewidth=3, label=strategy.capitalize())
                ax_lcc.plot(fractions, normalized_lcc, color=colors[strategy], linewidth=3, label=strategy.capitalize())
                ax_nat.plot(fractions, nat_conns, color=colors[strategy], linewidth=3, label=strategy.capitalize())
        
        for ax in [ax_edge, ax_lcc, ax_nat]:
            ax.legend(loc='upper right')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()

def plot_keystone_venn(keystone, days, figures_dir: Path):
    fixed_labels = ['CTR', 'PFA', 'AGP']

    all_data = []
    all_labels = []
    all_sets = []

    for day in days:
        day_labels = list(keystone[day].keys())
        day_sets = [set(v) for v in keystone[day].values()]

        ordered_sets = []
        ordered_labels = []
        for label in fixed_labels:
            if label in day_labels:
                idx = day_labels.index(label)
                ordered_sets.append(day_sets[idx])
                ordered_labels.append(label)
            else:
                ordered_sets.append(set())  
                ordered_labels.append(label)

        all_labels.append(ordered_labels)
        all_sets.append(ordered_sets)
        all_data.append([len(s) for s in ordered_sets])

    max_area = max(sum(day_data) for day_data in all_data) if all_data else 1

    def set_venn_scale(ax, true_area, reference_area=max_area):
        s = np.sqrt(reference_area / true_area) if true_area > 0 else 1
        ax.set_xlim(-s, s)
        ax.set_ylim(-s, s)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, (day, labels, sets, ax) in enumerate(zip(days, all_labels, all_sets, axes.flatten())):
        venn3(sets, set_labels=labels, set_colors=('blue', 'gold', 'red'), ax=ax)
        ax.set_title(f"Day {day}")
        day_total = sum(len(s) for s in sets)
        set_venn_scale(ax, day_total)

    plt.tight_layout()
    plt.savefig(figures_dir / 'keystone_venn_comparison_scaled.png', dpi=300, bbox_inches='tight')
    plt.close()


def get_n_distinct_colors(n):
    return distinctipy.get_colors(n)

def plot_upset_keystone(keystone_groups, species_genus_dict, species_family_dict, species_order_dict, species_class_dict, output_dir: Path, figures_dir: Path, by='Order'):
    all_elements = set(keystone_groups['CTR'] + keystone_groups['PFA'] + keystone_groups['AGP'])
    
    memberships = []
    for element in all_elements:
        member_sets = tuple(key for key in keystone_groups if element in keystone_groups[key])
        memberships.append(member_sets)
    
    intersection_elements = defaultdict(set)
    genus_list = []
    family_list = []
    order_list = []
    class_list = []
    
    for element in all_elements:
        member_sets = tuple(sorted(key for key in keystone_groups if element in keystone_groups[key]))
        intersection_elements[member_sets].add(element)
        genus_list.append(species_genus_dict.get(element))
        family_list.append(species_family_dict.get(element))
        order_list.append(species_order_dict.get(element))
        class_list.append(species_class_dict.get(element))
    
    df = pd.DataFrame({
        "intersection": [" & ".join(sets) for sets in intersection_elements.keys()],
        "species": [", ".join(sorted(elems)) for elems in intersection_elements.values()]
    })
    df.to_excel(output_dir / 'upsetplot_intersection_species.xlsx', index=False)
    
    data = from_memberships(memberships)
    new_order = ['CTR', 'PFA', 'AGP']
    data = data.reorder_levels(new_order)
    data = data.to_frame()
    data['Genus'] = genus_list
    data['Family'] = family_list
    data['Order'] = order_list
    data['Class'] = class_list
    
    upset = UpSet(
        data,
        subset_size='count',
        show_counts=True,
        sort_categories_by=None,
        intersection_plot_elements=0
    )
    
    colors = get_n_distinct_colors(len(set(data[by])))
    upset.add_stacked_bars(by=by, colors=colors, title="Keystone taxa count", elements=10)
    upset.plot()
    fig = plt.gcf()
    fig.set_size_inches(5, 6)
    ax = plt.gca()
    try:
        ax.legend(loc='center left', fontsize=7, bbox_to_anchor=(1.0, 0.5), title=by)
    except:
        pass 
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(figures_dir / 'all_time_keystone_upsetplot.jpg', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrices(cm_rf, cm_dummy, le, figures_dir: Path):
    avg_cm_rf = np.mean(cm_rf, axis=0)
    avg_cm_dummy = np.mean(cm_dummy, axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Random Forest vs Dummy Classifier', fontsize=14)

    sns.heatmap(avg_cm_rf, annot=True, fmt='.1f', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[0, 0])
    axes[0, 0].set_title('RF Confusion Matrix (Raw)')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')

    rf_cm_norm = avg_cm_rf / avg_cm_rf.sum(axis=1)[:, np.newaxis]
    sns.heatmap(rf_cm_norm, annot=True, fmt='.3f', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[0, 1])
    axes[0, 1].set_title('RF Confusion Matrix (Normalized)')
    axes[0, 1].set_ylabel('True Label')
    axes[0, 1].set_xlabel('Predicted Label')

    sns.heatmap(avg_cm_dummy, annot=True, fmt='.1f', cmap='Reds', xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[1, 0])
    axes[1, 0].set_title('Dummy Confusion Matrix (Raw)')
    axes[1, 0].set_ylabel('True Label')
    axes[1, 0].set_xlabel('Predicted Label')

    dummy_cm_norm = avg_cm_dummy / avg_cm_dummy.sum(axis=1)[:, np.newaxis]
    sns.heatmap(dummy_cm_norm, annot=True, fmt='.3f', cmap='Reds', xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[1, 1])
    axes[1, 1].set_title('Dummy Confusion Matrix (Normalized)')
    axes[1, 1].set_ylabel('True Label')
    axes[1, 1].set_xlabel('Predicted Label')

    plt.tight_layout()
    plt.savefig(figures_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

