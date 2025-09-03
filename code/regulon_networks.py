import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA, NMF, FastICA
import igraph as ig
import random
import warnings
from tqdm import tqdm, trange
from joblib import Parallel, delayed
from itertools import combinations


def GRN_to_TF_edges(
    GRN, 
    method='dot_product', 
    binary=False, 
    log=False, 
    weight_by_TF_TF_importance=False
):
    """
    Calculates TF-TF network edges from a TF-Gene regulatory network (regulon).

    This function creates a TF-by-Gene matrix of importance scores and then computes
    the similarity between all pairs of TFs in parallel.

    Args:
        GRN (pd.DataFrame): DataFrame with columns ['TF', 'Gene', 'Importance'].
        method (str): The method to calculate similarity. Options are:
                      'dot_product': Computes the dot product of TF importance vectors.
                      'weighted_jaccard': Computes the weighted Jaccard index (Ruzicka similarity).
                      Default is 'dot_product'.
        binary (bool): If True, binarizes the importance scores to 0 or 1 before calculation.
                       Not compatible with 'weighted_jaccard'. Default is False.
        log (bool): If True, log10 transforms importance scores. Default is False.
        weight_by_TF_TF_importance (bool): If True, further weight edges by direct
                                           TF-TF importance scores. Default is False.

    Returns:
        pd.DataFrame: A DataFrame of TF-TF edges with similarity scores and weights.
    """
    # --- Input Validation ---
    if method not in ['dot_product', 'weighted_jaccard']:
        raise ValueError("Method must be 'dot_product' or 'weighted_jaccard'")
    if method == 'weighted_jaccard' and binary:
        raise ValueError("Cannot use binary=True with the 'weighted_jaccard' method, as it requires continuous weights.")

    # --- Pre-processing ---
    if log:
        GRN = GRN.assign(Importance=lambda x: np.log10(x['Importance'] + 1))

    print("Pivoting TF-Gene matrix...")
    GRN_pivot = GRN.pivot_table(
        index='TF', 
        columns='Gene', 
        values='Importance', 
        fill_value=0
    )
    
    all_TFs = GRN_pivot.index
    all_pairs = np.array(list(combinations(range(all_TFs.shape[0]), 2)))

    matrix = GRN_pivot.values
    if binary:
        matrix = np.where(matrix > 0, 1, 0)

    # --- Core Calculation (Parallelized) ---
    print(f"Calculating similarity for {len(all_pairs)} TF pairs using '{method}' method...")

    # Define the core calculation function based on the selected method
    if method == 'dot_product':
        def calculate_similarity(i):
            p = all_pairs[i]
            x = matrix[p[0], :]
            y = matrix[p[1], :]
            return np.dot(x, y)
            
    elif method == 'weighted_jaccard':
        def calculate_similarity(i):
            p = all_pairs[i]
            x = matrix[p[0], :]
            y = matrix[p[1], :]
            # The formula is the sum of element-wise minima / sum of element-wise maxima
            numerator = np.sum(np.minimum(x, y))
            denominator = np.sum(np.maximum(x, y))
            # Handle case where both vectors are all-zero to avoid division by zero
            return 0.0 if denominator == 0 else numerator / denominator

    # Run the calculation in parallel
    edges = Parallel(n_jobs=-2)(
        delayed(calculate_similarity)(i) for i in trange(len(all_pairs))
    )

    # --- Post-processing ---
    print("Constructing final edge DataFrame...")
    TF_xy = [(all_TFs[p[0]], all_TFs[p[1]]) for p in all_pairs]
    TF_edges = (pd.DataFrame(TF_xy, columns=['TF_x','TF_y'])
                .assign(similarity=edges)
                .query("similarity > 0")
    )

    # Add total importance of each TF to edges
    total_importance = GRN.groupby('TF').agg({'Importance':'sum'})['Importance']
    TF_edges = (TF_edges
                .join(total_importance.rename('total_x'), on='TF_x')
                .join(total_importance.rename('total_y'), on='TF_y')
    )

    # Weight by TF-TF importance if requested
    if weight_by_TF_TF_importance:
        TF_to_TF = (GRN
                    .loc[lambda x: x['Gene'].isin(x['TF'].unique())]
                    .rename({'TF':'TF_x', 'Gene':'TF_y', 'Importance': 'TF-TF Importance'}, axis=1)
        )
        TF_edges = (TF_edges
                    .set_index(['TF_x', 'TF_y'])
                    .join(TF_to_TF.set_index(['TF_x', 'TF_y']))
                    .loc[lambda x: x['TF-TF Importance'].notnull()]
                    .assign(similarity=lambda x: x['TF-TF Importance'] * x['similarity'])
                    .reset_index()
        )

    return TF_edges




def _build_similarity_matrix(TF_edges, weight_col='similarity'):
    """Helper function to construct the dense TF-TF similarity matrix."""
    print("Constructing the full TF-TF similarity matrix...")
    all_tfs = pd.unique(TF_edges[['TF_x', 'TF_y']].values.ravel('K'))
    all_tfs.sort()
    
    tf_matrix = pd.DataFrame(0, index=all_tfs, columns=all_tfs, dtype=np.float64)
    pivoted = TF_edges.pivot(index='TF_x', columns='TF_y', values=weight_col)
    tf_matrix.update(pivoted)
    tf_matrix = tf_matrix + tf_matrix.T
    np.fill_diagonal(tf_matrix.values, 1.0)
    tf_matrix.fillna(0, inplace=True)
    return tf_matrix


def perform_soft_clustering(
    TF_edges, 
    method='pca', 
    n_components=20, 
    weight_col='similarity',
    random_seed=42
):
    """
    Performs dimensionality reduction on the full TF-TF similarity matrix for soft clustering.

    This function avoids edge thresholding by using the dense similarity matrix to find 
    major axes of variation (components), which can be interpreted as continuous 
    community assignments.

    Args:
        TF_edges (pd.DataFrame): DataFrame with ['TF_x', 'TF_y', weight_col].
        method (str): The method to use. One of 'pca', 'nmf', or 'ica'.
        n_components (int or 'auto'): The number of components to extract. If 'auto',
                                      the function will try to find an optimal number.
        weight_col (str): The column containing the similarity score.
        random_seed (int): Seed for reproducibility for NMF and ICA.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: A tidy DataFrame of TF loadings for each component.
            - pd.Series or None: Importance of each component (variance explained for PCA,
              None for other methods).
    """
    tf_matrix = _build_similarity_matrix(TF_edges, weight_col)

    print(f"Running {method.upper()} on the {tf_matrix.shape[0]}x{tf_matrix.shape[1]} matrix...")
    
    # --- Automatic selection of n_components ---
    if n_components == 'auto':
        if method == 'nmf':
            print("Finding optimal n_components for NMF using reconstruction error elbow...")
            k_range = range(2, 15) 
            errors = []
            for k in tqdm(k_range, desc="Testing k for NMF"):
                model = NMF(n_components=k, init='random', random_state=random_seed, max_iter=200)
                model.fit(tf_matrix)
                errors.append(model.reconstruction_err_)
            
            # Find the elbow point
            # We look for the point with the maximum perpendicular distance from the line
            # connecting the first and last points of the error curve.
            points = np.column_stack((k_range, errors))
            line_vec = points[-1] - points[0]
            line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
            vec_from_first = points - points[0]
            scalar_product = np.sum(vec_from_first * np.tile(line_vec_norm, (len(k_range), 1)), axis=1)
            vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
            vec_to_line_perp = vec_from_first - vec_from_first_parallel
            dist_to_line = np.sqrt(np.sum(vec_to_line_perp**2, axis=1))
            
            n_components = k_range[np.argmax(dist_to_line)]
            print(f"Optimal n_components for NMF found: {n_components}")

        elif method == 'ica':
            print("Finding optimal n_components for ICA using PCA variance heuristic...")
            pca = PCA()
            pca.fit(tf_matrix)
            cumsum_var = np.cumsum(pca.explained_variance_ratio_)
            # Find the number of components to explain 95% of variance
            n_components = np.argmax(cumsum_var >= 0.95) + 1
            print(f"Optimal n_components for ICA (explaining 95% variance): {n_components}")
        
        elif method == 'pca':
            print("For PCA, 'auto' uses n_components=20. PCA returns variance explained for all components.")
            n_components = 20

    # --- Run final model with selected n_components ---
    model_params = {'n_components': n_components, 'random_state': random_seed}
    loadings_matrix = None
    importance_series = None
    
    if method == 'pca':
        model = PCA(**model_params)
        model.fit(tf_matrix)
        loadings_matrix = model.components_.T
        importance_series = pd.Series(model.explained_variance_ratio_, name='importance')
    
    elif method == 'nmf':
        model = NMF(**model_params, init='random', max_iter=500)
        loadings_matrix = model.fit_transform(tf_matrix)
        
    elif method == 'ica':
        model = FastICA(**model_params, max_iter=500)
        loadings_matrix = model.fit_transform(tf_matrix)
        
    else:
        raise ValueError("Method must be one of 'pca', 'nmf', or 'ica'")

    # --- Format results into a tidy DataFrame ---
    comp_names = [f'{method}{i+1}' for i in range(n_components)]
    
    tf_loadings = (pd.DataFrame(loadings_matrix, columns=comp_names, index=tf_matrix.index)
                   .melt(ignore_index=False, var_name='Component', value_name='Loading')
                   .assign(Component=lambda x: pd.Categorical(x['Component'], categories=comp_names, ordered=True))
                   .reset_index(names='TF')
                   )
    
    if importance_series is not None:
        importance_series.index = comp_names

    print(f"{method.upper()} complete.")
    return tf_loadings, importance_series









# --- Step 1: Helper Functions (Refined from your original code) ---

def create_graph_from_edges(TF_edges, quantile_filter=0.01, weight_col='weight'):
    """
    Creates an igraph.Graph object from a DataFrame of TF edges.

    Args:
        TF_edges (pd.DataFrame): DataFrame with columns ['TF_x', 'TF_y', weight_col].
        quantile_filter (float): The quantile of edge weights to keep. 
                                 e.g., 0.01 keeps the top 1% of edges.
        weight_col (str): The name of the column to use for edge weights.

    Returns:
        igraph.Graph: A weighted, undirected graph object.
    """
    print(f"Filtering edges to keep the top {quantile_filter*100:.1f}%...")
    # Rank edges and filter based on the quantile
    min_weight = TF_edges[weight_col].quantile(1 - quantile_filter)
    TF_edges_filtered = TF_edges.query(f"{weight_col} >= @min_weight")
    
    # Create a list of tuples for graph creation
    edge_tuples = list(
        TF_edges_filtered[['TF_x', 'TF_y', weight_col]].itertuples(index=False, name=None)
    )
    
    # Create the graph
    g = ig.Graph.TupleList(edge_tuples, directed=False, weights=True)
    g.es['weight'] = TF_edges_filtered[weight_col].values
    g.es["log_weight"] = [np.log10(w + 1) for w in g.es['weight']]
    
    # Add node attributes like total gene connection strength
    if 'total_x' in TF_edges.columns and 'TF_x' in TF_edges.columns:
        total_importance = TF_edges.groupby('TF_x')['total_x'].first()
        g.vs['total_importance'] = [total_importance.get(name, 0) for name in g.vs['name']]
        
    print(f"Graph created with {g.vcount()} nodes and {g.ecount()} edges.")
    return g


def _annotate_graph_with_communities(g, communities):
    """
    Internal helper to add community structure, colors, and hubs to a graph.
    """
    # Assign community membership and color to each vertex
    for i, community in enumerate(communities):
        g.vs[community]["community_id"] = i
        g.vs[community]["color"] = i
    
    # Color edges based on the community of their source node
    g.es["color"] = [g.vs[e.source]["color"] for e in g.es]

    # Identify hubs within each community (e.g., top 3 by strength)
    g.vs['is_hub'] = False
    for i in range(len(communities)):
        community_nodes = g.vs.select(community_id_eq=i)
        if not community_nodes:
            continue

        strengths = [g.vs[node].strength(weights='log_weight') for node in community]
        # Get indices of top 3 nodes by strength
        hub_indices_in_community = np.argsort(strengths)[-3:]
        hub_global_indices = [node.index for node_idx, node in enumerate(community_nodes) if node_idx in hub_indices_in_community]
        
        if hub_global_indices:
            g.vs[hub_global_indices]['is_hub'] = True
            
    return g


def prune_small_communities(g, size_threshold=5):
    """
    Deletes vertices belonging to communities smaller than a given threshold.

    Args:
        g (igraph.Graph): An annotated graph with a 'community_id' vertex attribute.
        size_threshold (int): The minimum size for a community to be kept.

    Returns:
        igraph.Graph: The graph with small communities removed.
    """
    if 'community_id' not in g.vertex_attributes():
        print("Warning: No 'community_id' attribute found. Cannot prune.")
        return g
        
    community_counts = pd.Series(g.vs['community_id']).value_counts()
    small_communities = community_counts[community_counts < size_threshold].index
    
    if len(small_communities) > 0:
        to_delete_vs = g.vs.select(community_id_in=small_communities)
        print(f"Pruning {len(to_delete_vs)} nodes from {len(small_communities)} small communities...")
        g.delete_vertices(to_delete_vs)
        # Re-index communities after deletion
        g.vs['color'] = pd.Series(g.vs['color']).astype('category').cat.codes
        g.es['color'] = pd.Series(g.es['color']).astype('category').cat.codes.replace({-1:None})
    else:
        print("No communities were smaller than the threshold.")

    return g


# --- Step 2: The Main Consensus Clustering Function ---

def consensus_cluster_network(
    TF_edges, 
    quantile_filter=0.01,
    resolution_range=np.linspace(0.4, 1.5, 20),
    final_resolution=1.0,
    n_iterations=10,
    prune_threshold=5,
    random_seed=42
):
    """
    Performs ensemble-based consensus clustering on a TF-TF network.

    This function generates multiple network partitions using the Leiden algorithm
    across a range of resolutions, builds a consensus matrix from these partitions,
    and then clusters the consensus matrix to find robust communities.

    Args:
        TF_edges (pd.DataFrame): DataFrame of TF-TF edges.
        quantile_filter (float): Quantile of edges to use for the initial graph.
        resolution_range (np.array): A range of resolution parameters for the Leiden algorithm.
        final_resolution (float): The resolution to use for the final clustering of the consensus matrix.
        prune_threshold (int or None): If not None, removes final communities smaller than this size.
        random_seed (int): Seed for reproducibility.

    Returns:
        igraph.Graph: The original graph annotated with stable consensus communities.
    """
    # 1. Create the initial graph from the top N% of edges
    random.seed(random_seed)
    g = create_graph_from_edges(TF_edges, quantile_filter=quantile_filter, weight_col='similarity')

    # 2. Generate an ensemble of partitions
    print(f"Generating {len(resolution_range)} partitions for consensus matrix...")
    partitions = []
    for resolution in tqdm(resolution_range, desc="Partitioning"):
        part = g.community_leiden(weights='log_weight', resolution_parameter=resolution, n_iterations=n_iterations)
        partitions.append(part)

    # 3. Build the consensus matrix
    print("Building consensus matrix...")
    num_nodes = g.vcount()
    consensus_matrix = np.zeros((num_nodes, num_nodes))
    # Create a mapping from TF name to index for quick lookups
    name_to_idx = {name: i for i, name in enumerate(g.vs["name"])}

    for part in tqdm(partitions, desc="Aggregating"):
        # Create a membership array for the current partition
        membership = np.array(part.membership)
        # Efficiently update the consensus matrix
        for community_id in range(len(part)):
            nodes_in_comm = np.where(membership == community_id)[0]
            # Get all pairs of nodes within this community
            for i, j in combinations(nodes_in_comm, 2):
                consensus_matrix[i, j] += 1
                consensus_matrix[j, i] += 1
    
    # Normalize the matrix by the number of partitions
    consensus_matrix /= len(partitions)
    
    # 4. Cluster the consensus matrix to find final communities
    print("Clustering the consensus matrix...")
    # Create a new graph where edge weights are consensus scores
    g_consensus = ig.Graph.Weighted_Adjacency(consensus_matrix.tolist(), mode='undirected', attr="weight")
    g_consensus.vs["name"] = g.vs["name"] # Keep track of names

    # Find final communities on this highly stable graph
    final_communities = g_consensus.community_leiden(
        weights='weight', 
        resolution_parameter=final_resolution,
        n_iterations=n_iterations
    )
    
    print(f"Identified {len(final_communities)} stable communities.")
    
    # 5. Annotate the *original* graph with the final communities
    g = _annotate_graph_with_communities(g, final_communities)
    
    # 6. Optionally prune small communities
    if prune_threshold is not None and prune_threshold > 0:
        g = prune_small_communities(g, size_threshold=prune_threshold)

    # 7. Add total strength of gene connections of nodes
    g.vs['gene_strength'] = TF_edges.groupby('TF_x').agg({'total_x':'mean'})['total_x'].values

    return g





def plot_TF_graph(g, enrichment=None, random_seed=5, width_modifier=2, target=None, colors=None, hub_pct=None, quantile_filter=0.01):
    random.seed(random_seed)

    # Filter edges just for graph plotting
    edges = pd.Series(g.es['weight'])
    to_delete = list(edges[edges.rank(pct=True) < (1-quantile_filter)].index)
    g = g.copy()
    g.delete_edges(to_delete)

    # Set vertex size and edges width
    # g.vs["size"] = ig.rescale(g.strength(weights='log_weight'), out_range=(20,50))
    g.vs["size"] = ig.rescale(g.vs['gene_strength'], out_range=(10,75))
    g.es["width"] = ig.rescale(g.es["log_weight"], out_range=(.5,2*width_modifier))

    # Use edge width to define layout
    graph_design = {}
    graph_design["layout"] = g.layout("fr", weights=[w for w in g.es["log_weight"]])
    # graph_design["layout"] = g.layout("kamada_kawai", weights=[w for w in g.es["log_weight"]])

    # Colour by C1-3
    if enrichment is not None:
        cmap = sns.color_palette("vlag", as_cmap=True)
        lim = pd.Series(g.vs[enrichment]).abs().max()
        enrichment_scaled = ig.rescale(g.vs[enrichment], in_range=(-lim,lim), clamp=False)
        enrichment_rank = pd.Series(enrichment_scaled).rank()
        enrichment_invrank = pd.Series(enrichment_scaled).rank(ascending=False)
        enrichment_label = (enrichment_rank <= 8) | (enrichment_invrank <= 8)
        # enrichment_label = [q< 0.05 for q in g.vs[f'{enrichment}_q']]
        graph_design["vertex_color"] = [cmap(c) for c in enrichment_scaled]

        # Add labels for top enriched nodes
        g.vs['label'] = [name if to_label else None for name, to_label in zip(g.vs['name'], enrichment_label)]

    # Colour by module
    elif enrichment is None:
        n_communities = len(np.unique(g.vs['color']))
        if colors is None:
            graph_design['palette'] = ig.RainbowPalette(n=n_communities)
        else:
            from igraph.drawing.colors import PrecalculatedPalette, ClusterColoringPalette
            graph_design['palette'] = PrecalculatedPalette(colors)

        # Re-define hubs for plotting purpose
        if hub_pct is not None:
            g.vs['is_hub'] = pd.Series(g.strength(weights='log_weight')).rank(pct=True) > hub_pct

        # Add labels by size
        g.vs['label'] = [name if to_label else None for name, to_label in zip(g.vs['name'], g.vs['is_hub'])]
        # g.vs['label'] = [name if to_label else None for name, to_label in zip(g.vs['color'], g.vs['is_hub'])]
    
    # graph_design['vertex_size'] = [s for s in g.vs['size']]
    graph_design["vertex_label_size"] = 10
    graph_design["vertex_label_angle"] = np.pi/360*-60
    # graph_design["vertex_label_dist"] = .5#[s for s in g.vs["size"]]
    # graph_design["vertex_label_dist"] = [40/s for s in g.vs["size"]]
    graph_design["vertex_label_dist"] = [4/np.sqrt(s) for s in g.vs["size"]]
    graph_design["vertex_frame_width"] = .5
    # graph_design["vertex_frame_color"] = 'black'
    # graph_design["vertex_frame_color"] = 'grey'
    # graph_design["edge_color"] = 'grey'
    graph_design["margin"] = 0

    return ig.plot(g, target=target, **graph_design)
























### LEGACY



def add_communities_to_graph(g, method, weights='weight', resolution=0.8):
    if method=='leiden':
        communities = g.community_leiden(weights=weights, resolution=resolution)
    elif method=='infomap':
        communities = g.community_infomap(edge_weights=weights)
    elif method=='edge_betweenness':
        communities = g.community_edge_betweenness(weights=weights, directed=False).as_clustering()
    elif method=='fastgreedy':
        communities = g.community_fastgreedy(weights=weights).as_clustering()
    elif method=='eigenvector':
        communities = g.community_leading_eigenvector(weights=weights, clusters=12)
    elif method=='multilevel':
        communities = g.community_multilevel(weights=weights)

    for i, community in enumerate(communities):
        g.vs[community]["color"] = i
        community_edges = g.es.select(_within=community)
        community_edges["color"] = i
        # Set hubs to top n nodes in community
        community_node_strengths = [g.vs[node].strength(weights='log_weight') for node in community]
        g.vs[community]['is_hub'] = pd.Series(community_node_strengths).rank(ascending=False)<=5
    
    return g

def prune_small_clusters(TF_graph, size_threshold=5, communities=None, resolution=0.08):
    TF_clusters = (pd.DataFrame({
            'TF': TF_graph.vs['name'], 
            'cluster': TF_graph.vs['color']})
        .assign(size = lambda x: x.groupby('cluster').transform('size'))
    )

    # Pick vertices in small clusters to delete
    to_delete = TF_clusters.query("size<@size_threshold")['TF']
    # Delete the vertices
    TF_graph.delete_vertices(to_delete)
    if communities is not None:
        # Recompute clusters
        TF_graph = add_communities_to_graph(TF_graph, method=communities, resolution=resolution)
    else:
        # If communities are not to be recomputed, just re-assign colors
        # Convert NaN to None for plotting
        # TF_graph.vs['color'] = pd.factorize(TF_graph.vs['color'])[0]
        # TF_graph.es['color'] = pd.Series(pd.factorize(TF_graph.es['color'])[0]).replace({-1:None})
        TF_graph.vs['color'] = pd.Series(TF_graph.vs['color']).astype('category').cat.codes
        TF_graph.es['color'] = pd.Series(TF_graph.es['color']).astype('category').cat.codes.replace({-1:None})

    return TF_graph



def TF_edges_to_graph(TF_edges, stats=None, quantile_filter=0.01, communities='leiden', resolution=0.8, size_threshold=5, random_seed=5):
    TF_edges_filter = TF_edges.loc[lambda x: x['similarity'].rank(pct=True) > (1-quantile_filter)]
    TF_edges_list = list(
        TF_edges_filter[['TF_x','TF_y','similarity']].itertuples(index=False, name=None)
    )
    g = ig.Graph.TupleList(TF_edges_list, weights=True)

    # Add log weighted edges
    g.es["log_weight"] = [np.log10(w+1) for w in g.es['weight']]

    # Add hubs
    g.vs['is_hub'] = pd.Series(g.strength(weights='log_weight')).rank(pct=True) > 0

    # Add total strength of gene connections of nodes
    g.vs['gene_strength'] = TF_edges.groupby('TF_x').agg({'total_x':'size'})['total_x'].values

    # Add communities to graph
    if communities is not None:
        random.seed(random_seed)
        g = add_communities_to_graph(g, method=communities, weights='log_weight', resolution=resolution)

    # Prune small clusters
    if size_threshold is not None:
        g = prune_small_clusters(g, size_threshold=size_threshold, communities=None, resolution=resolution)

    # Add enrichment scores
    if stats is not None:
        stats_pivot_z = stats.pivot(index='TF', columns='enrichment', values='z')
        stats_pivot_q = stats.pivot(index='TF', columns='enrichment', values='q')
        with warnings.catch_warnings():
            # warnings.simplefilter("ignore")
            for enrichment in stats_pivot_z.columns:
                g.vs[enrichment] = pd.Series(stats_pivot_z[enrichment], index=list(g.vs['name']))
                g.vs[f'{enrichment}_q'] = pd.Series(stats_pivot_q[enrichment], index=list(g.vs['name']))
    
    return g

