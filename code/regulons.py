import numpy as np
import pandas as pd
import anndata as ad
import mygene
from enrichments import *


def get_ahba_GRN(
        path_to_ahba_weights="../data/ahba_dme_hcp_top8kgenes_weights.csv"
    ):
    """
    Load the AHBA Gene Regulatory Network (GRN) from a CSV file.
    Parameters:
    - path_to_ahba_weights: Path to the CSV file containing AHBA weights.
    Returns:
    - ahba_GRN: DataFrame containing the AHBA GRN with columns 'Network', 'Gene', and 'Importance'.
    """
    ahba = pd.read_csv(path_to_ahba_weights, index_col=0)

    ahba_melt = (ahba
                .reset_index(names='Gene')
                .melt(id_vars=['Gene'], var_name='Network', value_name='Importance'))

    ahba_GRNpos = (ahba_melt
                .sort_values('Importance', ascending=False)
                .loc[lambda x: x.groupby('Network')['Importance'].rank(ascending=False)<1000]
                .assign(Importance = 1)  # Set Importance to 1 for positive links
                # .loc[lambda x: x['Importance'] > 0]
                .assign(Network=lambda x: x['Network'] + '+')
    )

    ahba_GRNneg = (ahba_melt
                .sort_values('Importance', ascending=True)
                .loc[lambda x: x.groupby('Network')['Importance'].rank(ascending=True)<1000]
                .assign(Importance = 1)  # Set Importance to 1 for negative links
                # .loc[lambda x: x['Importance'] < 0]
                # .assign(Importance=lambda x: -x['Importance'])
                .assign(Network=lambda x: x['Network'] + '-')
    )

    ahba_GRN = pd.concat([ahba_GRNpos, ahba_GRNneg])
    return(ahba_GRN)


def project_GRN(adata, GRN, GRN_name='GRN', use_raw=False, use_residuals=False, normalize=False, use_highly_variable=True, log_transform=False):
    """
    Project a Gene Regulatory Network (GRN) onto an AnnData object.
    The default and recommended way to use this function is to apply it to expression data that has been normalized but NOT log transformed.
    Instead, apply the log transformation after the projection.
    Parameters:
    - GRN: DataFrame containing the GRN with columns 'TF', 'Gene', and 'Importance'.
    - GRN_name: Name to use for the resulting obsm key.
    - adata: AnnData object containing the data to project onto.
    - use_raw: If True, uses the raw data from adata.raw. Defaults to False.
    - use_residuals: If True, uses residuals from pearson residuals normalization instead of raw data. Defaults to False.
    - normalize: If True, normalizes the projected data to have a total sum of 1e6. Defaults to False.
    - log_transform: If True, applies log1p transformation to the projected data. Defaults to True.
    Returns:
    - None: The function modifies adata.obsm in place.
    """

    GRN_pivot = GRN.pivot_table(
        index='Network',
        columns='Gene', 
        values='Importance', 
        fill_value=0
    )

    if use_residuals:
        # Use the pearson residuals normalization 
        X = adata.uns['pearson_residuals_normalization']['pearson_residuals_df']
        # Matched genes are the intersection of the GRN genes and the normalized genes
        matched_genes = np.intersect1d(GRN_pivot.columns, X.columns)
    elif use_raw:
        # Use the raw data
        X = pd.DataFrame(adata.raw.X.todense(), index=adata.obs_names, columns=adata.var_names)
        # Matched genes are the intersection of the GRN genes and the adata var genes
        matched_genes = np.intersect1d(GRN_pivot.columns, adata.var.index)
    else:
        # Use whatever is in adata.X
        X = adata.to_df()
        # Optionally filter for highly variable genes
        if use_highly_variable:
            X = X.loc[:, adata.var['highly_variable']]
            
        # Matched genes are the intersection of the GRN genes and the adata var genes
        matched_genes = np.intersect1d(GRN_pivot.columns, X.columns)

    # Ensure unique columns and filter to matched genes
    X = X.loc[:, lambda x: ~x.columns.duplicated()].loc[:, matched_genes]
    Y = GRN_pivot.loc[:, matched_genes]

    # Project X onto Y
    projected = (X @ Y.T)

    if normalize:
        # Normalize the projected data to have a total sum of 1e6
        projected = (projected.T / projected.sum(axis=1)).T * 1e4
    if log_transform:
        # Apply log1p transformation
        projected = np.log1p(projected)

    # Add to adata.obsm as array to work with scanpy plotting
    adata.obsm[GRN_name] = projected.values
    adata.uns[GRN_name + '_names'] = GRN_pivot.index.tolist()

    # adata_TFs = adata.copy()
    # adata_TFs.var.index = adata_TFs.var.index.astype(str)
    # adata_TFs.obs.index = adata_TFs.obs.index.astype(str)
    # adata_TFs.obs_names_make_unique()

    # matched_genes = np.intersect1d(GRN_pivot.columns, adata_TFs.var.index)
    # X = adata_TFs.to_df(layer=layer).loc[:, lambda x: ~x.columns.duplicated()].loc[:, matched_genes]

    # projected = (X @ GRN_pivot.loc[:, matched_genes].T)

    

    # Don't return an adata, instead add to adata.obsm
    # if map_TFs:
    #     mg = mygene.MyGeneInfo()
    #     mg_out = mg.querymany(GRN_pivot.index, 
    #                         scopes='symbol', 
    #                         fields='ensembl.gene', 
    #                         species='human', 
    #                         as_dataframe=True, 
    #                         df_index=False)
    #     TF_var = (mg_out
    #             .rename({'query':'Symbol', 'ensembl.gene':'ensemblID'}, axis=1)
    #             .loc[:, ['Symbol', 'ensemblID']]
    #             .set_index('Symbol', drop=False)
    #             )
    # else:
    #     TF_var = pd.DataFrame(index=GRN_pivot.index, data={'Symbol': GRN_pivot.index})
    
    # adata_TFs = ad.AnnData(X=projected, obs=adata_TFs.obs, var=TF_var)

    


def compute_weighted_jaccard(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the weighted Jaccard index for each set between two dataframes.

    The weighted Jaccard index is calculated as the sum of the minimum importances
    for common genes divided by the sum of the maximum importances for all genes
    in the union of the sets.

    Args:
        df1: A pandas DataFrame with columns ['TF', 'Gene', 'Importance'].
        df2: A second pandas DataFrame with the same structure as df1.

    Returns:
        A pandas DataFrame with columns ['TF', 'Jaccard'] containing the
        computed weighted Jaccard index for every unique set from both dataframes.
    """
    # Validate input DataFrames
    for df_name, df in [('df1', df1), ('df2', df2)]:
        if not all(col in df.columns for col in ['TF', 'Gene', 'Importance']):
            raise ValueError(f"'{df_name}' must contain 'TF', 'Gene', and 'Importance' columns.")

    # Get a list of all unique sets from both dataframes
    all_TFs = pd.concat([df1['TF'], df2['TF']]).unique()

    results = []

    # Iterate over each unique TF
    for s in all_TFs:
        # Filter data for the current TF
        tf1_data = df1[df1['TF'] == s]
        tf2_data = df2[df2['TF'] == s]

        num_genes_1 = len(tf1_data)
        num_genes_2 = len(tf2_data)

        # Handle cases where the TF is in only one of the dataframes
        if tf1_data.empty or tf2_data.empty:
            jaccard_index = 0.0
            classic_jaccard = 0.0
            num_intersection = 0
            num_union = num_genes_1 if num_genes_2 == 0 else num_genes_2
            results.append({
                'TF': s, 
                'Jaccard': jaccard_index,
                'Jaccard_Classic': classic_jaccard,
                'Num_Genes_df1': num_genes_1,
                'Num_Genes_df2': num_genes_2,
                'Num_Genes_Intersection': num_intersection,
                'Num_Genes_Union': num_union
            })
            continue

        # Merge the two dataframes on 'Gene' to get a union of all genes for the TF
        merged_df = pd.merge(
            tf1_data[['Gene', 'Importance']],
            tf2_data[['Gene', 'Importance']],
            on='Gene',
            how='outer',
            suffixes=('_1', '_2')
        )

        # Calculate intersection and union counts from the merged frame
        num_union = len(merged_df)
        num_intersection = len(merged_df.dropna())

        # Fill NaN values with 0 for genes present in only one of the sets
        merged_df.fillna(0, inplace=True)

        # Calculate the numerator: sum of the minimum of importances for each gene
        numerator = np.minimum(merged_df['Importance_1'], merged_df['Importance_2']).sum()

        # Calculate the denominator: sum of the maximum of importances for each gene
        denominator = np.maximum(merged_df['Importance_1'], merged_df['Importance_2']).sum()

        # Calculate the weighted Jaccard index
        if denominator == 0:
            # This case occurs if a set exists but has no genes or all importances are 0
            jaccard_index = 0.0
        else:
            jaccard_index = numerator / denominator

        # Calculate the classic (non-weighted) Jaccard index
        if num_union == 0:
            classic_jaccard = 0.0
        else:
            classic_jaccard = num_intersection / num_union

        results.append({
            'TF': s, 
            'Jaccard': jaccard_index,
            'Jaccard_Classic': classic_jaccard,
            'Num_Genes_df1': num_genes_1,
            'Num_Genes_df2': num_genes_2,
            'Num_Genes_Intersection': num_intersection,
            'Num_Genes_Union': num_union
        })

    # Create the final result DataFrame
    result_df = pd.DataFrame(results)

    return result_df





# input dictionary of GRN versions, output AHBA stats
def get_GRN_AHBA_stats(GRN_versions, n_shuffle=1000):
    AHBA = pd.read_csv("../data/ahba_dme_hcp_top8kgenes_weights.csv", index_col=0)
    AHBA_null = AHBA.pipe(shuffle_weights, n=n_shuffle)

    GRN_AHBA_stats = {}
    for name, version in GRN_versions.items():
        GRN_AHBA_stats[name] = compute_null_p(
            *compute_enrichments(AHBA, AHBA_null, version, label_column='TF', gene_column='Gene')
        )
        print(f"Computed AHBA enrichments for {name} ({version.shape[0]} TF-Gene links).")

    return(GRN_AHBA_stats)


def plot_enrichments_vs_size(GRN_AHBA_stats, name_replacements=None, figsize=(10,12)):
    to_plot = (pd.concat(GRN_AHBA_stats).reset_index(0, names='version')
        .sort_values('n_genes', ascending=False)
        .assign(label = lambda x: pd.Categorical(x['label'].astype('str'), ordered=True, categories = x['label'].astype('str').unique()))
        .assign(r = lambda x: x['z'].abs().groupby(x['version']+x['C']+x['pos'].astype('str')).rank(ascending=False, method='dense'))
    )

    if name_replacements is not None:
        to_plot = (to_plot
            .assign(version = lambda x: pd.Categorical(
                x['version'].replace(name_replacements), 
                ordered=True, categories=name_replacements.values())
            )
        )

    text_labels = (to_plot
        # .query("r<=1 & q<0.05 & version.str.contains('cell-lineage genes only')")['label'].unique().astype('str')
        .query("r<=1 & q<0.05 & version=='velmeshev_ex-1'")['label'].unique().astype('str')
    )
    
    adjust_text = {
    #    "expand": (2, 2),
        "max_move": 100,
        "ensure_inside_axes": True,
        "expand_axes": True,
        "arrowprops": {"arrowstyle": "-", "color": "black"}
    }
    print(f"TFs to label: {text_labels}")
    # to_plot.query("r<=3 & q<0.05 & version=='velmeshev'")
    p = (to_plot
        .pipe(ggplot, aes(x='n_genes', y='z'))
        + facet_grid('version~C')
        + geom_point(aes(color='q>0.05'), size=.4, alpha=.6)
        # + geom_text(data=to_plot.query("(r<=3) & (q<0.05)"), mapping=aes(label='label'), size=6, ha='left', nudge_x=1)
        + geom_text(data=to_plot.query("label in @text_labels"), 
            mapping=aes(label='label'), adjust_text=adjust_text,
            size=8, ha='left', nudge_x=1)
        + scale_color_discrete(labels=['FDR<0.05','FDR>0.05'])
        + scale_x_continuous(name='TFs arranged by number of genes in regulon')
        + ylab('Enrichment z-score')
        + theme_minimal()
        + theme(
            figure_size = figsize,
            text = element_text(family='sans-serif'),
            plot_background = element_rect(color='white', fill='white'),
            legend_position = 'bottom',
            legend_title = element_blank(),
            axis_line = element_line(size=.5),
            axis_ticks = element_blank(),
            strip_text = element_text(rotation=0, face='bold'),
            strip_background = element_blank()
        )
        + ggtitle('Regulon enrichments in AHBA C1-3')
    )
    return p

def get_AHBA_enriched_regulon_pct(GRN_AHBA_stats, order=None):
    if order is None:
        order = GRN_AHBA_stats.keys()

    df_percents = (pd.concat(GRN_AHBA_stats).reset_index(0, names='version')
        .assign(version = lambda x: pd.Categorical(x['version'], ordered=True, categories=order))
        .assign(sig = lambda x: np.select([
                (x['q']<0.05) & (x['pos']),
                (x['q']<0.05) & (~x['pos']),
                x['q']>0.05
            ], ['pos','neg','notsig']
            ))
        .groupby(['version','C','sig']).size().to_frame('n')
        .unstack(2)
        .droplevel(0, axis=1)
        .assign(total = lambda x: x.sum(axis=1))
        .drop('notsig', axis=1)
        .assign(neg = lambda x: (x['neg']/x['total']))
        .assign(pos = lambda x: (x['pos']/x['total']))
        .rename_axis(None, axis=1)
        .drop('total', axis=1)
    )

    df_percents = (pd.concat({'Enriched %':df_percents}, axis=1)
                   .style.format({
                       ('Enriched %','neg'):'{:,.0%}', 
                       ('Enriched %','pos'):'{:,.0%}'}))
    return df_percents


def get_regulon_overlaps(version_A, version_B, importance_quantile=None):
    A = version_A.set_index('TF')
    if importance_quantile is not None:
        A = A.loc[lambda x: x['Importance']>=x['Importance'].quantile(importance_quantile)]
    B = version_B.set_index('TF')
    matched_TFs = set(A.index) & set(B.index)
    matched_TF_pct = len(matched_TFs) / len(set(A.index))

    overlaps = np.ndarray(len(matched_TFs))
    for i,TF in enumerate(matched_TFs):
        a = A.loc[TF,'Gene']
        b = B.loc[TF,'Gene']
        overlaps[i] = len(set(a) & set(b))/len(set(a))
    # return pd.Series(overlaps, index=matched_TFs)
    return matched_TF_pct, overlaps

from itertools import permutations
def get_all_regulon_overlaps(GRN_versions, **kwargs):
    pairs = list(permutations(GRN_versions.keys(), 2))
    TF_overlaps = {}
    gene_overlaps = {}
    for pair in pairs:
        version_A = GRN_versions[pair[0]]
        version_B = GRN_versions[pair[1]]
        TF_overlaps[pair], gene_overlaps[pair] = \
            get_regulon_overlaps(version_A, version_B, **kwargs)
    return TF_overlaps, gene_overlaps


def plot_gene_overlaps(gene_overlaps, order=None):
    if order is None:
        order = np.unique([x[0] for x in gene_overlaps.keys()])

    p = (pd.concat({k:pd.Series(v) for k, v in gene_overlaps.items()})
        .reset_index([0,1]).set_axis(['x','y','overlap'], axis=1)
        .assign(
            x = lambda a: pd.Categorical(a['x'], ordered=True, categories=order),
            y = lambda a: pd.Categorical(a['y'], ordered=True, categories=order)
        )
        .pipe(ggplot)
        + facet_grid('x~y')
        #  + geom_histogram(aes(x='overlap', y=after_stat('ncount')), binwidth=.03)
        + geom_histogram(aes(x='overlap'), binwidth=.03)
        + scale_x_continuous(name='% of genes from x in y in matched TF regulons',
                            labels=lambda l: ["%d%%" % (v * 100) for v in l])
        + theme_minimal()
        + theme(
            figure_size = (10,9),
            aspect_ratio = 1,
            plot_background = element_rect(color='white', fill='white'),
            legend_position = 'bottom',
            legend_title = element_blank(),
            panel_grid_minor = element_blank(),
            axis_line = element_line(size=.5),
            axis_ticks = element_blank(),
            strip_text = element_text(rotation=0, face='bold'),
            strip_background = element_blank()
        )
    )
    return p






def map_TF_stats_to_genes(TF_stats, GRN):
    TF_stats_by_genes = (TF_stats
     .dropna()
     .join(GRN.set_index('TF'))
     .assign(weighted_stat = lambda x: x['stat'] * x['Importance'])
     .reset_index(names='TF')
    #  .loc[lambda x: x.groupby('Gene')['weighted_stat'].apply(lambda y: y.abs().idxmax()), :]
    #  .loc[lambda x: x.groupby('Gene')['Importance'].apply(lambda y: y.idxmax()), :]
    #  .loc[lambda x: x.groupby('Gene')['padj'].apply(lambda y: y.idxmin()), :]
    #  .set_index('Gene')
      .groupby('Gene')
      .agg({'stat': 'mean', 'weighted_stat': 'mean', 'log2FoldChange':'mean', 'padj': 'mean'})
    )
    return(TF_stats_by_genes)



from scipy.spatial import cKDTree as KDTree

def get_KL(x, y, xtree, ytree):
    n,d = x.shape
    m,dy = y.shape

    r = xtree.query(x, k=2, eps=.01, p=2)[0][:,1]
    s = ytree.query(x, k=1, eps=.01, p=2)[0]

    rs_ratio = r/s

    KL = -np.log(rs_ratio).sum() * d / n + np.log(m / (n - 1.))
    KL = np.maximum(KL, 0)

    return(KL)

    """
def get_invKL(x, y, symmetric=True):
    Get inverse KL across cells for two TF regulons
    Expects one column named 'TF', other columns numeric (and z-scored, probably)
    """
    xtree = KDTree(x)
    ytree = KDTree(y)

    KL12 = get_KL(x, y, xtree, ytree)
    KL21 = get_KL(y, x, ytree, xtree)

    if symmetric:
        KL = (KL12 + KL21)/2
    else:
        KL = KL12
    
    invKL = 1/(1+KL)
    return(invKL)

# Notebook code for running KL

# from itertools import permutations
# from joblib import Parallel, delayed
# from tqdm.notebook import trange

# # TF_scores = [scores.query("TF==@TF").drop('TF', axis=1) for TF in scores['TF'].unique()]
# TFs_to_test = scores['TF'].unique()
# # pairs = list(permutations(range(0, len(TFs_to_test)), 2))
# pairs = list(permutations(TFs_to_test, 2))

# def do_GRN_KL(i):
#     pair = pairs[i]
#     scores1 = scores.query("TF==@pair[0]").drop('TF', axis=1)
#     scores2 = scores.query("TF==@pair[1]").drop('TF', axis=1)
#     # scores1 = TF_scores[pair[0]]
#     # scores2 = TF_scores[pair[1]]
#     invKL = get_invKL(scores1, scores2)
#     return invKL

# similarities = Parallel(n_jobs = -2) (
#     delayed(do_GRN_KL)(i) for i in trange(len(pairs))
# )
# similarities = pd.Series(similarities, index=pd.MultiIndex.from_tuples(pairs))
# similarities.to_csv("../outputs/similarities.csv")         


# Legacy
# def load_emani(path = "../emani2024", unify_celltypes=True): 
#     CellOrder = ['L2.3.IT', 'L4.IT', 'L5.6.NP','L5.ET', 'L5.IT', 'L6.CT', 'L6.IT', 'L6.IT.Car3', 'L6b', 
#         'Lamp5','Lamp5.Lhx6', 'Pax6', 'Pvalb', 'Sncg', 'Sst','Vip', 'VLMC',
#         'Ast', 'Chandelier', 'End', 'Mic', 'Oli', 'Immune', 'OPC']
    
#     cell_regulons_dict = {}

#     for cell in CellOrder:
#         cell_regulons_dict[cell] = (
#             pd.read_csv(f"{path}/GRNs/{cell}_GRN.txt", sep='\t', usecols=['method','TF','TG', 'edgeWeight'])
#             .loc[lambda x: x['method']=='SCENIC', ['TF','TG', 'edgeWeight']]
#         )
    
#     GRNs_emani = (
#         pd.concat(cell_regulons_dict)
#         .reset_index(0, names='CellType')
#         .rename({'TG':'Gene', 'edgeWeight':'Importance'}, axis=1)
#     )

#     if unify_celltypes:
#         GRNs_emani = GRNs_emani.drop('CellType', axis=1).groupby(['TF','Gene'], as_index=False).mean().drop_duplicates()
    
#     return GRNs_emani


# def load_velmeshev(unify_lineages=True):
#     regulons = pd.read_csv("../velmeshev2023/velmeshev2023_S3.csv", usecols = ['TF', 'Gene', 'lineage']).drop_duplicates()
    
#     if unify_lineages:
#         regulons = regulons.loc[:, ['TF', 'Gene']].drop_duplicates()
     
#     return regulons
