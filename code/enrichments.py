import numpy as np, pandas as pd
import subprocess
import os
from scipy.stats import percentileofscore
from scipy.stats import fisher_exact, f_oneway
from statsmodels.stats.multitest import multipletests
import mygene

def shuffle_weights(weights, n=100, rank=False):
    """
    Make null model by randomizing gene weights / ranks
    """
    n_components=weights.shape[1]
    null_weights = np.repeat(weights.values[:,:n_components, np.newaxis], n, axis=2)
    # null_weights = np.take_along_axis(null_weights, np.random.randn(*null_weights.shape).argsort(axis=0), axis=0)
    for c in range(n_components):
        for i in range(n):
            np.random.shuffle(null_weights[:,c,i])
    
    if rank:
        null_ranks = null_weights.argsort(axis=0).argsort(axis=0)
        return null_ranks
    else:
        return null_weights


def match_genes(gene_labels, weights, label_column='label', gene_column='gene'):
    """
    Make mask of which genes from each gene list are in which components
    """
    genes = weights.index
    gene_masks = {}
    gene_counts = {}
    for l in gene_labels[label_column].unique():
        genes_in_label = gene_labels.loc[lambda x: x[label_column] == l, gene_column]
        matches = np.isin(genes, genes_in_label)
        if sum(matches)>0:
            gene_masks[l] = matches
        
        gene_counts[l] = pd.Series({
            'n_genes': len(genes_in_label),
            'n_matches': sum(matches)
        })
    gene_counts = pd.concat(gene_counts).unstack()
    
    return gene_masks, gene_counts



def compute_enrichments(weights, null_weights, gene_labels, 
                        how='mean', norm=False, posneg=None, **kwargs):
    """
    Compute scores for each gene label, either mean, or median rank
    """
    n_components = weights.shape[1]
    component_names = list(weights.columns)
    gene_masks, gene_counts = match_genes(
        gene_labels, weights, 
        label_column = kwargs.get('label_column','label'),
        gene_column = kwargs.get('gene_column','gene')
        )
    
    weights = weights.copy().values
    nulls = null_weights.copy()
    # Take absolute values of standardized weights
    if norm:
        weights = StandardScaler().fit_transform(weights)
        for i in range(nulls.shape[2]):
            nulls[:,:,i] = StandardScaler().fit_transform(nulls[:,:,i])
            
    if posneg =='abs':
        weights = np.abs(weights)
        nulls = np.abs(nulls)
    elif posneg=='pos':
        weights = np.where(weights<0, np.nan, weights)
        nulls = np.where(nulls<0, np.nan, nulls)
    elif posneg=='neg':
        weights = np.where(weights>0, np.nan, weights)
        nulls = np.where(nulls>0, np.nan, nulls)

    true_enrichments = {}
    null_enrichments = {}
    
    for label, mask in gene_masks.items():
        if how == 'mean':
            true_enrichments[label] = pd.Series(np.nanmean(weights[mask, :], axis=0))
            null_enrichments[label] = pd.DataFrame(np.nanmean(nulls[mask, :, :], axis=0)).T
        elif how == 'median': #### not working
            true_ranks = weights.argsort(0).argsort(0)
            true_enrichments[label] = pd.Series(np.nanmedian(true_ranks[mask, :], axis=0))
            null_enrichments[label] = pd.DataFrame(np.nanmedian(nulls[mask, :, :], axis=0)).T

    true_enrichments = pd.concat(true_enrichments).unstack(1).set_axis(component_names, axis=1)
    null_enrichments = pd.concat(null_enrichments).set_axis(component_names, axis=1).reset_index(level=0).rename({'level_0':'label'}, axis=1)

    return true_enrichments, null_enrichments, gene_counts

def compute_null_p(true_enrichments, null_enrichments, 
                   gene_counts=None, adjust='fdr_bh', adjust_by_label=False,
                   order=None):
    """
    Compute null p values
    """
    null_pct = np.zeros(true_enrichments.shape)
    for m, label in enumerate(true_enrichments.index):
        for i in range(true_enrichments.shape[1]):
            nulls_ = null_enrichments.set_index('label').loc[label].iloc[:,i]
            true_ = true_enrichments.iloc[m, i]
            pct = percentileofscore(nulls_, true_)/100
            null_pct[m, i] = pct
            
    true_mean = true_enrichments.stack().rename('true_mean')

    null_mean = (null_enrichments
                 .groupby('label').agg(['mean','std'])
                 .stack(0, future_stack=True)
                 .rename_axis([None,None])
                 .set_axis(['null_mean', 'null_std'], axis=1)
                )
            
    null_p = (pd.DataFrame(null_pct,
                           index=true_enrichments.index,
                           columns=true_enrichments.columns)
              .stack(future_stack=True).rename('pct').to_frame()
              .join(true_mean)
              .join(null_mean)
              .assign(z = lambda x: (x['true_mean'] - x['null_mean'])/x['null_std'])
              .assign(pos = lambda x: [pct > 0.5 for pct in x['pct']])
              .assign(p = lambda x: [(1-pct)*2 if pct>0.5 else pct*2 for pct in x['pct']]) # x2 for two-sided
             )
    
    # Apply multiple comparisons
    if adjust is not None:
        # Adjust across components only (not by label)?
        if adjust_by_label:
            null_p = (null_p
                .assign(q = lambda x: x.groupby(level=0)
                                       .apply(lambda y: pd.Series(multipletests(y['p'], method=adjust)[1], index=y.index))
                                       .reset_index(0, drop=True) # make index match
                                       )
                .assign(sig = lambda x: x['q'] < .05)
                # .assign(q_abs = lambda x: [1-q if pos else q for pos, q in zip(x['pos'], x['q'])])
                )
        else:
            null_p = (null_p
                .assign(q = lambda x: multipletests(x['p'], method=adjust)[1])
                .assign(sig = lambda x: x['q'] < .05)
                # .assign(q_abs = lambda x: [1-q if pos else q for pos, q in zip(x['pos'], x['q'])])
                )
    else:
        null_p = (null_p
             .assign(q = lambda x: x['p'])
             .assign(sig = lambda x: x['q'] < .05)
            )
    
    null_p = (null_p
              .reset_index()
              .rename({'level_0':'label', 'level_1':'C'}, axis=1)
             )
    
    if gene_counts is not None:
        null_p = null_p.join(gene_counts, on='label')
    
    # Fix order of gene labels
    if order is None:
        order = true_enrichments.index
    null_p = (null_p
              .assign(label = lambda x: pd.Categorical(x['label'], ordered=True, categories=order))
              .sort_values('label')
         )

    return null_p




### GSEA
import pandas as pd
import gseapy
from typing import Dict

def run_go_enrichment(df: pd.DataFrame, background=None) -> Dict[str, pd.DataFrame]:
    """
    Performs Gene Ontology (GO) Biological Process enrichment analysis for each
    gene network defined in the input DataFrame.

    This function uses the 'gseapy' library to run Enrichr analysis.

    Args:
        df (pd.DataFrame): A DataFrame with two columns: 'Network' and 'Gene'.
                           'Network' should be the identifier for each gene set,
                           and 'Gene' should be the gene symbol.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary where keys are the network names
                                 and values are pandas DataFrames containing the
                                 GO enrichment results for that network.
                                 Returns an empty dictionary if input is invalid.
    """
    # Validate the input DataFrame
    if df.empty or 'Network' not in df.columns or 'Gene' not in df.columns:
        print("Error: Input DataFrame is empty or missing 'network' or 'gene' columns.")
        return {}

    # Create a dictionary to store the results
    enrichment_results = {}

    # Group the DataFrame by network and iterate through each group
    for network_name, group_df in df.groupby('Network'):
        # Get the list of genes for the current network
        gene_list = group_df['Gene'].tolist()

        # Check if there are enough genes to run the analysis
        if not gene_list:
            print(f"Warning: network '{network_name}' is empty. Skipping analysis.")
            continue

        print(f"Running GO enrichment for network: '{network_name}' with {len(gene_list)} genes.")

        try:
            # Perform the enrichment analysis using Enrichr
            # We use 'GO_Biological_Process_2023' as the gene set library.
            # Adjust the library name as needed for different versions or types.
            enr = gseapy.enrichr(gene_list=gene_list,
                                #  gene_sets='GO_Biological_Process_2023',
                                 gene_sets='../data/c5.go.bp.v2025.1.Hs.symbols.gmt',
                                 background=background,
                                 organism='Human', # You can change this to 'Mouse', etc.
                                 outdir=None, # Set to a path if you want to save the results to a file
                                 cutoff=0.05) # p-value cutoff

            # Store the enrichment results DataFrame
            enrichment_results[network_name] = enr.results
        except Exception as e:
            print(f"An error occurred during enrichment for network '{network_name}': {e}")
            continue

    return enrichment_results





#### MAGMA

def symbol2entrez(df, symbol_column='Gene'):
    """
    Map symbols to ensembl IDs, exploding duplicates
    """
    mg = mygene.MyGeneInfo()
    mg_out = mg.querymany(df[symbol_column].unique(),
                        scopes='symbol', 
                        fields='entrezgene',
                        species='human',
                        as_dataframe=True, 
                        df_index=False)

    return(df.join(mg_out.set_index('query')['entrezgene'], on=symbol_column))


def symbol2ensembl(df, symbol_column='Gene'):
    """
    Map symbols to ensembl IDs, exploding duplicates
    """
    mg = mygene.MyGeneInfo()
    mg_out = mg.querymany(df[symbol_column].unique(),
                        scopes='symbol', 
                        fields='ensembl.gene', 
                        species='human', 
                        as_dataframe=True, 
                        df_index=False)
    mg_explode = (mg_out
        .explode('ensembl')
        .reset_index(drop=True)
        .assign(ensembl = lambda x: pd.json_normalize(x['ensembl']))
        .assign(ensembl = lambda x: x['ensembl'].fillna(x['ensembl.gene']))
        .set_index('query')['ensembl']
    )
    return(df.join(mg_explode, on=symbol_column))



def get_magma_geneset_enrichments(input_table, 
                                   input_name='network_regulons',
                                   gwas_name='SCZ',
                                   gwas_dir='',
                                   gwas_map_name='hmagma_adult',
                                   col='1,2',
                                   method='geneset',
                                   verbose=False):
    # First write the input table to file
    input_file_path = f'../tools/MAGMA/inputs/input_{input_name}.txt'
    input_table.to_csv(input_file_path, sep=' ', index=False)

    # Then define the files to be given to the magma function
    input_file = f'inputs/input_{input_name}.txt'
    gene_results_file = os.path.join('gene_analysis/', gwas_dir, f'{gwas_name}.{gwas_map_name}.genes.raw')
    output_prefix = f'outputs/{input_name}.{gwas_name}.{gwas_map_name}'

    if verbose:
        print(f'''
        Running magma enrichment test
            method: {method}
            input file to test: {input_file}
            gene mapping: {gene_results_file}
            outputs saved to: {output_prefix}.gsa.out
        ''')

    # Combine the filenames into the magma input command
    if method=='geneset':
        input_command = f'''
        cd ../tools/MAGMA
        ./magma \\
        --set-annot {input_file} col={col} \\
        --gene-results {gene_results_file} \\
        --out {output_prefix}
        '''
    elif method=='covar':
        input_command = f'''
        cd ../tools/MAGMA
        ./magma \\
        --gene-covar {input_file} missing-values="drop" max-miss=0.2 \\
        --gene-results {gene_results_file} \\
        --out {output_prefix}
        '''
        
    # Send the command to magma
    process = subprocess.run(["bash", "-c", input_command], 
                             stdout=subprocess.PIPE, 
                             stderr=subprocess.PIPE, 
                             text=True)

    if verbose:
        print(process.stdout)
        print(process.stderr)

    # Read the magma results back to a DataFrame
    results_file_path = f'../tools/MAGMA/{output_prefix}.gsa.out'
    test_read_skip4 = pd.read_csv(results_file_path, skiprows=4, nrows=1, header=None, sep='\s+').loc[0,0]
    test_read_skip3 = pd.read_csv(results_file_path, skiprows=3, nrows=1, header=None, sep='\s+').loc[0,0]
    if test_read_skip4 == 'VARIABLE':
        results = pd.read_csv(results_file_path, skiprows=4, sep='\s+')
    elif test_read_skip3 == 'VARIABLE':
        results = pd.read_csv(results_file_path, skiprows=3, sep='\s+')
    else:
        raise ValueError(f"Unexpected format in {results_file_path}. Check the file structure.")
    
    return results


def GRN_magma_enrichment(GRN, mapping=None, method='geneset'):
    """
    Perform MAGMA enrichment analysis for a given gene regulatory network (GRN).
    """
    # Map gene symbols to entrez IDs
    if mapping is not None:
        network_magma_input = GRN.join(mapping.set_index('Gene'), on='Gene', how='inner')
    else:
        network_magma_input = (GRN
               .pipe(symbol2entrez)
               .loc[lambda x: x['entrezgene'].notnull()]
            #    .pipe(symbol2ensembl)
            #    .loc[lambda x: x['ensembl'].notnull()]
    )

    # Get MAGMA enrichments
    if method=='geneset':
        # Select columns for magma geneset
        network_magma_input = network_magma_input.loc[:, ['Network', 'entrezgene']].drop_duplicates()

        network_magma_SCZ = get_magma_geneset_enrichments(network_magma_input, 
            input_name = 'GRN',
            gwas_name = 'SCZ',
            gwas_map_name = 'magma',
            col='2,1'
        )
        network_magma_MDD = get_magma_geneset_enrichments(network_magma_input, 
            input_name = 'GRN',
            gwas_name = 'MDD2025',
            gwas_map_name = 'magma',
            col='2,1'
            )
    elif method=='covar':
        # Pivot to format for magma covar
        network_magma_input = (network_magma_input
            .pivot(index='entrezgene', columns='Network', values='Importance').reset_index()
        )

        network_magma_SCZ = get_magma_geneset_enrichments(network_magma_input, 
            input_name = 'GRN',
            gwas_name = 'SCZ',
            gwas_map_name = 'magma',
            method = 'covar'
        )
        network_magma_MDD = get_magma_geneset_enrichments(network_magma_input, 
            input_name = 'GRN',
            gwas_name = 'MDD2025',
            gwas_map_name = 'magma',
            method = 'covar'
        )

    network_magma_results = (pd.concat({
            'SCZ': network_magma_SCZ, 
            'MDD': network_magma_MDD
        })
        .reset_index(0, names='enrichment')
        .assign(Q = lambda x: multipletests(x['P'], method='fdr_bh')[1])
        .rename({'VARIABLE':'Network', 'BETA':'z', 'P':'p', 'Q':'q'}, axis=1)
    )

    out = (network_magma_results
    .assign(q = lambda x: x.groupby('enrichment')['p'].transform(lambda y: multipletests(y, method='fdr_bh')[1]))
    .sort_values(['enrichment','p'])
    )

    return out