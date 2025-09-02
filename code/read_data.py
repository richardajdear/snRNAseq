import numpy as np, pandas as pd
from ast import literal_eval

def read_velmeshev_meta(
        base_path="../velmeshev2023/cell_meta/",
):
    # Read metadata on velmeshev cells
    meta = (pd.concat({
                'Ex': pd.read_csv(f"{base_path}/ex_meta.tsv", sep='\t', low_memory=False).rename({'Age_(days)':'Age_Num'}, axis=1),
                'In': pd.read_csv(f"{base_path}/in_meta.tsv", sep='\t', low_memory=False).rename({'Age_(days)':'Age_Num', 'cellId':'Cell_ID'}, axis=1),
                'Macro': pd.read_csv(f"{base_path}/macro_meta.tsv", sep='\t'),
                'Micro': pd.read_csv(f"{base_path}/micro_meta.tsv", sep='\t').assign(Cell_Type = 'Microglia')
            })
            .reset_index(0, names='Cell_Class').set_index('Cell_ID')
            # Logic to assign 0 as birth
            .assign(Age_Years = lambda x: np.select(
                [
                    (x['Age'].str.contains('GW')) & (x['Age_Num'] > 268),
                    (~x['Age'].str.contains('GW')) & (x['Age_Num'] < 268)
                ],
                [-0.01,0],
                default = (x['Age_Num']-268)/365)
            )
            .assign(Cell_Class = lambda x: x['Cell_Class'].replace({'Macro':'Glia', 'Micro':'Glia'})) 
            .assign(Cell_Class = lambda x: pd.Categorical(x['Cell_Class'], ordered=True, categories=['Ex','In','Glia']))
            .assign(Cell_Type = lambda x: x['Cell_Type'].replace({'PV_MP':'PV', 'SST_RELN':'SST'}))
            .assign(Cell_Type = lambda x: pd.Categorical(x['Cell_Type'], ordered=True, categories=x['Cell_Type'].unique()))
            .assign(Cell_Lineage = lambda x: np.select(
                [
                    x['Cell_Class'] == 'Ex', 
                    x['Cell_Class'] == 'In',
                    x['Cell_Class'] == 'Glia'
                ],
                ['Excitatory', 'Inhibitory', x['Cell_Type']],
                default='Other'
            ))
            .assign(Cell_Lineage = lambda x: x['Cell_Lineage'].replace({'Fibrous_astrocytes':'Astrocytes', 'Protoplasmic_astrocytes':'Astrocytes'}))
            .assign(Cell_Lineage = lambda x: pd.Categorical(x['Cell_Lineage'], ordered=True, categories=['Excitatory', 'Inhibitory', 'Astrocytes', 'Oligos', 'OPC', 'Microglia', 'Glial_progenitors']))
            .assign(Age_Range2 = lambda x: pd.Categorical(np.where(
                    np.isin(x['Age_Range'], ['2nd trimester', '3rd trimester']), 
                    x['Age_Range'],
                    pd.cut((x['Age_Num']-273)/365, 
                            # bins=[-np.inf,1,2,5,9,16,25,np.inf],
                            # labels=['0-1','1-2','2-5','5-9','9-16','16-25','25+'])
                            bins=[-np.inf,1,2,9,18,25,np.inf],
                            labels=['0-1','1-2','2-9','9-18', '18-25','25+'])
                    ),
                ordered=True, 
                categories=['2nd trimester', '3rd trimester']+['0-1','1-2','2-9','9-18','18-25','25+'])
            )
            .assign(Age_Range3 = lambda x: pd.Categorical(np.select(
                    [
                        np.isin(x['Age_Range'], ['2nd trimester', '3rd trimester']), 
                        np.isin(x['Age_Range2'], ['0-1', '1-3'])
                    ],
                    [
                        x['Age_Range'],
                        pd.cut(x['Age_Num']-273, 
                                bins=[-np.inf,91,182,365,2*365,3*365],
                                labels=['0-3m','3m-6m','6m-1y', '1-2y', '2-3y']
                        )
                    ],
                    default = x['Age_Range2']
                ),
                ordered=True, 
                categories=['2nd trimester', '3rd trimester']+['0-3m','3m-6m','6m-1y', '1-2y', '2-3y']+['3-9','9-18','18-25','25+'])
            )
            .assign(Age_Range4 = lambda x: pd.Categorical(np.select(
                    [
                        np.isin(x['Age_Range'], ['2nd trimester', '3rd trimester']), 
                        x['Age_Years'] < 1,
                        x['Age_Years'] < 9,
                        x['Age_Years'] < 25,
                        x['Age_Years'] >= 25
                    ],
                    [
                        'Prenatal',
                        'Infancy',
                        'Childhood',
                        'Adolescence',
                        'Adulthood'
                    ],
                    default = x['Age_Range']
                ),
                ordered=True,
                categories = ['Prenatal', 'Infancy', 'Childhood', 'Adolescence', 'Adulthood']
            ))
            # .assign(Age_Years = lambda x: (x['Age_Num']-273)/365)
            .assign(Age_log2 = lambda x: np.log2(1 + x['Age_Years'] ) )
            .assign(Age_log10 = lambda x: np.log10(1 + x['Age_Years'] ) )
            .assign(Age_Postnatal = lambda x: ~np.isin(x['Age_Range'], ['2nd trimester', '3rd trimester']))
            .assign(Age_Postinfant = lambda x: x['Age_Years'].fillna(0)>=2)
            .assign(Individual = lambda x: x['Individual'].astype('str'))
            .assign(Pseudotime_pct = lambda x: x.groupby('Cell_Class')['Pseudotime'].apply(lambda y: y*100/y.max()).reset_index(0, drop=True)) # 
            .drop('PMI', axis=1)
        )

    return(meta)


def read_scenic_regulons(path, explode=True, unify=True):
    regulons = (pd.read_csv(path, header=1)
        .transform(lambda x: x.set_axis(['TF', 'MotifID'] + list(x.columns[2:]), axis=1)).iloc[1:,:]
    )
     
    if not explode:
        return regulons

    regulons_explode = (regulons.loc[:,['TF','TargetGenes']]
        .assign(TargetGenes = lambda x: x['TargetGenes'].apply(literal_eval))
        .explode('TargetGenes')
        .assign(
            Gene = lambda x: x['TargetGenes'].str[0],
            Importance = lambda x: x['TargetGenes'].str[1]
        )
        .drop('TargetGenes', axis=1)
    )

    if unify:
        # return regulons_explode.drop('score', axis=1).drop_duplicates()
        return regulons_explode.groupby(['TF','Gene'], as_index=False).mean().drop_duplicates()
    else:
        return regulons_explode.drop_duplicates()
    

def read_scenic_aucell(path):
    aucell = (pd.read_csv(path, index_col='Cell')
              .rename(columns = lambda x: x.replace('(+)',''))
            )
    return(aucell)


# Complicated function to filter gene importance and return file in same format
# from pyscenic.utils import load_motifs
# def filter_regulon_gene_importance(f_input_regulons, percentile=0.5, return_df=False):
#     # This function takes a filepath with the output from pyscenic ctx
#     # and filters the regulon links by a percentile of the gene's importance

#     # Load the motifs df from the input file
#     df = load_motifs(f_input_regulons)

#     # Explode the list of (Gene, Importances) tuples
#     # Filter by percentile of the importances
#     # Recombine back into a list of tuples for each motif
#     targetGenes = df[('Enrichment','TargetGenes')].explode()
#     targetGenes_filtered = (
#         pd.DataFrame([*targetGenes], 
#                      index=targetGenes.index, 
#                      columns=['Gene', 'Importance'])
#             .assign(percentile = lambda x: x['Importance'].rank(pct=True))
#             .loc[lambda x: x['percentile']>percentile]
#             .assign(tuple = lambda x: list(zip(x['Gene'], x['Importance'])))
#             .groupby(level=[0,1])
#             .agg({'tuple': lambda x: repr(x.to_list())})
#     )

#     # Replace the column in the original df with the filtered lists in object type
#     df[('Enrichment','TargetGenes')] = targetGenes_filtered['tuple']
#     # Drop motifs that no longer have any genes
#     df.dropna(inplace=True)

#     # And save to a new csv
#     f_filtered = f_input_regulons.split('_reg')[0] + f'_reg-imp{int(percentile*100)}.csv'
#     df.to_csv(f_filtered)
#     print(f"Filtered TF-Gene links for percentile {percentile} and saved to {f_filtered}")
#     if return_df: 
#         return(df)
#         # return(targetGenes_filtered)