import pandas as pd, numpy as np, seaborn as sns
import matplotlib.pyplot as plt

def plot_TFs_by_time(adata_projected, TFs_to_plot, size=(10,12),
                     timevar='Pseudotime', colorvar='Age'):

    cell_df = (adata_projected.obs[['Age_Num', 'Pseudotime', 'Cell_Type', 'Cell_Class']]
            #    .groupby('Cell_Class')['Pseudotime'].apply(lambda y: y*100/y.max())
            .assign(Pseudotime = lambda x: x.groupby('Cell_Class')['Pseudotime'].apply(lambda y: y*100/y.max()).reset_index(0, drop=True)) 
            .assign(Cell_Class = lambda x: pd.Categorical(x['Cell_Class'], ordered=True, categories=['Ex','In','Glia']))
            .sort_values(['Cell_Class', 'Pseudotime'])
            .assign(Cell_Type = lambda x: pd.Categorical(x['Cell_Type'], ordered=True, categories=x['Cell_Type'].unique()))
            .assign(Age = lambda x: np.log10(x['Age_Num']))
    )

    sns.set_theme(style="white") 
    pal = sns.color_palette('crest', as_cmap=True)
    age_ticks_years = [-0.5,0,1,2,5,18,50]

    g = (adata_projected.to_df()
    .loc[:, TFs_to_plot]
    .melt(ignore_index=False, var_name='TF', value_name='regulon_expression')
    .join(cell_df)
    .pipe((sns.relplot, 'data'), kind='scatter',
        x=timevar, y='regulon_expression', 
        # hue=colorvar, palette=pal,
        hue='Cell_Type', palette=sns.color_palette("tab10"),
        row='TF', col='Cell_Class',
        facet_kws={
            'sharey':False, 'legend_out':False, 'margin_titles':True},
        s=2)
    .set_titles(col_template="{col_name}", row_template="{row_name}")
    .set_ylabels("")
    # .set_axis_labels("Pseudotime", "Regulon Experession")
    )
    g.figure.set_size_inches(size)

    g.axes[0][0].get_legend().remove()
    g.tight_layout()



    # g.set(xticks=[np.log10(x*365+300) for x in age_ticks_years])
    # g.set_xticklabels(age_ticks_years)

    # norm = plt.Normalize(cell_df[colorvar].min(), cell_df[colorvar].max())
    # sm = plt.cm.ScalarMappable(cmap=pal, norm=norm)
    # cax = g.fig.add_axes([1.02, .12, .02, .8])
    # cax.set_title(colorvar)
    # cbar = g.figure.colorbar(sm, cax=cax)
    # cbar.set_ticks([np.log10(x*365+300) for x in age_ticks_years])
    # cbar.set_ticklabels(age_ticks_years)

    return(g)