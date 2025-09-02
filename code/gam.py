import numpy as np, pandas as pd
from statsmodels.gam.api import GLMGam, BSplines


def fit_cell_curves(data, genes_to_fit, cells_to_fit, alpha=1, cell_field='Cell_Type'):
    cell_preds = {}
    for cell in cells_to_fit:
        data_cell = data.query(f"{cell_field} == '{cell}'")
        models = fit_gam_models(data_cell, genes_to_fit = genes_to_fit, alpha=alpha)
        cell_preds[cell] = predict_gam_curves(models, data_cell, genes_to_fit = genes_to_fit).rename({'gene':'TF'}, axis=1)
    preds = pd.concat(cell_preds).reset_index(0, names=cell_field)
    return(preds)
        


def fit_gam_models(data, genes_to_fit, value_var='aucell', age_var='Age_log2', df=12, degree=3, alpha=1.0):
    """
    Fit GAM for each gene
    """
    models = {}
    for gene in genes_to_fit:
        data_gene = data.query(f"TF == '{gene}'") 
        # Add this because prediction at 0 fails when missing cells at 0 age
        MIN_AGE_DUMMY = data_gene.iloc[[0]].assign(Age_log2 = 0)
        data_gene = pd.concat([data_gene, MIN_AGE_DUMMY])

        spline_x = data_gene[age_var]
        basis_splines = BSplines(spline_x, df=df, degree=degree)
        alpha=alpha

        formula = f'{value_var} ~ {age_var} + Sex + Region'
        models[gene] = GLMGam.from_formula(formula=formula, data=data_gene, 
                                        smoother=basis_splines, alpha=alpha).fit()
        
    return models


def predict_gam_curves(models, data, genes_to_fit,
                       region='BA9', age_var='Age_log2', n_preds=100):
    """
    Predict GAM curves for a single region and both genders
    """
    # Vector of ages to predict at
    ages_to_predict = np.linspace(0, max(data[age_var]), n_preds)
    # Dataframe to predict
    df_preds = pd.DataFrame({
        age_var: ages_to_predict,
        'Sex': 'Female',
        'Region': region
    })

    # Make predictions for each gene
    preds = {gene: models[gene].predict(df_preds, exog_smooth=df_preds[age_var]) for gene in genes_to_fit}
    # Combine genes into df
    df_preds = (df_preds
                .join(pd.concat(preds, axis=1))
                .melt(id_vars=[age_var, 'Sex', 'Region'], var_name='gene', value_name='pred')
                # Add gene predictions normalised to 75th quantile
                # .assign(pred_q75=lambda x: x.groupby(['gene','gender','region'])
                #         .apply(lambda y: y['pred']/np.quantile(y['pred'],.75)).reset_index([0,1,2], drop=True))
                # .assign(age = lambda x: (10**x['Age_log2']-40*7)/365)
    )   
    return df_preds
