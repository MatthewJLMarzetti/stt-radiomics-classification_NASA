# -*- coding: utf-8 -*-
"""
Pre-processing utilities for radiomic feature analysis.

Includes:
- Removal of unstable features identified through repeat segmentation.
- Univariate feature ranking using AUC and Mann-Whitney U tests.
- Visualisation of top-ranked features.


Created on Tue Nov 18 13:40:22 2025

@author: marzettm
"""

from sklearn.metrics import roc_auc_score
import numpy as np
from scipy.stats import mannwhitneyu
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# provide a text file with a list of features found to be unstabel on re-segmentation
def remove_unstable(X, unstable_features_list_file):
    """
    Remove features identified as unstable on repeat segmentation.
     
    Parameters
    ----------
    X : pandas.DataFrame
    Feature matrix.
     
    unstable_features_list_file : str
    Path to a text file containing one feature name per line.
    Features listed in the file will be removed if present.
     
    Returns
    -------
    pandas.DataFrame
    Feature matrix with unstable features removed.
     
    Notes
    -----
    Features not present in X are ignored.
    This allows the same exclusion file to be reused across datasets.
    """
        
    
    
    # Load excluded features from text file
    with open(unstable_features_list_file) as f:
        excluded_features = [line.strip() for line in f if line.strip()]
    
    print(f"Number of excluded features: {len(excluded_features)}")
    #print(f"Original number of features: {X.shape[-1]}")
    # Remove them from X (ignore errors in case some don't exist)
    X = X.drop(columns=excluded_features, errors="ignore")
    
    return X







# pre-processing univariate analysis
def pval_and_auc(X, y, n = 20):
    """
    Perform univariate feature analysis using ROC AUC and the
    Mann-Whitney U test.
     
    Each feature is evaluated independently against the binary target
    variable. Discriminative performance is quantified using the area
    under the receiver operating characteristic curve (AUC), while
    statistical differences between classes are assessed using a
    two-sided Mann-Whitney U test.
     
    Features with constant values or only missing values are excluded
    from the analysis.
     
    Parameters
    ----------
    X : pandas.DataFrame
    Feature matrix where columns correspond to individual features
    and rows correspond to samples.
     
    y : pandas.Series or array-like
    Binary ground-truth labels. Positive and negative classes should
    be encoded as 1 and 0, respectively.
     
    n : int, default=20
    Number of top-ranked features to display. Features are ranked
    primarily by Mann-Whitney U test p-value, with AUC used as a
    tie-breaker.
     
    Returns
    -------
    results_df : pandas.DataFrame
    DataFrame containing one row per feature with the following
    columns:
     
    - 'Feature' : feature name.
    - 'AUC' : univariate ROC AUC.
    - 'p_value' : two-sided Mann-Whitney U test p-value.
     
    Notes
    -----
    - AUC values are calculated independently for each feature and do
    not account for interactions between features.
    - Mann-Whitney U tests compare feature distributions between the
    two outcome groups without assuming normality.
    - Missing values are excluded on a feature-by-feature basis.
    - The function prints the top-ranked features and generates boxplots
    through `plot_top_univariate()`.
    - Statistical tests are uncorrected for multiple comparisons and are
    intended for exploratory feature assessment rather than formal
    hypothesis testing.
    """

    
    results = []
    
    for col in X.columns:
        feature_vals = X[col]
        label_vals = y
        
        # Skip features that are constant or all NaN
        if feature_vals.nunique() <= 1 or feature_vals.isna().all():
            continue
    
        # AUC (drop NaNs)
        try:
            auc_univariate = roc_auc_score(label_vals[~feature_vals.isna()], feature_vals[~feature_vals.isna()])
        except:
            auc_univariate = np.nan
    
        # Mann-Whitney U test
        group0 = feature_vals[label_vals == 0].dropna()
        group1 = feature_vals[label_vals == 1].dropna()
        try:
            stat, p = mannwhitneyu(group0, group1, alternative='two-sided')
        except:
            p = np.nan
    
        results.append({'Feature': col, 'AUC': auc_univariate, 'p_value': p})
    
    # Create DataFrame and sort
    results_df = pd.DataFrame(results)
    results_df_sorted = results_df.dropna().sort_values(by=['p_value','AUC'], ascending=[True,False])
    top_n = results_df_sorted.head(n)
    
    print(f"Top {n} features ranked by p-value (AUC used as tie-breaker):")
    print(top_n.to_string(index=False, formatters={'AUC': '{:.3f}'.format, 'p_value': '{:.1e}'.format}))
    
    plot_top_univariate(results_df, X, y)
    
    
    return results_df


def plot_top_univariate(p_val_df, X, y, n = 20):
    """
    Visualise the top-ranked univariate features using boxplots.
     
    Features are ranked according to the Mann-Whitney U test p-value,
    and the top n features are displayed as class-stratified boxplots.
    The corresponding univariate AUC and p-value are shown in each
    subplot title.
     
    Parameters
    ----------
    p_val_df : pandas.DataFrame
    Output from `pval_and_auc()`, containing at least the columns:
    - 'Feature'
    - 'AUC'
    - 'p_value'
     
    X : pandas.DataFrame
    Feature matrix.
     
    y : pandas.Series or array-like
    Binary ground-truth labels.
     
    n : int, default=20
    Number of top-ranked features to visualise.
     
    Returns
    -------
    None
     
    Notes
    -----
    Features are ranked using ascending Mann-Whitney U test p-value.
    Boxplots are generated using seaborn and displayed in a grid layout.
    """
    
    train_data = X.copy()
    train_data['Label'] = y
    
    p_val_df_sorted = p_val_df.dropna().sort_values(by=['p_value'], ascending=[True])
    top_n = p_val_df_sorted.head(n)
    # Prepare for plotting
    top_features = top_n['Feature'].tolist()
    plot_data = train_data[top_features + ['Label']]
    #melted = pd.melt(plot_data, id_vars='Label', var_name='Feature', value_name='Value')
    
    # Set up 5x4 subplot grid
    sns.set(style='whitegrid')
    
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20), sharex=False)
    axes = axes.flatten()
    
    # Loop through features
    for i, feature in enumerate(top_features):
        ax = axes[i]

        sns.boxplot(
            data=plot_data,
            x='Label',
            y=feature,
            hue='Label',        # explicitly assign hue
            ax=ax,
            palette='Set2',
            width=0.5,
            dodge=False,        # optional: prevents separation if you want boxes together
            legend=False        # don't show legend for each subplot
        )
        auc_val = top_n.loc[top_n['Feature'] == feature, 'AUC'].values[0]
        p_val = top_n.loc[top_n['Feature'] == feature, 'p_value'].values[0]
        ax.set_title(f"{feature}\nAUC={auc_val:.2f}, p={p_val:.1e}", fontsize=10)
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    # Remove empty subplots if fewer than 20
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()