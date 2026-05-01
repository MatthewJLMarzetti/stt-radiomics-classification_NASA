# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 14:59:55 2026


Purpose:
--------
This script analyses ADC and IVIM parametric maps for a prospective
soft-tissue tumour cohort.

Specifically, it:
- Loads ADC and IVIM parameter maps (D, f, D*)
- Computes lesion-wise summary statistics using binary tumour masks
- Performs group-wise statistical comparisons (Benign vs Non-Benign)
- Computes ROC AUC values for each summary statistic
- Saves extracted features for downstream radiomics / modelling

This code is intended for reproducibility and transparency for the PhD
thesis and is not written as a general-purpose software package.



@author: marzettm
"""



# imports
import os
import pandas as pd
import itk
import sys
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu
from sklearn.metrics import roc_auc_score

# local helper functions
sys.path.append(r"F:\PhD\Prospective Study\Python helper scripts")
from utilityFunctions import display, display_masks
#from confidenceinterval import roc_auc_score


clinical_data = pd.read_csv(r"D:\Radiomics\RetrospectiveData\Test_sets\Prospective\Registered_T1_T2FS\prospective_test_info_final.csv")


quant_folder_base = r"D:\ProspectiveData\Analysis\IVIM_Maps"
ADC_path = r"D:\ProspectiveData\RegisteredImagesForSeg\ADC_p2"

results = []




def masked_stats(map_arr, mask_arr):

    """
    Computes robust summary statistics of a parametric map
    within a binary lesion mask.

    Parameters
    ----------
    map_arr : np.ndarray (3D)
        Quantitative parametric map (e.g. ADC, D, f, D*).
    mask_arr : np.ndarray (3D, boolean or 0/1)
        Binary lesion mask in the same space as map_arr.

    Returns
    -------
    dict
        Dictionary of summary statistics intended to be robust
        to outliers and noise:
        - mean
        - standard deviation
        - median
        - interquartile range (IQR)
        - 5th and 95th percentiles

    Notes
    -----
    - Non-finite values (NaN, inf) are removed before computation.
    - Min/max are intentionally excluded due to sensitivity to noise.
    """

    vals = map_arr[mask_arr > 0]

    # safety
    vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        return {
            #min': np.nan,
            'p05': np.nan,
            'median': np.nan,
            'mean': np.nan,
            'sd': np.nan,
            'IQR': np.nan,
            'p95': np.nan,
            #'max': np.nan
        }

    return {
        'mean': np.nanmean(vals),
        'sd': np.nanstd(vals),
        #'min': np.nanmin(vals),
        'median': np.nanmedian(vals),
        'IQR': np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25),
        'p05': np.nanpercentile(vals, 5),
        'p95': np.nanpercentile(vals, 95),
        #'max': np.nanmax(vals)
    }






for index, row in clinical_data.iterrows():

    
    patient_ID = row['PseudoPatientID']
    
    
    print(patient_ID)
    
    if patient_ID == "NASA137": # No DWI signal, D is all negative as a result of noise
        continue

        

        
    map_base_path = os.path.join(quant_folder_base, patient_ID + "_p2", "bi_exp_segmented","segmented_fit_raw")
    
    
    D_map_path = os.path.join(map_base_path, "segmented_D.npy")
    f_map_path = os.path.join(map_base_path, "segmented_f.npy")
    Dstar_map_path = os.path.join(map_base_path, "segmented_Dstar.npy")
    ADC_map_path = os.path.join(ADC_path, patient_ID + "_ADC_reg_p2.nii")
    
    
    D_map = np.load(D_map_path)
    f_map = np.load(f_map_path)
    Dstar_map = np.load(Dstar_map_path)
    ADC_map = itk.imread(ADC_map_path)
    ADC_map = itk.array_from_image(ADC_map)
    
    # Ensure consistent orientation between NPY IVIM maps and NIfTI ADC maps
    D_map = np.transpose(D_map, (2, 1, 0))
    f_map = np.transpose(f_map, (2, 1, 0))
    Dstar_map = np.transpose(Dstar_map, (2, 1, 0))
    
    
    
    # Enforce physical constraints on fitted IVIM parameters
    # (negative values arise from noise / fitting instability)

    D_map[D_map < 0] = 0
    f_map[f_map < 0] = 0
    f_map[f_map > 1] = 1 # > 1 turned into 2, < 0 turned to -1
    # Remove implausible D* values where fitting is unstable
    Dstar_map[Dstar_map < 0] = 0 #turn any -ve values into nan
    Dstar_map[f_map>= 1] = np.nan # remove any values were f > 1 as this means D in noise floor
    Dstar_map[f_map <0.001] = np.nan # remove any hard to fit values where f is tiny
    Dstar_map[Dstar_map > 0.1] = np.nan # These are silly values where D* just vanishes immediately
    
    print(D_map.shape)
    print(ADC_map.shape)
    
    
    try:
        mask_path = os.path.join(r"D:\ProspectiveData\Analysis\Masks\Final_masks", patient_ID+".nii.gz")
        mask = itk.imread(mask_path)
    except RuntimeError:
        print(f"No mask for {patient_ID}")
        continue

    mask = itk.array_from_image(mask)
    mask = mask > 0
    
    # Handle known slice offset mismatch for this case (manual correction)    
    if patient_ID == "NASA001": 
        mask = mask[1:,:,:]
        ADC_map = ADC_map[1:,:,:]
        
    ADC_map = ADC_map / 1E6 # Siemens scale ADC maps, so putting on same scale as my ones
        
    
    
    maps = {
        'ADC': ADC_map,
        'D': D_map,#itk.array_from_image(D_map),
        'f': f_map,#itk.array_from_image(f_map),
        'Dstar': Dstar_map,#itk.array_from_image(Dstar_map),

    }
    
    row_out = {
        'PseudoPatientID': patient_ID,
    
        # clinical columns
        'WHO_category': row['WHO_category'],
        'subtype_grouped': row['subtype_grouped'],
        'Grade': row['Grade'],
        'Any Pathology': row['Any Pathology'],
        'final_dx': row['final_dx'],
    }
    
        # --- Compute stats ---
    for param, arr in maps.items():
        stats = masked_stats(arr, mask)
        for stat_name, value in stats.items():
            row_out[f"{param}_{stat_name}"] = value
    
    results.append(row_out)
    
    
    
    if patient_ID == "NASA137":
        print(mask.sum())
        
        print(mask.shape)
        print(D_map.shape)
        print(f_map.shape)
        print(Dstar_map.shape)
        
        display_masks(D_map[11,:,:], mask[11,:,:], im_min = 0, im_max = 0.01)
        display_masks(f_map[11,:,:], mask[11,:,:])
        display_masks(Dstar_map[11,:,:], mask[11,:,:])
        
        
ivim_summary_df = pd.DataFrame(results)

ivim_summary_df_no_lipoma = ivim_summary_df[ivim_summary_df['subtype_grouped'] != "Lipoma"]






def plot_B_nonB(map_type, title):
    
    # plot D
    plt.figure(figsize=(10, 8))
    
    # Boxplot first (background)
    sns.boxplot(
        data=ivim_summary_df_no_lipoma,
        x='final_dx',
        y= map_type,
        whis=[5, 95],        # Optional: show 5th-95th percentile as whiskers
        color='lightgray',
        showfliers=False     # Don’t show points for boxplot itself
    )
    
    # Overlay the individual points
    sns.stripplot(
        data=ivim_summary_df_no_lipoma,
        x='final_dx',
        y= map_type,
        jitter=True,
        alpha=0.7,
        color='blue'
    )
    
    
    plt.xlabel('Final diagnosis')
    plt.ylabel(title)
    plt.title(f'{title} by final diagnosis', fontsize = 18)
    #plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


plot_B_nonB(map_type = 'ADC_mean', title = "Mean ADC")
plot_B_nonB(map_type = 'D_mean', title = "Mean D")
plot_B_nonB(map_type = 'f_mean', title = "Mean f")
plot_B_nonB(map_type = 'Dstar_mean', title = "Mean D*")




def plot_ivim_grid_pretty(df, param_stats=['mean','mean','mean','mean'],
                          param_names=['D','f','Dstar','f*Dstar'],
                          titles=['D','f','D*','f*D*'],
                          figsize=(12,10),
                          wspace=0.3, hspace=0.3):
    """
    Publication-ready 2x2 grid IVIM plots with units and nicer aesthetics.
    Scaling: D, D* and f*D* x1000 (x10^-3 mm2/s), f x100 (%)
    """

    df = df.copy()
    
    # Compute f*D* if requested
    if 'f*Dstar' in param_names:
        df['f*Dstar_mean'] = df['f_mean'] * df['Dstar_mean']
        df['f*Dstar_median'] = df['f_median'] * df['Dstar_median']
        df['f*Dstar_p05'] = df['f_p05'] * df['Dstar_p05']
        df['f*Dstar_p95'] = df['f_p95'] * df['Dstar_p95']
    
    # Scale values for plotting
    for param in param_names:
        for stat in ['mean','median','p05','p95']:
            col = f"{param}_{stat}"
            if col in df.columns:
                if param in ['D','Dstar','f*Dstar']:
                    df[col] = df[col]*1000
                elif param == 'f':
                    df[col] = df[col]*100
    
    # Rename x-axis for clarity
    df['final_dx_plot'] = df['final_dx'].map(lambda x: 'Benign' if x in [0,'0','Benign'] else 'Non-Benign')
    
    fig, axes = plt.subplots(2,2, figsize=figsize)
    axes = axes.flatten()
    
    # Define units   
    units = {
        'D': r"$\times 10^{-3}$ mm²/s",
        'f': '%',
        'Dstar': r"$\times 10^{-3}$ mm²/s",
        'f*Dstar': r"$\times 10^{-3}$ mm²/s"
    }
    
    
    # Aesthetic parameters
    sns.set(style="whitegrid")
    strip_color = '#0072B2'  # nice blue
    strip_alpha = 0.6
    strip_size = 8
    title_fontsize = 20
    label_fontsize = 16
    tick_fontsize = 14
    
    for i, (param, stat, title) in enumerate(zip(param_names, param_stats, titles)):
        y_col = f"{param}_{stat}"
        
        order = ['Benign', 'Non-Benign']  # force order
        
        sns.boxplot(
            data=df,
            x='final_dx_plot',
            y=y_col,
            whis=[5,95],
            color='lightgray',
            showfliers=False,
            ax=axes[i],
            order=order
        )
        sns.stripplot(
            data=df,
            x='final_dx_plot',
            y=y_col,
            jitter=True,
            alpha=strip_alpha,
            color=strip_color,
            size=strip_size,
            ax=axes[i],    
            order=order
        )
        
        axes[i].set_title(f'{title} ({stat})', fontsize=title_fontsize)
        axes[i].set_xlabel('')
        axes[i].set_ylabel(f"{title} [{units.get(param,'')}]",
                           fontsize=label_fontsize)
        axes[i].tick_params(axis='x', labelsize=tick_fontsize)
        axes[i].tick_params(axis='y', labelsize=tick_fontsize)
        
        axes[i].set_ylim(bottom=0)
    
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.show()




# display plots






plot_ivim_grid_pretty(
    ivim_summary_df_no_lipoma,
    param_stats=['mean','mean','mean','mean'],
    param_names=['D','f','Dstar','f*Dstar'],
    titles=['Mean D','Mean f','Mean D*',r'Mean f$\times$D*'],
    figsize=(16,10), hspace = 0.5
)




def plot_ADC_pretty(df, stat='mean', figsize=(12,8)):
    """
    Publication-ready ADC plot (Benign vs Non-Benign)
    Scaling: x10^-3 mm^2/s
    """
    
    df = df.copy()
    
    # --- Scale ADC ---
    col = f'ADC_{stat}'
    df[col] = df[col] * 1000  # convert to x10^-3 mm^2/s
    
    # --- Map labels ---
    df['final_dx_plot'] = df['final_dx'].map(
        lambda x: 'Benign' if x in [0,'0','Benign'] else 'Non-Benign'
    )
    
    # --- Aesthetics ---
    sns.set(style="whitegrid")
    strip_color = '#0072B2'
    strip_alpha = 0.6
    strip_size = 8
    
    title_fontsize = 28
    label_fontsize = 18
    tick_fontsize = 18
    
    plt.figure(figsize=figsize)
    
    order = ['Benign', 'Non-Benign']
    
    # --- Boxplot ---
    sns.boxplot(
        data=df,
        x='final_dx_plot',
        y=col,
        whis=[5,95],
        color='lightgray',
        showfliers=False,
        order=order
    )
    
    # --- Scatter overlay ---
    sns.stripplot(
        data=df,
        x='final_dx_plot',
        y=col,
        jitter=True,
        alpha=strip_alpha,
        color=strip_color,
        size=strip_size,
        order=order
    )
    
    plt.title(f'ADC ({stat})', fontsize=title_fontsize, pad = 30)
    plt.xlabel('')
    plt.ylabel(r'ADC [$\times 10^{-3}$ mm$^2$/s]', fontsize=label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    plt.show()


plot_ADC_pretty(ivim_summary_df_no_lipoma, stat='mean')






def compare_groups(df, group_col, value_col):

    """
    Compares a quantitative parameter between two groups.

    The function:
    - Checks normality (Shapiro-Wilk) per group
    - Uses Welch t-test / Student t-test where assumptions are met
    - Otherwise defaults to Mann–Whitney U
    - Computes ROC AUC as a measure of discriminative ability

    Notes
    -----
    - NaNs are dropped prior to analysis
    - AUC values < 0.5 are flipped for interpretability
    """

    # Drop rows with NaN in either column
    df_clean = df[[group_col, value_col]].dropna()
    
    groups = df_clean[group_col].unique()
    
    if len(groups) != 2:
        raise ValueError("AUC can only be computed for two groups.")

    # Split the data
    data = {
        g: df_clean.loc[df_clean[group_col] == g, value_col]
        for g in groups
    }

    # --- Normality per group ---
    normality = {
        g: shapiro(vals)[1] if len(vals) >= 3 else np.nan
        for g, vals in data.items()
    }

    # --- Variance equality ---
    levene_p = levene(*data.values()).pvalue

    # --- Choose test ---
    if all(p > 0.05 for p in normality.values()):
        test_name = "t-test (Welch)" if levene_p < 0.05 else "t-test"
        stat, pval = ttest_ind(
            *data.values(),
            equal_var=levene_p >= 0.05,
            nan_policy='omit'
        )
    else:
        test_name = "Mann–Whitney U"
        stat, pval = mannwhitneyu(
            *data.values(),
            alternative='two-sided'
        )

    # --- Compute AUC ---
    auc = roc_auc_score(df_clean[group_col], df_clean[value_col])
    
    if auc < 0.5:
        auc = 1- auc

    return {
        'test': test_name,
        'p_value': pval,
        'normality_p': normality,
        'levene_p': levene_p,
        'AUC': auc
    }


print(compare_groups(
    ivim_summary_df_no_lipoma,
    group_col='final_dx',
    value_col='D_mean'
))


print(compare_groups(
    ivim_summary_df_no_lipoma,
    group_col='final_dx',
    value_col='f_mean'
))

print(compare_groups(
    ivim_summary_df_no_lipoma,
    group_col='final_dx',
    value_col='Dstar_mean'
))





# --- Step 1: compute p-values ---
def compute_pvals(df, group_col='final_dx'):
    """
    Computes univariate p-values comparing two diagnostic groups
    (e.g. Benign vs Non-Benign) for all quantitative features.

    A non-parametric Mann–Whitney U test is used for *all* features
    to ensure consistency across measures and to avoid assumptions
    of normality, given the modest sample sizes and potential
    skewness/outliers in IVIM parameters.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing group labels and quantitative features.
    group_col : str
        Name of the column defining the two groups.

    Returns
    -------
    pandas.DataFrame
        Table containing feature name, p-value, and test used.
    """

    
    
    pvals = []
    cols_to_test = df.columns[df.columns.get_loc(group_col)+1:]  # everything after final_dx
    
    for col in cols_to_test:
        data = df[[group_col, col]].dropna()
        groups = data[group_col].unique()
        if len(groups) != 2:
            continue  # skip if not two groups
        
        vals = {g: data.loc[data[group_col]==g, col] for g in groups}
        
        # normality per group
        normality = [shapiro(v)[1] if len(v)>=3 else np.nan for v in vals.values()]
        
        # Use t-test only if both groups normal - jsut going to use Mann-WHitney so all are comparable
        #if all(p > 0.05 for p in normality):
        #    # Check variances for Welch vs standard t-test
        #    equal_var = levene(*vals.values()).pvalue >= 0.05
        #    _, pval = ttest_ind(*vals.values(), equal_var=equal_var, nan_policy='omit')
        #    test_used = 't-test'
        #else:
        _, pval = mannwhitneyu(*vals.values(), alternative='two-sided')
        test_used = 'Mann-Whitney'
        
        pvals.append({'measure': col, 'p_value': pval, 'test': test_used})
    
    return pd.DataFrame(pvals)

pval_df = compute_pvals(ivim_summary_df_no_lipoma)
pval_df


def compute_stats(df, group_col='final_dx'):
    """
    Computes both statistical significance (Mann–Whitney p-value)
    and discriminative performance (ROC AUC) for each quantitative feature.

    This function is intended to identify features that are not only
    statistically different between groups, but also potentially useful
    for downstream classification / radiomics models.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing group labels and quantitative features.
    group_col : str
        Name of the column defining the two groups.

    Returns
    -------
    pandas.DataFrame
        Table containing feature name, p-value, and ROC AUC.
    """

    results = []
    cols_to_test = df.columns[df.columns.get_loc(group_col)+1:]
    
    for col in cols_to_test:
        data = df[[group_col, col]].dropna()
        groups = data[group_col].unique()
        
        if len(groups) != 2:
            continue
        
        # Map groups to 0 and 1
        group_map = {groups[0]: 0, groups[1]: 1}
        y = data[group_col].map(group_map)
        x = data[col]
        
        # Mann–Whitney test
        vals = [x[y==0], x[y==1]]
        _, pval = mannwhitneyu(*vals, alternative='two-sided')
        
        # ROC AUC
        try:
            auc = roc_auc_score(y, x)
        except:
            auc = np.nan
            
        if auc < 0.5:
            auc = 1 - auc
        
        results.append({
            'measure': col,
            'p_value': pval,
            'AUC': auc
        })
    
    return pd.DataFrame(results)

stats_df = compute_stats(ivim_summary_df_no_lipoma)



# Save extracted IVIM and ADC summary statistics
# for use as input features in downstream radiomics models

ivim_summary_df_no_lipoma.to_csv(r'D:\Radiomics\RetrospectiveData\Final_Results\2 step model + IVIM\IVIM_ADC_results.csv')



plt.figure(figsize=(10,8))
sns.set(style="whitegrid")
sns.barplot(x='p_value', y='measure', data=pval_df, color='skyblue')

plt.axvline(0.05, color='red', linestyle='--', label='p=0.05')
plt.xlabel('p-value', fontsize = 16)
plt.xticks(fontsize = 14)
plt.ylabel('IVIM measure', fontsize = 16)
plt.yticks(fontsize = 14)
plt.title('Group comparison p-values', fontsize = 20)
plt.legend(fontsize = 14)
plt.tight_layout()
plt.show()




stats_df = stats_df.iloc[::-1]

fig, (ax1, ax2) = plt.subplots(
    ncols=2,
    sharey=True,
    figsize=(14, 10),
    gridspec_kw={'width_ratios': [1, 1]}
)

sns.set(style="whitegrid")

# --- LEFT: p-values ---
ax1.barh(stats_df['measure'], stats_df['p_value'])
ax1.axvline(0.05, linestyle='--', color = 'red')
ax1.set_xlim(0.5, 0)  # reverse axis (right to left)
ax1.set_xlabel("p-value", fontsize=14)
ax1.set_ylabel("")
ax1.set_title("p-values", fontsize=16)

# Put labels in middle
ax1.yaxis.tick_right()
# Increase padding so labels sit between plots
ax1.tick_params(axis='y', labelsize=16, pad=25)
ax1.tick_params(axis='y', labelsize=14)

# --- RIGHT: AUC ---
ax2.barh(stats_df['measure'], stats_df['AUC'])
ax2.set_xlim(0, 1)
ax2.set_xlabel("AUC", fontsize=14)
ax2.set_title("ROC AUC", fontsize=16)
ax2.tick_params(axis='y', left=False, labelleft=False)

plt.suptitle("Group Comparison: IVIM Parameters", fontsize=18)
plt.tight_layout()
plt.show()











'''
No longer used as not doing grading

ivim_summary_malignant_only = ivim_summary_df[ivim_summary_df['WHO_category'] == "M"].copy()

ivim_summary_malignant_only['Grade_clean'] = (
    ivim_summary_malignant_only['Grade']
    .str.strip()
    .str.lower()
)

ivim_summary_malignant_only = ivim_summary_malignant_only[~ivim_summary_malignant_only['Grade_clean'].isin(
    ['not specified', 'no grade', 'not sarcoma', 'not known', 'not known - treated abroad']
)]

high_grade_map = {
    '1': 0,
    'low grade': 0,
    '2': 1,
    '3': 1,
    'high grade': 1
}


ivim_summary_malignant_only['High_grade'] = ivim_summary_malignant_only['Grade_clean'].map(high_grade_map)




def plot_high_vs_low(map_type, title):
    
    
    # plot D
    plt.figure(figsize=(10, 8))
    
    # Boxplot first (background)
    sns.boxplot(
        data=ivim_summary_malignant_only,
        x='High_grade',
        y= map_type,
        whis=[5, 95],        # Optional: show 5th-95th percentile as whiskers
        color='lightgray',
        showfliers=False     # Don’t show points for boxplot itself
    )
    
    # Overlay the individual points
    sns.stripplot(
        data=ivim_summary_malignant_only,
        x='High_grade',
        y= map_type,
        jitter=True,
        alpha=0.7,
        color='blue'
    )
    
    plt.xlabel('Final diagnosis')
    plt.ylabel(title)
    plt.title(f'{title} by grade', fontsize = 18)
    plt.xticks(
        ticks=[0, 1],
        labels=['Low grade', 'High grade']
    )
    
    #plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


plot_high_vs_low(map_type = 'D_mean', title = "Mean D")
plot_high_vs_low(map_type = 'f_mean', title = "Mean f")
plot_high_vs_low(map_type = 'Dstar_mean', title = "Mean D*")



print(compare_groups(
    ivim_summary_malignant_only,
    group_col='High_grade',
    value_col='D_mean'
))


print(compare_groups(
    ivim_summary_malignant_only,
    group_col='High_grade',
    value_col='f_mean'
))

print(compare_groups(
    ivim_summary_malignant_only,
    group_col='High_grade',
    value_col='Dstar_mean'
))




# Lookign at some minimum values


plot_B_nonB(map_type = 'D_p05', title = "D_p05")
plot_B_nonB(map_type = 'f_p05', title = "f_p05")
plot_B_nonB(map_type = 'Dstar_p05', title = "Dstar_p05")

print(compare_groups(
    ivim_summary_df_no_lipoma,
    group_col='final_dx',
    value_col='D_p05'
))


print(compare_groups(
    ivim_summary_df_no_lipoma,
    group_col='final_dx',
    value_col='f_p05'
))

print(compare_groups(
    ivim_summary_df_no_lipoma,
    group_col='final_dx',
    value_col='Dstar_p05'
))

'''
