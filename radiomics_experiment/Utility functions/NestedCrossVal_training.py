# -*- coding: utf-8 -*-
"""
Custom implementation of nested cross-validation
with explicit control over hyperparameter tuning,
threshold optimisation, feature-selection tracking,
and model evaluation.


Created on Thu Oct 23 15:04:28 2025

@author: marzettm
"""

import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_curve, auc, confusion_matrix
from joblib import Parallel, delayed
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import os
import joblib
import pandas as pd
from matplotlib.colors import ListedColormap
import seaborn as sns
from mlxtend.classifier import EnsembleVoteClassifier
import scipy.stats as stats
from scipy.stats import mannwhitneyu


from collections import Counter


# all from here: https://github.com/jacobgil/confidenceinterval - will need to reference though
from confidenceinterval import roc_auc_score
from confidenceinterval.bootstrap import bootstrap_ci
from confidenceinterval import precision_score, recall_score, f1_score
from confidenceinterval import (accuracy_score,
                                ppv_score,
                                npv_score,
                                tpr_score,
                                fpr_score,
                                tnr_score)
from confidenceinterval import classification_report_with_ci



# -----------------------------------------------------------------------------
#              NESTED CROSS-VALIDATION TRAINING AND EVALUATION
# -----------------------------------------------------------------------------


def custom_inner_cv_threshold_tuning(
    pipe, param_grid, inner_cv, X_train, y_train,
    target_sensitivity = 0.95, n_jobs=-1, random_search=False
):
    """
    Perform custom inner cross-validation for hyperparameter tuning with threshold optimization.

    This function evaluates multiple hyperparameter combinations using inner CV folds.
    For each combination, it:
    - Fits the pipeline on training folds
    - Predicts probabilities on validation folds
    - Computes ROC curves and selects a threshold that meets or exceeds a target sensitivity
    - Calculates AUC, sensitivity, and specificity at the selected threshold

    The best parameter set is selected based on mean AUC across folds.
    The final model is trained on the full inner training set using the best parameters.

    Parameters
    ----------
    pipe : sklearn.pipeline.Pipeline
        Unfitted pipeline containing preprocessing, feature selection, and estimator.

    param_grid : dict
        Dictionary of hyperparameters to search over (like GridSearchCV).

    inner_cv : sklearn.model_selection.BaseCrossValidator
        Cross-validation splitter for inner loop (e.g., StratifiedKFold).

    X_train : pd.DataFrame
        Feature matrix for inner CV.

    y_train : pd.Series
        Target vector for inner CV.

    target_sensitivity : float, default=0.95
        If provided (0 < target_sensitivity ≤ 1), selects a classification threshold
        that achieves at least this sensitivity while maximizing specificity.
        If None, uses the ROC-optimal Youden threshold.

    n_jobs : int, default=-1
        Number of parallel jobs to run. If -1, uses all available cores.

    random_search : bool, default=False
        If True, randomly samples up to 100 parameter combinations from the grid.

    Returns
    -------
    best_model : sklearn.pipeline.Pipeline
        Pipeline fitted on full training data with best hyperparameters.

    best_result : dict
        Dictionary containing best parameters, mean AUC, threshold, sensitivity, and specificity.

    results_all : list of dict
        List of results for each parameter combination, including:
        - 'params': hyperparameter set
        - 'mean_auc': average AUC across folds - based on validaiton performance
        - 'mean_threshold': average threshold selected
        - 'mean_youden_threshold': mean ROC-optimal Youden threshold
        - 'train_sens_at_threshold': average sensitivity on training folds
        - 'train_spec_at_threshold': average specificity on training folds

    Notes
    -----
    - Assumes binary classification with `predict_proba` available.
    - Uses mean AUC across folds to select the best model.
    - Threshold selection is based on maximizing specificity while meeting target sensitivity.
    - If `target_sensitivity` is None, the Youden-optimal threshold is used.

    """


    from itertools import product

    # Generate all param combos like GridSearchCV does
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in product(*values)]


    n_folds = inner_cv.get_n_splits(X_train, y_train) if hasattr(inner_cv, 'get_n_splits') else 'unknown'

    if random_search:
        random_seed = 42 # reproducibility purposes
        np.random.seed(random_seed)
        total_combinations = len(param_combinations)
        sample_size = min(100, total_combinations) # should use at least 10 * n_hyper params
        param_combinations = list(np.random.choice(param_combinations, size=sample_size, replace=False))
    
    n_combinations = len(param_combinations)
    


    if n_jobs != -1:
        n_jobs = min(n_jobs, n_folds)

    print(f"[Inner CV] Running {n_combinations} parameter combinations "
          f"with {n_folds} folds = {n_combinations * n_folds} total fits "
          f"({n_jobs} jobs in parallel).")
    




    def eval_one_fold(params, train_idx, val_idx):
        X_inner_train, X_inner_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_inner_train, y_inner_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = clone(pipe)
        model.set_params(**params)
        model.fit(X_inner_train, y_inner_train)

        # Predict probs on inner val fold
        y_proba = model.predict_proba(X_inner_val)[:, 1]

        # Compute ROC curve + AUC
        fpr, tpr, thresholds_roc = roc_curve(y_inner_val, y_proba)
        auc_score = auc(fpr, tpr)
    
        # Youden threshold        
        youden = tpr - fpr
        youden_idx = np.argmax(youden)
        youden_thresh = thresholds_roc[youden_idx]
        
        

        # Find threshold(s) with tpr >= target_sensitivity
        if target_sensitivity is None:
            thresh = youden_thresh
        elif 0 < target_sensitivity <= 1:
            # Sensitivity-constrained threshold
            idx = np.where(tpr >= target_sensitivity)[0]
            if len(idx) > 0:
                best_idx = idx[np.argmax(1 - fpr[idx])]
                thresh = thresholds_roc[best_idx]
            else:
                thresh = 0.0  # fallback: classify all positive
        else:
            raise ValueError("target_sensitivity must be None or in (0, 1]")
                
            

    
    
    
        #idx = np.where(tpr >= target_sensitivity)[0]
        #if len(idx) > 0:
        #    best_idx = idx[np.argmax(1 - fpr[idx])]
        #    thresh = thresholds_roc[best_idx]
        #else:
        #    thresh = 0.0  # fallback: classify all as positive

        # Train fold sens/spec at threshold
        y_proba_train = model.predict_proba(X_inner_train)[:, 1]
        y_pred_train = (y_proba_train >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_inner_train, y_pred_train).ravel()
        sens_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec_val = tn / (tn + fp) if (tn + fp) > 0 else 0

        return auc_score, thresh, youden_thresh, sens_val, spec_val

    results_all = []

    for params in param_combinations:

        #print(f"Currently testing following paramters: {params}")

        fold_results = Parallel(n_jobs=n_jobs)(
            delayed(eval_one_fold)(params, train_idx, val_idx)
            for train_idx, val_idx in inner_cv.split(X_train, y_train)
        )

        aucs, thresholds, youden_thresholds, sens_list, spec_list = zip(*fold_results)
        #print(f"Youden thresholds: {youden_thresholds}")

        mean_auc = np.mean(aucs)
        mean_thresh = np.mean(thresholds)
        youden_thresholds_clean = [t for t in youden_thresholds if np.isfinite(t)]
        mean_youden = np.mean(youden_thresholds_clean)
        mean_train_sens = np.mean(sens_list)
        mean_train_spec = np.mean(spec_list)

        result_dict = {
            "params": params,
            "mean_auc": mean_auc,
            "mean_threshold": mean_thresh,
            "mean_youden_threshold": mean_youden,
            "train_sens_at_threshold": mean_train_sens,
            "train_spec_at_threshold": mean_train_spec
        }
        results_all.append(result_dict)

    # Pick best
    best_result = max(results_all, key=lambda r: r["mean_auc"])
    best_params = best_result["params"]

    #print(f"Best auc: {best_result['mean_auc']}")
    #print(f"Best params: {best_params}")

    # Fit final model on full training data
    best_model = clone(pipe)
    best_model.set_params(**best_params)
    best_model.fit(X_train, y_train)

    return best_model, best_result, results_all




def nested_training(
    X_train, y_train, pipe, grid, model_name, model_save_dir, outer_cv, inner_cv,
    feature_selector_in_pipe=True, random_search=True, target_sensitivity=None,
    title_text="Training dataset", plot_mean_roc_curve=False, return_outer_proba=False,
    interpret_features=False
):
    """
    Perform nested cross-validation with threshold tuning, feature selection tracking,
    and ROC curve plotting for a given pipeline and hyperparameter grid.

    This function trains and evaluates a model using nested cross-validation:
    - The inner loop performs hyperparameter tuning (optionally via random search).
    - The outer loop evaluates generalization performance.
    - Feature selection steps (e.g., Coefficient of Variation and Correlation filters)
      are tracked per fold.
    - ROC curves are plotted and saved, and trained models are serialized.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.

    y_train : pd.Series
        Training target vector.

    pipe : sklearn.pipeline.Pipeline
        Pipeline containing preprocessing, feature selection, and estimator.

    grid : dict or list of dict
        Hyperparameter grid for tuning the estimator within the pipeline.

    model_name : str
        Name used for saving models and results.

    model_save_dir : str
        Directory path where models and results will be saved.

    outer_cv : sklearn.model_selection.BaseCrossValidator
        Cross-validator for the outer loop (e.g., StratifiedKFold).

    inner_cv : sklearn.model_selection.BaseCrossValidator
        Cross-validator for the inner loop used in hyperparameter tuning.

    feature_selector_in_pipe : bool, default=True
        Whether the final feature selector is part of the pipeline.

    random_search : bool, default=True
        If True, performs randomized search; otherwise, uses grid search.

    target_sensitivity : float or None, default=None
        Optional sensitivity target to highlight on ROC plot.

    title_text : str, default="Training dataset"
        Title for the ROC plot.

    plot_mean_roc_curve : bool, default=False
        If True, calls an external function to plot the mean ROC curve across folds.

    return_outer_proba : bool, default=False
        If True, returns concatenated true labels and predicted probabilities from outer folds.

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame containing per-fold results including:
        - Best hyperparameters
        - AUC scores
        - Thresholds
        - Sensitivity and specificity (train/test)
        - Number of features after each selection step

    all_y_true : np.ndarray, optional
        Concatenated true labels from outer test folds (only if return_outer_proba=True).

    all_y_proba : np.ndarray, optional
        Concatenated predicted probabilities from outer test folds (only if return_outer_proba=True).

    Notes
    -----
    - Saves trained models per fold as `.joblib` files.
    - Saves ROC plot and results CSV to disk.
    - Assumes pipeline contains named steps: 'cv_filter', 'corr_filter', and optionally 'feature_selection'.
    - Uses a custom function `custom_inner_cv_threshold_tuning` for inner loop optimization.
    """
    
    
    
    best_params_per_fold = []
    best_auc_val = []
    nested_scores = []
    #partial_auc_list = []
    best_models = []

    n_features = []    
    selected_features_per_fold = [] # list containing n outer lists of selected features
    n_CoV_list = []
    n_corr_list = []

    youden_thresholds = []
    thresholds = []
    train_sensitivities = []
    train_specificities = []
    test_sens_at_threshold = []
    test_spec_at_thresholds = []
    
    
    
    # for overall ROC   
    all_y_true = []
    all_y_proba = []
    all_indices = [] # - keep track of original indices, so can do failure analysis
    all_folds = []

    

    # Choose a perceptually uniform colormap (good for colour blindness)
    cmap = cm.get_cmap("viridis", 10)  # 5 evenly spaced colours from blue→green→yellow, all distinct - only going to use first 5 to avoid the yellow
    #cmap = cm.get_cmap("PuBuGn", 7)
    plt.figure(figsize=(8, 6))


    for fold_number, (train_idx, test_idx) in enumerate(outer_cv.split(X_train, y_train), start=1): # validation done inside inner loop
        X_train_cv, X_test_cv = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_train_cv, y_test_cv = y_train.iloc[train_idx], y_train.iloc[test_idx]
        
        print(f"Size of test data: {len(y_test_cv)}, ({np.sum(y_test_cv)} positive labels)")
        
        
        test_indices = X_train.index[test_idx]  # <-- original indices preserved
        all_indices.append(test_indices)



        best_model, best_result, inner_results = custom_inner_cv_threshold_tuning(
            pipe, grid, inner_cv, X_train_cv, y_train_cv, 
            target_sensitivity=target_sensitivity,
            n_jobs = 10,
            random_search = random_search
        )
        
        best_model.fit(X_train_cv, y_train_cv)
        

        # save the best performing model from inner loop
        best_models.append(best_model)
        best_params_per_fold.append(best_result["params"])
        best_auc_val.append(best_result["mean_auc"])
        youden_thresholds.append(best_result["mean_youden_threshold"])
        thresholds.append(best_result["mean_threshold"])
        train_sensitivities.append(best_result["train_sens_at_threshold"]) # these are on the training folds, but AUC is on validation
        train_specificities.append(best_result["train_spec_at_threshold"])
        
        # Access the feature selector from the best estimator and save the number of selected features - extend to get number of variance threshold and correlation
        cv_mask = best_model.named_steps['cv_filter'].get_support()
        corr_mask = best_model.named_steps['corr_filter'].get_support()
        n_cv = cv_mask.sum()
        n_corr = corr_mask.sum()
        n_CoV_list.append(n_cv)
        n_corr_list.append(n_corr)

        if feature_selector_in_pipe:
            feature_selector = best_model.named_steps['feature_selection']
            # Step 1: CoV filter
            cv_mask = best_model.named_steps['cv_filter'].get_support()
            names_after_cv = X_train_cv.columns[cv_mask]
            
            # Step 2: Correlation filter
            corr_mask = best_model.named_steps['corr_filter'].get_support()
            names_after_corr = names_after_cv[corr_mask]
            
            # Step 3: Final feature selector
            fs_mask = best_model.named_steps['feature_selection'].get_support()
            selected_features = names_after_corr[fs_mask].tolist()
            
            
            #selected_mask = feature_selector.get_support()
            #selected_features = X_train_cv.columns[selected_mask].tolist()
            # Number of selected features
            n_selected = len(selected_features)
        else:
            n_selected = X_train.shape[1]
            selected_features = X_train.columns.tolist()
            
        n_features.append(n_selected)
        selected_features_per_fold.append(selected_features)

        # No need to calculate threshold on full train fold here anymore


        y_proba = best_model.predict_proba(X_test_cv)[:, 1]
        
        # testing fitting threshold on full outer test fold
        #y_proba_outer_train = best_model.predict_proba(X_train_cv)[:, 1]
        #fpr_tr, tpr_tr, thr_tr = roc_curve(y_train_cv, y_proba_outer_train)
        
        
        #idx_tr = np.where(tpr_tr >= target_sensitivity)[0]
        #if len(idx_tr) > 0:
        #    best_idx = idx_tr[np.argmax(1 - fpr_tr[idx_tr])]
        #    thresh_tr = thr_tr[best_idx]
        #else:
        #    thresh_tr = 0.0  # fallback: classify all positive
        #print(thresh_tr)
        
        
        all_y_proba.append(y_proba)
        all_y_true.append(y_test_cv)
        all_folds.append(np.full(len(y_test_cv), fold_number))
        
        fpr_test, tpr_test, thresholds_test = roc_curve(y_test_cv, y_proba)
        spec_test = 1 - fpr_test
        score = auc(fpr_test, tpr_test)
        nested_scores.append(score)

        # plotting
        color = cmap(fold_number-1)  
        plt.plot(fpr_test, tpr_test, lw=2, color=color, label=f'Fold {fold_number} AUC={score:.2f}')


        best_threshold = best_result["mean_threshold"]
        if best_threshold is not None:
            y_pred_test = (y_proba >=  best_threshold).astype(int)
            
            # Compute confusion matrix elements
            tn, fp, fn, tp = confusion_matrix(y_test_cv, y_pred_test).ravel()
            
            sens_test = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec_test = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            # fallback if threshold was None (rare)
            sens_test = None
            spec_test = None

        # save test sens/spec
        test_sens_at_threshold.append(sens_test)
        test_spec_at_thresholds.append(spec_test)



        # Save the best model for this fold
        model_save_folder = os.path.join(model_save_dir, model_name)
        os.makedirs(model_save_folder, exist_ok = True)
        
        model_filename = os.path.join(model_save_folder , f'{model_name}_best_model_fold_{fold_number}.joblib')
        
        #y_proba_check = best_model.predict_proba(X_test_cv)[:, 1]
        #print(f"Testing: {np.allclose(y_proba, y_proba_check)}")
        
        joblib.dump(best_model, model_filename)
        print(f"Saved model for fold {fold_number} to {model_filename}")



    results_df = pd.DataFrame(best_params_per_fold)

    # scores from training on the best model
    results_df.insert(0, "fold", range(1, len(results_df) + 1))
    results_df['n_features_after_CoV_filter'] = n_CoV_list
    results_df['n_features_after_corr_filter'] = n_corr_list
    results_df["n_features_afterFeatSelection"] = n_features
    results_df["auc_val"] = best_auc_val
    results_df["youden_threshold"] = youden_thresholds
    results_df["threshold"] = thresholds
    results_df["train_sens_at_threshold"] = train_sensitivities
    results_df["train_spec_at_threshold"] = train_specificities
    # scores on test sets
    results_df["test_auc_scores"] = nested_scores
    #results_df["partial_auc(sens>=95%)"] = partial_auc_list
    results_df["sensitivity(at_threshold)"] = test_sens_at_threshold
    results_df["specificity(at_threshold)"] = test_spec_at_thresholds
    

    results_csv_save_dir = os.path.join(model_save_dir, "Results")
    os.makedirs(results_csv_save_dir, exist_ok=True)

    results_df = results_df.round({
        "auc_val": 3,
        "youden_threshold":4,
        "threshold": 4,
        "train_sens_at_threshold": 3,
        "train_spec_at_threshold": 3,
        "test_auc_scores": 3,
        #"partial_auc(sens>=95%)": 3,
        "sensitivity(at_threshold)": 3,
        "specificity(at_threshold)": 3,
    })

    results_df.to_csv(os.path.join(results_csv_save_dir, model_name + '_results.csv'), index=False) 


    # Plot chance line
    plt.plot([0, 1], [0, 1], 'k--', label='Chance')

    # Plot horizontal line at chosen sensitivity
    if target_sensitivity:
        plt.axhline(y=target_sensitivity, color='red', linestyle='--', label=f'Target Sensitivity = {target_sensitivity:.2f}')

    # Add grid lines
    plt.grid(True, which='both', linestyle=':', linewidth=0.7)
    plt.xticks(np.arange(0.0, 1.1, 0.1), fontsize=12)
    plt.yticks(np.arange(0.0, 1.1, 0.1), fontsize=12)


    # Labels and title
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=16)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=16)
    #plt.title(f'ROC Curves Across 5 Outer Folds - {model_name}')
    plt.title(title_text, fontsize=18)
    plt.legend(loc='lower right', fontsize=16)
    plt.tight_layout()

    roc_plot_filename = os.path.join(results_csv_save_dir, f"{model_name}_roc_5folds.jpg")
    plt.savefig(roc_plot_filename, dpi=300, bbox_inches='tight')
    print(f"ROC plot saved to {roc_plot_filename}")

    # Show plot
    plt.show()

    if plot_mean_roc_curve:
        plot_mean_roc(X_train, y_train, model_name, model_save_dir, outer_cv, title_text, results_csv_save_dir)


    if interpret_features:
        all_selected = [feat for sublist in selected_features_per_fold for feat in sublist]
        
        feature_freq_df = (
            pd.Series(all_selected)
            .value_counts()
            .rename_axis("feature")
            .reset_index(name="frequency")
        )
        
        feature_freq_df.to_csv(os.path.join(results_csv_save_dir, model_name + '_features.csv'), index=False) 


    


    if return_outer_proba:
        all_y_true = np.concatenate(all_y_true)
        all_y_proba = np.concatenate(all_y_proba)
        all_indices = np.concatenate(all_indices)
        all_folds = np.concatenate(all_folds)
        
        df_all_preds = pd.DataFrame({
            "true": all_y_true,
            "proba": all_y_proba,
            "fold": all_folds,
            "train_index": all_indices
        })
        
        # can do
        # df["patient_id"] = original_df.loc[df["index"], "patient_id"].values 
        # to match ID's to predictions - useful in failure analysis
        
        
    if return_outer_proba and interpret_features:
        return results_df, df_all_preds, feature_freq_df
        
    if interpret_features:
        return results_df, feature_freq_df
    if return_outer_proba:
        return results_df, df_all_preds
    else:
        return results_df




# -----------------------------------------------------------------------------
#                       SUMMARISE OUTER FOLDS
# -----------------------------------------------------------------------------


def mean_ci(values, confidence=0.95):
    """
    Compute the mean and two-sided confidence interval using the t-distribution.

    This is intended for summarising cross-validation performance metrics
    (e.g., mean AUC from outer folds). The confidence interval is calculated
    from the standard error of the mean (SEM), using the sample standard
    deviation (ddof=1) and the appropriate critical t-value.

    Parameters
    ----------
    values : array-like
        Sequence of numeric values (e.g., AUCs from cross-validation folds).
    confidence : float, optional
        Confidence level for the interval (default is 0.95 for a 95% CI).

    Returns
    -------
    mean : float
        Sample mean of the input values.
    lower : float
        Lower bound of the confidence interval.
    upper : float
        Upper bound of the confidence interval.

    Notes
    -----
    - The SEM is computed using scipy.stats.sem, which uses ddof=1
      (i.e., the unbiased sample standard deviation).
    - The confidence interval is calculated as:

          mean ± t_crit * SEM

      where t_crit is obtained from the Student's t-distribution with
      n - 1 degrees of freedom.
    """
    values = np.array(values)
    n = len(values)
    mean = np.mean(values)
    sem = stats.sem(values)  # standard error (uses ddof=1) # this means using sample std dev, not population. So this is correct for me
    
    t_crit = stats.t.ppf((1 + confidence) / 2, df=n-1)
    margin = t_crit * sem
    
    return mean, mean - margin, mean + margin


def plot_overall_AUC(all_y_true, all_y_proba, title = 'Overall ROC Curve Across All Outer Test Folds', save_folder = None):
    
        
    """
    Plot a pooled ROC curve using predictions aggregated across all outer
    cross-validation test folds.
     
    Unlike averaging ROC curves across folds, this function concatenates
    all ground-truth labels and predicted probabilities from the outer
    test sets and computes a single ROC curve from the combined data.
    This provides an overall estimate of model discrimination across all
    held-out samples.
     
    The function also calculates the area under the ROC curve (AUC) and
    its associated 95% confidence interval using the
    `confidenceinterval` package.
     
    Parameters
    ----------
    all_y_true : array-like
    Ground-truth binary class labels.
     
    all_y_proba : array-like
    Predicted probabilities for the positive class.
     
    title : str, default='Overall ROC Curve Across All Outer Test Folds'
    Title displayed on the ROC plot.
     
    save_folder : str or None, default=None
    Directory in which the ROC figure should be saved.
    If None, the figure is displayed but not saved.
     
    Returns
    -------
    auc_all : float
    Overall AUC computed from the pooled predictions.
     
    ci : tuple of float
    Lower and upper bounds of the 95% confidence interval for the AUC.
     
    Notes
    -----
    This pooled ROC approach differs from plotting the mean ROC across
    folds. It reflects model performance across all individual test
    samples rather than averaging fold-level ROC curves.
    """
    
    fpr_all, tpr_all, _ = roc_curve(all_y_true, all_y_proba)
    #auc_all = auc(fpr_all, tpr_all)
    
    auc_all, ci = roc_auc_score(all_y_true,
                        all_y_proba,
                        confidence_level=0.95)
    
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr_all, tpr_all, lw=2, #color='darkorange',
             #label=f'Overall ROC (AUC = {auc_all:.2f} \n   [{ci[0]:.2f} - {ci[1]:.2f}])')
             label=f'Overall ROC (AUC = {auc_all:.2f} [{ci[0]:.2f} - {ci[1]:.2f}])')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Chance')
    
    plt.grid(True, which='both', linestyle=':', linewidth=0.7)
    plt.xticks(np.arange(0.0, 1.1, 0.1), fontsize=12)
    plt.yticks(np.arange(0.0, 1.1, 0.1), fontsize=12)
    
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=16)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=16)
    plt.title(title, fontsize=18)
    plt.legend(loc='lower right', fontsize=16)
    plt.grid(True, linestyle=':')
    plt.tight_layout()
    
    if save_folder != None: 
        roc_plot_filename = os.path.join(save_folder, f"{title}_pooled_ROC_allTestSets.jpg")
        plt.savefig(roc_plot_filename, dpi=300, bbox_inches='tight')
        print(f"ROC plot saved to {roc_plot_filename}")
    

    plt.show()
    


    return(auc_all, ci)




def overall_AUC_thresholds(predictions_df):
    """
    Evaluate commonly used ROC-derived classification thresholds on
    pooled model predictions.
      
    This function computes the ROC curve from a set of predicted
    probabilities and reports performance at:
     
    1. The Youden-optimal threshold
    (maximising sensitivity + specificity - 1).
     
    2. The top-left threshold
    (minimising Euclidean distance to the ideal ROC point (0,1)).
     
    For each threshold, the function prints the corresponding accuracy,
    sensitivity, and specificity.
     
    Parameters
    ----------
    predictions_df : pandas.DataFrame
    DataFrame containing:
    - 'true' : binary ground-truth labels.
    - 'proba' : predicted probabilities for the positive class.
     
    Returns
    -------
    None
     
    Notes
    -----
    This function is intended as an exploratory utility for examining
    operating points on an ROC curve. It does not return values and
    instead prints performance metrics directly to the console.
    """
    
    


    # Extract true labels and probabilities
    y_true = predictions_df['true'].values
    y_proba = predictions_df['proba'].values
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)


    # --------------------------
    # 1. YOUDEN THRESHOLD
    # --------------------------
    
    # Compute Youden's Index for each threshold
    youden_index = tpr - fpr
    
    # Get best threshold (max J)
    best_idx = youden_index.argmax()
    best_threshold = thresholds[best_idx]
    
    print(f"Best threshold (Youden): {best_threshold:.3f}")
    
    #print(f"Sensitivity: {tpr[best_idx]:.3f}")
    #print(f"Specificity: {1 - fpr[best_idx]:.3f}")
    
    # ------------------------------
    # Get Accuracy, Sensitivity, Specificity at that threshold
    # ------------------------------
    
    # Convert probabilities to class predictions
    y_pred = (y_proba >= best_threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Sensitivity: {sensitivity:.3f}")
    print(f"Specificity: {specificity:.3f}")



    # --------------------------
    # 2. TOP-LEFT THRESHOLD
    # --------------------------
    distance = np.sqrt((fpr - 0)**2 + (1 - tpr)**2)
    best_topleft_idx = np.argmin(distance)
    best_topleft_threshold = thresholds[best_topleft_idx]
    
    print(f"\nBest threshold (Top-Left): {best_topleft_threshold:.3f}")
    
    # Convert probabilities to class predictions
    y_pred = (y_proba >= best_topleft_threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Sensitivity: {sensitivity:.3f}")
    print(f"Specificity: {specificity:.3f}")


def plot_mean_roc(X_train, y_train, model_name, model_save_dir, outer_cv, title_text, results_csv_save_dir, show_95sens = False):
    """
    Plot the mean ROC curve across outer cross-validation folds.
     
    Individual ROC curves from each outer test fold are plotted in grey,
    while the mean ROC curve is calculated by interpolating true-positive
    rates (TPRs) onto a common false-positive rate (FPR) grid and then
    averaging across folds. The plot also displays ±1 standard deviation
    around the mean ROC curve.
     
    Parameters
    ----------
    X_train : pandas.DataFrame
    Feature matrix used during nested cross-validation.
     
    y_train : pandas.Series
    Binary target labels corresponding to X_train.
     
    model_name : str
    Name of the model. Used to locate saved fold-specific models and
    generate the output filename.
     
    model_save_dir : str
    Directory containing the saved models generated during nested
    cross-validation.
     
    outer_cv : sklearn.model_selection.BaseCrossValidator
    Cross-validation splitter defining the outer folds.
     
    title_text : str
    Title displayed on the ROC plot.
     
    results_csv_save_dir : str
    Directory in which the figure will be saved.
     
    show_95sens : bool, default=False
    If True, adds a horizontal reference line at a sensitivity of
    0.95 to facilitate visual assessment of performance at a
    clinically relevant operating point.
     
    Returns
    -------
    None
     
    Notes
    -----
    This function differs from `plot_overall_AUC()`, which pools all
    outer-fold predictions into a single ROC analysis. Here, ROC curves
    are first computed separately for each fold and then averaged,
    providing a visual representation of cross-validation variability.
     
    The shaded region represents ±1 standard deviation of the
    interpolated true-positive rates across folds and should be
    interpreted as a measure of variability rather than a formal
    confidence interval.
     
    Trained models are loaded from disk using joblib and are expected
    to have been previously saved by the `nested_training()` function.
    """
    
    from sklearn.metrics import roc_curve, auc
    import numpy as np
    import matplotlib.pyplot as plt
    import joblib
    import os

    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    plt.figure(figsize=(8, 6))

    # Plot each individual ROC curve in grey
    for fold_number, (train_idx, val_idx) in enumerate(outer_cv.split(X_train, y_train), start=1):
        X_train_cv, X_test_cv = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_cv, y_test_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model_path = os.path.join(model_save_dir, model_name, f"{model_name}_best_model_fold_{fold_number}.joblib")
        best_model = joblib.load(model_path)

        y_proba = best_model.predict_proba(X_test_cv)[:, 1]
        fpr, tpr, _ = roc_curve(y_test_cv, y_proba)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        plt.plot(fpr, tpr, color='grey', lw=1.2, alpha=0.6)

        # Interpolate to a common FPR grid
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    # Compute mean and std of the interpolated TPRs
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # Plot mean ROC
    plt.plot(
        mean_fpr, mean_tpr, color='blue',
        label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})',
        lw=2.5
    )
    plt.fill_between(
        mean_fpr,
        np.maximum(mean_tpr - std_tpr, 0),
        np.minimum(mean_tpr + std_tpr, 1),
        color='blue', alpha=0.2, label='±1 SD'
    )

    # Plot chance line
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Chance')

    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=16)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=16)
    
    plt.grid(True, which='both', linestyle=':', linewidth=0.7)
    plt.xticks(np.arange(0.0, 1.1, 0.1))
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    if show_95sens:
        plt.axhline(y=0.95, color='red', linestyle='--', label='Target Sensitivity = 0.95')
    #plt.title(f'{title_text} \n Mean ROC Across Outer Folds', fontsize=16)
    plt.title(f'{title_text}', fontsize=18)
    plt.legend(loc='lower right', fontsize=16)
    plt.grid(True, which='both', linestyle=':', linewidth=0.7)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    # Save
    avg_roc_plot_filename = os.path.join(results_csv_save_dir, f"{model_name}_roc_mean.jpg")
    plt.savefig(avg_roc_plot_filename, dpi=300, bbox_inches='tight')
    print(f"Mean ROC plot saved to {avg_roc_plot_filename}")
    plt.show()


def plot_two_test_roc(
    y1, proba1, title1,
    y2, proba2, title2,
    save_path=None
):
    """
    Plot two independent test ROC curves side-by-side
    in a single figure with identical axis scaling.

    Parameters
    ----------
    y1, y2 : array-like
        Ground truth labels for dataset 1 and 2.
    proba1, proba2 : array-like
        Predicted probabilities for dataset 1 and 2.
    title1, title2 : str
        Titles for the left and right subplot.
    save_path : str, optional
        Full path (including filename) to save the figure.

    Returns
    -------
    (auc1, ci1), (auc2, ci2)
        AUC and 95% CI for each dataset.
    """

    # --- Compute ROC 1 ---
    fpr1, tpr1, _ = roc_curve(y1, proba1)
    auc1, ci1 = roc_auc_score(y1, proba1, confidence_level=0.95)

    # --- Compute ROC 2 ---
    fpr2, tpr2, _ = roc_curve(y2, proba2)
    auc2, ci2 = roc_auc_score(y2, proba2, confidence_level=0.95)

    # --- Create figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    for ax, fpr, tpr, auc_val, ci, title in zip(
        axes,
        [fpr1, fpr2],
        [tpr1, tpr2],
        [auc1, auc2],
        [ci1, ci2],
        [title1, title2]
    ):

        ax.plot(
            fpr,
            tpr,
            lw=2,
            label=f'AUC = {auc_val:.2f} [{ci[0]:.2f}–{ci[1]:.2f}]'
        )

        ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Chance performance')

        ax.set_xlim(-0.025, 1.025)
        ax.set_ylim(-0.025, 1.025)

        ax.set_xticks(np.arange(0.0, 1.1, 0.1))
        ax.set_yticks(np.arange(0.0, 1.1, 0.1))

        ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=16)
        ax.set_title(title, fontsize=18)

        ax.grid(True, linestyle=':')
        ax.legend(loc='lower right', fontsize=16)

    axes[0].set_ylabel('True Positive Rate (Sensitivity)', fontsize=16)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()

    return (auc1, ci1), (auc2, ci2)    


# -----------------------------------------------------------------------------
#                       INTERPRETABILITY AND FEATURES
# -----------------------------------------------------------------------------

    
def get_features_from_models(eclf, X):
    """
    Extract and summarise feature-selection frequencies across all models
    contained within an ensemble classifier.
     
    For each constituent model, the original feature-selection pipeline is
    replayed in order to identify the features retained after imputation,
    coefficient-of-variation filtering, correlation filtering, and final
    feature selection.
     
    The frequency with which each feature is selected across models is
    recorded and summarised.
     
    Parameters
    ----------
    eclf : mlxtend.classifier.EnsembleVoteClassifier
    Ensemble classifier whose constituent models are fitted
    pipelines containing feature-selection steps.
     
    X : pandas.DataFrame
    Original feature matrix used to reconstruct feature-selection
    decisions.
     
    Returns
    -------
    None
     
    Notes
    -----
    The function prints:
    - Number of selected features per model.
    - Total unique selected features.
    - Features selected by multiple models.
    - Features selected by all models.
     
    A feature-frequency DataFrame is also generated internally and
    displayed.
    """
    
    
    
    feature_names = np.array(X.columns)
    feature_counter = Counter()

    for i, model in enumerate(eclf.clfs_):
        pipe = model
    
        # Re-run imputer
        X_step1 = pipe.named_steps['imputer'].transform(X)
    
        # Re-run cv_filter
        X_step2 = pipe.named_steps['cv_filter'].transform(X_step1)
        names_after_cv = feature_names[pipe.named_steps['cv_filter'].keep_indices_]
    
        # Re-run corr_filter
        X_step3 = pipe.named_steps['corr_filter'].transform(X_step2)
        names_after_corr = np.array(names_after_cv)[pipe.named_steps['corr_filter'].keep_columns_]
    
        # Now feature_selection mask
        fs_mask = pipe.named_steps['feature_selection'].get_support()
        kept_features = names_after_corr[fs_mask]
    
        print(f"Model {i+1}: {len(kept_features)} features kept")
    
        # Count how many models kept each feature
        feature_counter.update(kept_features)

    # Features kept in multiple models
    multi_model_features = {f: c for f, c in feature_counter.items() if c > 1}

    print(f"\nTotal unique features across all models: {len(feature_counter)}")
    print(f"Features kept in more than one model: {len(multi_model_features)}")
    # Features kept in all models
    all_model_features = {f: c for f, c in feature_counter.items() if c == len(eclf.clfs_)}

    print(f"\n{len(all_model_features)} features kept in all {len(eclf.clfs_)} models:")
    #for feat in all_model_features:
    #    print(feat)
     
     
     # Convert to dataframe
    feature_freq_df = (
        pd.Series(feature_counter)
        .rename("frequency")
        .reset_index()
        .rename(columns={"index": "feature"})
        .sort_values("frequency", ascending=False)
        .reset_index(drop=True)
    )
    
    print(feature_freq_df)
      
    
def interpret_features(features_df, X, y, top_n=28):
    """
    Perform univariate interpretation of selected radiomic features.
     
    For each feature, this function calculates:
     
    - Univariate AUC.
    - Mann-Whitney U test p-value.
    - Selection frequency across models.
     
    Results are visualised using:
    - Horizontal bar plots of AUC values.
    - Horizontal bar plots of Mann-Whitney p-values.
    - Boxplots comparing feature distributions between classes.
     
    Parameters
    ----------
    features_df : pandas.DataFrame
    DataFrame containing feature names and selection frequencies.
    Expected columns include:
    - 'feature'
    - 'frequency'
     
    X : pandas.DataFrame
    Feature matrix containing the original feature values.
     
    y : array-like
    Binary class labels.
     
    top_n : int, default=28
    Maximum number of features to include in the boxplot grid.
     

    Returns
    -------
    None
     
    Notes
    -----
    Feature AUC values should be interpreted as univariate measures of
    discriminative ability and do not necessarily reflect the importance
    of features within the final multivariable model.
     
    Mann-Whitney U tests are two-sided and assess differences in feature
    distributions between classes.
    """


    # Ensure y is a 1D array
    y = np.array(y).ravel()

    # -----------------------------
    # 1. Compute AUC + MW p-values
    # -----------------------------
    results = []


    for feat in features_df["feature"]:


        if feat not in X.columns:
            print(f"Skipping {feat} (not in X)")
            continue
        
        # Extract frequency for this feature
        freq = features_df.loc[features_df["feature"] == feat, "frequency"].values[0]
        
        values = X[feat].values

        # AUC requires both classes present
        if len(np.unique(y)) != 2:
            raise ValueError("y must be binary")

        auc = roc_auc_score(y, values)

        # Mann-Whitney U test
        group1 = values[y == 0]
        group2 = values[y == 1]

        mw_stat, mw_p = mannwhitneyu(group1, group2, alternative="two-sided")

        results.append({
            "feature": str(feat),
            "feature_label": f"{str(feat)} (freq={freq})",
            "AUC": auc,
            "MW_p": mw_p,
            "frequency": freq
        })
    

    results_df = pd.DataFrame(results).sort_values("frequency", ascending=False).reset_index(drop=True)



    # Force to string to avoid MultiIndex issues
    results_df["feature"] = results_df["feature"].astype(str)
    results_df["feature_label"] = results_df["feature_label"].astype(str)

    # --- AUC horizontal bar plot (explicit order) ---
    plt.figure(figsize=(10, max(6, len(results_df) * 0.25)))
    y_order = results_df["feature_label"].tolist()  # preserve current sorted order
    # use matplotlib barh for robustness
    aucs = results_df["AUC"].values
    y_pos = np.arange(len(y_order))[::-1]  # reverse so top is highest in list
    plt.barh(y_pos, aucs)
    plt.yticks(y_pos, y_order)
    plt.xlabel("AUC")
    plt.title("Feature AUC values")
    plt.xlim(0.0, 1.0)
    plt.tight_layout()
    plt.show()

    # --- Mann-Whitney p-values horizontal bar plot (log-scaled x-axis) ---
    plt.figure(figsize=(10, max(6, len(results_df) * 0.25)))
    pvals = results_df["MW_p"].values
    # plot on log scale — use small positive floor to avoid log(0)
    eps = 1e-300
    plt.barh(y_pos, np.maximum(pvals, eps))
    plt.yticks(y_pos, y_order)
    plt.xlabel("p-value (log scale)")
    plt.xscale("log")
    plt.title("Mann-Whitney U Test p-values (log-scaled)")
    plt.tight_layout()
    plt.show()

    # --- Top-N boxplots in grid (fix indexing) ---
    n_plot = min(top_n, len(results_df))
    selected = results_df.head(n_plot).reset_index(drop=True)  # reset index to 0..n_plot-1

    n_cols = 4
    n_rows = int(np.ceil(n_plot / n_cols))
    plt.figure(figsize=(6 * n_cols, 4 * n_rows))

    for i, row in selected.iterrows():
        feat = row["feature"]
        label = row["feature_label"]
        ax = plt.subplot(n_rows, n_cols, i + 1)
        sns.boxplot(x=y, y=X[feat], ax=ax)
        ax.set_title(f"{label}\nAUC={row['AUC']:.2f}, p-val={row['MW_p']:.2f}")
        ax.set_xlabel("Class (y)")
        ax.set_ylabel("Value")

    # fill empty subplots if any
    total_subplots = n_rows * n_cols
    if total_subplots > n_plot:
        for j in range(n_plot, total_subplots):
            plt.subplot(n_rows, n_cols, j + 1)
            plt.axis('off')

    plt.tight_layout()
    plt.show()



def plot_feature_frequency(feature_freq_df, column_name = 'feature', fig_size = (8,8)):
    """
    Plot feature selection frequencies as a horizontal bar chart.

    Features are sorted by frequency so that the most frequently selected
    features appear at the top of the plot.

    Parameters
    ----------
    feature_freq_df : pandas.DataFrame
        DataFrame containing at least:
        - a column with feature names (default: 'feature')
        - a column named 'frequency' containing numeric counts
    column_name : str, optional
        Name of the column containing feature names.
        Default is 'feature'.

    Returns
    -------
    None
        Displays a matplotlib horizontal bar plot.

    Notes
    -----
    - The DataFrame must contain a column named 'frequency', or a different column name needs to be used
    - The plot is generated using matplotlib and displayed immediately
      via plt.show().
    """

    # Sort features by frequency so the largest is on top
    feature_freq_df_sorted = feature_freq_df.sort_values(by='frequency', ascending=True)

    # Plot
    plt.figure(figsize=fig_size)
    plt.barh(
        feature_freq_df_sorted[column_name],
        feature_freq_df_sorted['frequency'],
        color='steelblue'
    )

    plt.xlabel('Frequency', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.title('Feature Frequencies', fontsize=16, fontweight='bold')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()
    
    

# -----------------------------------------------------------------------------
#                       CONFUSION MATRICES ON TEST DATA
# -----------------------------------------------------------------------------



def plot_conf_matrix(y_true, y_proba, threshold = None, title = "Confusion Matrix"):
    
    """
    Plot a confusion matrix at a chosen classification threshold and
    report associated performance metrics with bootstrap confidence
    intervals.
     
    The threshold can be specified manually, determined using Youden's
    Index, or selected using the ROC top-left method. The function
    computes classification metrics, generates a colour-coded confusion
    matrix, and reports confidence intervals obtained through bootstrap
    resampling.
     
    Parameters
    ----------
    y_true : array-like
    Ground-truth binary class labels.
     
    y_proba : array-like
    Predicted probabilities for the positive class.
     
    threshold : float, str, or None, default=None
    Threshold used to convert probabilities into class predictions.
     
    Accepted values are:
    - float : user-specified threshold.
    - None : automatically use the Youden-optimal threshold.
    - 'youden' : use the Youden-optimal threshold.
    - 'top-left' : use the threshold nearest the ideal ROC point
    (0,1).
     
    title : str, default='Confusion Matrix'
    Title displayed above the confusion matrix.
     
    Returns
    -------
    threshold : float
    The threshold ultimately used for classification.
     
    Notes
    -----
    The following metrics are reported with 95% bootstrap BCa
    confidence intervals:
     
    - Accuracy
    - Sensitivity (Recall)
    - Specificity
    - Positive Predictive Value (PPV)
    - Negative Predictive Value (NPV)
     
    AUC and its confidence interval are also reported for reference.
    """
        
    
    # Calculate auc with CI analytically
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    
    auc, ci = roc_auc_score(y_true,
                            y_proba,
                            confidence_level=0.95)
    
    print(f"AUC: {auc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]\n")
    
    
    # Apply threshold

    if isinstance(threshold, (int, float)):
        print(f"Using user-defined threshold: {threshold:.3f}")

    elif threshold == None or threshold.lower() == "youden":
        J = tpr - fpr
        optimal_idx = np.argmax(J)
        threshold = thresholds[optimal_idx]
        print(f"Using Youden's Index threshold: {threshold:.3f}")
    elif isinstance(threshold, str) and threshold.lower() == "top-left":
        # Top-left method = minimize distance to (0,1)
        distances = np.sqrt((fpr ** 2) + ((1 - tpr) ** 2))
        optimal_idx = np.argmin(distances)
        threshold = thresholds[optimal_idx]
        print(f"Using Top-left threshold: {threshold:.3f}")

    else:
        raise ValueError("Threshold must be None, 'top-left', or a numeric value.")
        
        
    y_pred_thresh_test = (y_proba >= threshold).astype(int)
    
    #print(f"my stupid tests: {np.unique(y_pred_thresh_test)}")
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred_thresh_test)
    tn, fp, fn, tp = cm.ravel()
    
    # Print the results
    print(f"Confusion Matrix at threshold = {threshold:.3f}")
    print(f"[[TN: {tn}, FP: {fp}]\n [FN: {fn}, TP: {tp}]]\n")
    
    # Optionally print derived metrics too
    #sensitivity = tp / (tp + fn) if (tp + fn) else 0
    #specificity = tn / (tn + fp) if (tn + fp) else 
    
    # See here for differnet bootstrapping methods: https://en.wikipedia.org/wiki/Bootstrapping_(statistics)
    # I'm using BCA as seems to be widely agreed it's good
    accuracy, ci_acc = accuracy_score(y_true, y_pred_thresh_test, confidence_level=0.95, method='bootstrap_bca', n_resamples=5000)
    sensitivity, ci_sens  = tpr_score(y_true, y_pred_thresh_test, confidence_level=0.95, method='bootstrap_bca', n_resamples=5000)
    fpr, ci_fpr  = fpr_score(y_true, y_pred_thresh_test, confidence_level=0.95, method='bootstrap_bca', n_resamples=5000)
    specificity = 1-fpr
    
    #ppv = tp / (tp + fp) if (tp + fp) else 0
    #npv = tn / (tn + fn) if (tn + fn) else 0
    ppv, ci_ppv = ppv_score(y_true, y_pred_thresh_test, confidence_level=0.95, method='bootstrap_bca', n_resamples=5000)
    npv, ci_npv = npv_score(y_true, y_pred_thresh_test, confidence_level=0.95, method='bootstrap_bca', n_resamples=5000)
    
    
    
    
    labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
    labels = [f'{label}\n{value}' for label, value in zip(labels, cm.ravel())]
    labels = np.array(labels).reshape(2, 2)  # <-- reshape to 2x2 for annot
    
    
    colors = np.array([['#9ecae1', '#fcae91'],   # lighter blue, lighter red
                       ['#fcae91', '#9ecae1']])  # from ColorBrewer palettes
    
  
    
    cmap = ListedColormap(['#9ecae1', '#fcae91']) # We will map values 0 and 1 to these colors
    
    # But since the confusion matrix values aren't just 0 and 1, we can't use the cm values directly.
    # Instead, we create a matrix where each cell is 0 or 1 indicating TP/TN or FP/FN.
    
    # Map: 0 for true positives/negatives, 1 for false positives/negatives
    color_map_matrix = np.array([[0, 1],
                                 [1, 0]])
    
    plt.figure(figsize=(6,5))
    ax = sns.heatmap(cm, annot=labels, fmt='', cbar=False,
                     xticklabels=['Predicted Negative', 'Predicted Positive'],
                     yticklabels=['Actual Negative', 'Actual Positive'],
                     linewidths=0.5, linecolor='gray', square=True,
                     cmap=cmap, mask=None,
                     annot_kws={"fontsize":14, "color":"black"})  # <-- bigger, black font
    
    # Override the colors for each cell manually using the lighter shades
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True,
                                       edgecolor='gray',
                                       facecolor=colors[i, j],
                                       lw=1))
    
    plt.title(title, fontsize=16)
    #plt.ylabel('Actual', fontsize=14)
    #plt.xlabel('Predicted', fontsize=14)
    plt.yticks(rotation=90, fontsize=14)
    plt.xticks(rotation=0, fontsize=14)
    plt.show()
    
    
    print(f"Accuracy: {accuracy: .2f} [{ci_acc[0]:.2f}, {ci_acc[1]:.2f}]")
    print(f"Sensitivity (Recall for positive class): {sensitivity:.2f} [{ci_sens[0]:.2f}, {ci_sens[1]:.2f}]")
    print(f"Specificity (Recall for negative class): {specificity:.2f} [{1-ci_fpr[1]:.2f}, {1-ci_fpr[0]:.2f}]")
    print(f"PPV: {ppv:.2f}[{ci_ppv[0]:.2f}, {ci_ppv[1]:.2f}]")
    print(f"NPV: {npv:.2f}[{ci_npv[0]:.2f}, {ci_npv[1]:.2f}]")
    

    return(threshold)

