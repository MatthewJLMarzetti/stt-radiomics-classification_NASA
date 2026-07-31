# -*- coding: utf-8 -*-
"""
Custom ML pipeline steps for feature reduction


Created on Wed Aug 13 11:43:00 2025

@author: marzettm
"""

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np




class CoefficientOfVariationFilter(BaseEstimator, TransformerMixin):
    
    """
    Feature selector that removes features with low coefficient of variation.

    The coefficient of variation (CV) is defined as the ratio of the standard deviation
    to the absolute mean of a feature. This transformer removes features whose CV falls
    below a specified threshold, under the assumption that low-variability features are
    less informative.

    Parameters
    ----------
    threshold : float, default=0.01
        Minimum coefficient of variation required for a feature to be retained.
        Features with CV below this threshold will be removed
        while features with CV equal to or greater than the threshold are retained..

    Attributes
    ----------
    keep_indices_ : ndarray of int
        Indices of features that passed the CV threshold.
    n_features_ : int
        Total number of features in the input data.

    Methods
    -------
    fit(X, y=None)
        Computes the CV for each feature and identifies those to retain.
    
    transform(X)
        Reduces the input data to only the selected features.
    
    get_support(indices=False)
        Returns a mask or indices of the selected features.
    """

    
    
    def __init__(self, threshold=0.01):  # 0.01 = 1%
        self.threshold = threshold  # Minimum CV threshold (e.g., 0.01 = 1%)
        self.keep_indices_ = None

    def fit(self, X, y=None):
        #  Ensure input is a DataFrame for easier column-wise operations
        X = pd.DataFrame(X)
        self.n_features_ = X.shape[1]   # Store number of features
        
        
        # Compute mean and standard deviation for each feature and compute coefficient of variation
        means = X.mean(axis=0)
        stds = X.std(axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            cv = stds / means.abs()
            cv = cv.fillna(0)
            
        # Keep indices of features with CV above the threshold   
        self.keep_indices_ = np.where(cv >= self.threshold)[0]
        
        
        #print(f"Coefficient of Variation filter: {len(self.keep_indices_)} features pass the threshold of {self.threshold}")
        return self

    def transform(self, X):
        # Select only the features that passed the CV threshold
        return X.iloc[:, self.keep_indices_] if hasattr(X, 'iloc') else X[:, self.keep_indices_]
    
    
    def get_support(self, indices=False):
        # Return a boolean mask or indices of selected features
        if self.keep_indices_ is None or self.n_features_ is None:
            raise RuntimeError("fit must be called before get_support.")

        support_mask = np.zeros(self.n_features_, dtype=bool)
        support_mask[self.keep_indices_] = True
        return self.keep_indices_ if indices else support_mask
    
    
    
class CorrelationFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.9, method='pearson'):
    
        """
        Feature selector that removes highly correlated features.
    
        This transformer computes pairwise correlations between features and removes
        one feature from each pair whose correlation exceeds a specified threshold.
        It retains only one representative feature from each correlated group.
        
        When two features exceed the correlation threshold,
        the feature appearing later in the input column order is removed
    
        Parameters
        ----------
        threshold : float, default=0.9
            Correlation threshold above which features are considered redundant.
            One feature from each correlated pair will be removed.
    
        method : {'pearson', 'spearman', 'kendall'}, default='pearson'
            Method used to compute pairwise correlations.
    
        Attributes
        ----------
        keep_columns_ : list of str
            Names of features that are retained after filtering.
        
        all_columns_ : Index
            Original column names of the input data.
    
        Methods
        -------
        fit(X, y=None)
            Computes the correlation matrix and identifies features to retain.
        
        transform(X)
            Reduces the input data to only the selected features.
        
        get_support(indices=False)
            Returns a mask or indices of the selected features.
            
        """
        
        
        self.threshold = threshold
        self.method = method
        self.keep_columns_ = None
        self.all_columns_ = None  # store all original column names

    def fit(self, X, y=None):
        # Ensure input is a DataFrame to compute correlations by column name
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.all_columns_ = X.columns  # store all columns
        # Compute absolute correlation matrix
        corr_matrix = X.corr(method = self.method).abs()
        # Extract upper triangle of the correlation matrix to avoid duplicate pairs
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Identify columns to drop: those with correlation > threshold with any other column
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > self.threshold)]
        self.keep_columns_ = [col for col in X.columns if col not in to_drop] # Keep only the columns not marked for removal

        # Debugging print statement
        #print(f"Correlation filter: {len(self.keep_columns_)} features remain after filtering with a threshold of {self.threshold}")

        return self
    

    def transform(self, X):
        # Return filtered features as a datafrme
        if isinstance(X, pd.DataFrame):
            return X[self.keep_columns_].values
        else:
            # Convert to DataFrame to support column names lookup
            df = pd.DataFrame(X, columns=self.all_columns_)
            return df[self.keep_columns_].values

    def get_support(self, indices=False):
        # Return a boolean mask or indices of selected features
        if self.all_columns_ is None or self.keep_columns_ is None:
            raise RuntimeError("fit must be called before get_support.")

        support_mask = [col in self.keep_columns_ for col in self.all_columns_]
        support_mask = np.array(support_mask, dtype=bool)

        if indices:
            return np.where(support_mask)[0]
        return support_mask
    
    
    
# Define a scorer that returns spec at 95% sensitivity
# check this: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
# Not used as doesn't add much


from sklearn.metrics import roc_curve


def specificity_at_95_sensitivity_score(y_true, y_proba):

    """
    Compute specificity at a minimum sensitivity of 95%.


    Parameters
    ----------
    y_true : array-like
    True binary class labels.

    y_proba : array-like
    Predicted probabilities for the positive class.

    Returns
    -------
    float
    Specificity corresponding to the first ROC threshold
    achieving at least 95% sensitivity. Returns 0.0 if
    no threshold achieves the target sensitivity.
    """
    
    
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) == 0:
        return 0.0  # or np.nan if you want to penalise models failing to reach target sensitivity
    best_idx = idx[0]
    specificity = 1 - fpr[best_idx]
    return specificity

    # use as follows:
    #specificity_scorer = make_scorer(specificity_at_95_sensitivity_score, needs_proba=True)