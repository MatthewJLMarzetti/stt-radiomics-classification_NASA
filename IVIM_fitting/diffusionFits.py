# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 10:17:08 2023

Different IVIM equations


@author: marzettm
"""


import numpy as np


def mono_exp(b, S0, D):
    # mono exponential decay
    return S0 * np.exp(-b * D)


def bi_exp(b, S0, D, f, D_star):
    # here refit everything
    return S0 * ((1 - f) * np.exp(-b * D) + f * np.exp(-b * D_star))


def calculate_aic(n, rss, k, corrected = True):
    '''

    For information on this equation, and also for when we select the modified AIC see:
        https://pmc.ncbi.nlm.nih.gov/articles/PMC6962709/
        
    Above paper recommends the corrected AIC when k>(n/40) 
        k - number of params, n - number of data points
        
    Section 3.3 in Model selection has both formulas.
    Note they are only true when measurement errors are independent, 
    identically and normally distributed with the same variance

    Parameters
    ----------
    n : TYPE
        number of data points, 12 for me in my IVIM acquisition
    rss : TYPE
        residual sum of squares. Can be single number or array
    k : TYPE
        number of params, 2 for mono exp (S0 and D), 4 for bi exp (S0, D, f, D*)

    corrected : bool, optional
        If True, calculate the corrected AIC (AICc); otherwise, calculate the standard AIC. 
        Default is True. For my case of 12 points, regardless if I'm looking at mono-exp (k = 2) or
        bi exp (k =4) we'll meet this criteria.

    Returns
    -------
    float
        AIC or AICc value.

    '''
    
    aic = 2*k + n * np.log(rss / n) # fourth equation in section 3.3

    # Apply correction for small sample size if corrected is True
    # second condition to avoid div by 0
    # Note there is a secomnd equation for AIC_c which ends in 2kn/(n-k-1) - this is equivalent to what we have below
    if corrected and n - k - 1 > 0:
        aic += (2 * k * (k + 1)) / (n - k - 1) # this is second eq in section 3.3


    return aic


def IVIM_model_notUsed(b, S0, D, f, D_star):
    '''
    
    Legacy formulation retained for comparison with alternative IVIM definitions.
    Not used in the final analysis.

    
    calculate IVIM signal given a b value, S0, D, f, D_star
    
    Equation same as: https://mriquestions.com/ivim.html#:~:text=S%2FSo%20%3D%20fe%E2%88%92,%2B%20(1%E2%88%92f)e&text=Here%20f%20(dimensionless)%20is%20the,diffusion%20coefficient%20D)%20take%20place.
    and also Lemke paper

    𝑆(𝑏)=𝑆(0)∙[(1−𝑓) 𝑒^(−𝑏∙𝐷 )+𝑓𝑒^(−𝑏∙(D + D*)) ]


    Parameters
    ----------
    b : TYPE
        DESCRIPTION.
    S0 : TYPE
        DESCRIPTION.
    D : TYPE
        DESCRIPTION.
    f : TYPE
        DESCRIPTION.
    D_star : TYPE
        DESCRIPTION.

    Returns
    -------
    float
        IVIM model - generates the signal when given IVIM parameters. Coded by ChatGPT

    '''
    return S0 * ((1 - f) * np.exp(-b * D) + f * np.exp(-b * (D + D_star)))
    
    
    
def IVIM_model(b, S0, D, f, D_star):

    """
    Bi-exponential IVIM signal model using the standard two-compartment formulation.

    This implementation follows the common IVIM convention in which the
    pseudo-diffusion term is modelled directly by D* (D_star), rather than
    combining tissue diffusion and pseudo-diffusion as (D + D*).

    The signal model is:

        S(b) = S0 * [ (1 − f) * exp(−b · D)  +  f * exp(−b · D*) ]

    where:
    - the first term represents true molecular (tissue) diffusion,
    - the second term represents perfusion-related pseudo-diffusion.

    This formulation corresponds to that used in Zyad et al. and is the
    model adopted for the final analyses in this thesis.

    Parameters
    ----------
    b : float or array_like
        Diffusion weighting factor(s) (s/mm²).
    S0 : float
        Signal intensity at b = 0.
    D : float
        Tissue diffusion coefficient (mm²/s).
    f : float
        Perfusion fraction (dimensionless, typically constrained to [0, 1]).
    D_star : float
        Pseudo-diffusion coefficient associated with microvascular perfusion
        (mm²/s), typically D* >> D.

    Returns
    -------
    float or ndarray
        Modelled diffusion-weighted signal S(b) corresponding to the input
        b-value(s) and IVIM parameters.

    Notes
    -----
    - This model assumes no exchange between compartments and independent
      contributions from diffusion and perfusion.
    - The parameters D, f, and D* are typically estimated via non-linear
      least squares fitting.
    """

    return S0 * ((1 - f) * np.exp(-b * D) + f * np.exp(-b * D_star))


def IVIM_model_2(b, D, f, D_star):
    '''
    Not used, I do fit S0 now
    
    One fewer input as S0 is no longer fit
    Instead all images need to be divided through by S(0) before starting


    𝑆(𝑏)/𝑆(0)=∙[(1−𝑓) 𝑒^(−𝑏∙𝐷_𝑡 )+𝑓𝑒^(−𝑏∙𝐷_𝑝 ) ]

    Parameters
    ----------
    b : TYPE
        DESCRIPTION.
    S0 : TYPE
        DESCRIPTION.
    D : TYPE
        DESCRIPTION.
    f : TYPE
        DESCRIPTION.
    D_star : TYPE
        DESCRIPTION.

    Returns
    -------
    float
        IVIM model - generates the signal when given IVIM parameters. Coded by ChatGPT

    '''
    return ((1 - f) * np.exp(-b * D) + f * np.exp(-b * D_star))



    
    
def ADC_model(b, S0, ADC):
    '''
    calculate DWI signal given a b value and ADC

    Parameters
    ----------
    b : float
        DESCRIPTION.
    S0 : float
        DESCRIPTION.
    ADC : float
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    
    return S0 * np.exp(-b*ADC)



'''
Functions for quick calculations
'''


def calc_quick_ADC(dwi_data, bvals):
    '''
    Calculate the Apparent Diffusion Coefficient (ADC) map from Diffusion Weighted Imaging (DWI) data
    using two specific b-values.

    The ADC map is calculated based on the logarithmic difference between two DWI images with distinct 
    b-values (b400 and b800 in this case). Any negative ADC values are replaced with -1 for later filtering.
    
    ADC < 0 now just returned as is
    
    Parameters
    ----------
    dwi_data : numpy.ndarray
        A 4D array of diffusion-weighted images with dimensions (x, y, z, b-value), where the last 
        dimension corresponds to images at different b-values.
        Shape: (x, y, z, b-values).
        
    bvals : numpy.ndarray
        A 1D array of b-values corresponding to the last dimension of dwi_data.
        Shape: (b-values,).

    Returns
    -------
    adc_map : numpy.ndarray
        A 3D array representing the calculated ADC map, with negative values replaced by -1. 
        Shape: (x, y, z).
    
    Notes
    -----
    - The function assumes that `dwi_data` contains at least two images acquired with different b-values 
      (typically b400 and b800).
    - A small epsilon value is used to avoid division by zero or taking the logarithm of zero.
    - Negative ADC values, which are non-physical, are set to -1 for further processing.
    '''
    # Uses the two highest b-values only to minimise perfusion contamination
    dwi_b400 = dwi_data[:,:,:,-2]
    dwi_b800 = dwi_data[:,:,:,-1]
   
    b1 = bvals[-2]
    b2 = bvals[-1]
    
    
    epsilon = 1e-6  # Small epsilon value to avoid division by zero
    dwi_b400[dwi_b400 < epsilon] = epsilon
    dwi_b800[dwi_b800 < epsilon] = epsilon

    adc_map = np.log(dwi_b400) - np.log(dwi_b800)
    adc_map /= (b2 - b1)


    #adc_map[adc_map < 0] = -1 # replace all -ve ADC values with -1. 
    # These can then be filtered out later

    return adc_map


def calc_quick_S0(adc_map, dwi_data, bvals):
    
    
    
    dwi_b400 = dwi_data[:,:,:,-2]
    b = bvals[-2]
    
    b0_image = dwi_b400 / np.exp(-b*adc_map)
    
    #b0_image[b0_image < 0] = -1
    
    return b0_image



def fit_adc_volume_vectorized(bvals, signal_volume):
    """
    Perform vectorized linear fitting for ADC calculation over a 4D signal volume.

    Tested in Test Code/Vect_ADC_Test.py - works as well as the 2 methods above over 14 patients. Differences are < 0.0001% (1e-6)

    Parameters:
    -----------
    bvals : ndarray
        Array of b-values (shape: [num_bvals]) used for diffusion imaging.
    signal_volume : ndarray
        4D array of diffusion signals (shape: [x, y, z, num_bvals]).
        Each voxel contains signal intensities corresponding to the b-values.

    Returns:
    --------
    adc_map : ndarray
        3D array of ADC values (shape: [x, y, z]).
    S0_map : ndarray
        3D array of estimated S0 values (shape: [x, y, z]).

    Notes:
    ------
    The mono-exponential model used is:
        S(b) = S0 * exp(-b * ADC)
    By taking the natural logarithm:
        ln(S(b)) = ln(S0) - b * ADC
    This allows linear fitting using least squares regression.
    This is of the form y = mX + c
    """
    # Step 1: Ensure no signal value is too small (e.g., 0 or negative)
    # Since log(0) is undefined, replace such values with a very small number (1e-10)
    signal_volume = np.clip(signal_volume, 1e-10, None)

    # Step 2: Take the natural logarithm of the signal
    # This changes the equation from exponential to linear form
    log_signal = np.log(signal_volume)

    # Step 3: Create the "design matrix" for linear regression
    # This is like setting up the x-axis (b-values) and a constant offset
    # Each row is [-b_value, 1] where the constant 1 is for the intercept (ln(S0))
    X = np.vstack((-bvals, np.ones_like(bvals))).T  # Shape: (num_bvals, 2)- matrix ocntaining coefficients - in this case b-values

    # Step 4: Reshape the signal volume for matrix operations
    # Flatten the spatial dimensions (x, y, z) into a single "voxels" dimension
    shape = signal_volume.shape[:3]  # Save the original shape (x, y, z)
    flat_signal = log_signal.reshape(-1, log_signal.shape[-1])  # Shape: (num_voxels, num_bvals)
    # this flattens the 3 spatial dimensions into 1 dimension, so each voxel taken seperately
    
    #print(flat_signal.shape) # this is number of voxels x number of b-values
    #print(flat_signal[0,:])
    
    
    
    # Step 5: Perform the linear regression for all voxels
    # np.linalg.lstsq finds the best-fit slope and intercept for all voxels
    # The result 'beta' contains [slope, intercept] for each voxel
    # lstsq returns beta, residuals, rank, s (singular values of X)
    beta  = np.linalg.lstsq(X, flat_signal.T, rcond=None)[0]  # Shape beta: (2, num_voxels)



    # Step 6: Extract ADC and S0 from the regression results
    # ADC is the negative slope, and S0 is the exponent of the intercept
    adc_map = beta[0, :].reshape(shape)  # Reshape back to 3D
    S0_map = np.exp(beta[1, :].reshape(shape))  # Reshape back to 3D and undo log



    num_voxels = flat_signal.shape[0]
    num_bvals = flat_signal.shape[1]

    # Step 7: Compute errors using covariance matrix
    if num_bvals == 2: # set to 0 if only 2 b-values
        adc_error = np.zeros_like(adc_map)
        S0_error = np.zeros_like(S0_map)
    else:
        
        
        # Compute covariance matrix: (X^T X)^(-1)
        XTX_inv = np.linalg.inv(X.T @ X)  # Shape: (2,2)





        # Compute residuals: difference between observed and fitted log-signals
        fitted_log_signal = (X @ beta).T  # Shape: (num_voxels, num_bvals)
        residuals = flat_signal - fitted_log_signal  # Shape: (num_voxels, num_bvals)
        
        
        
        # Compute residual variance per voxel
        sigma_sq = np.sum(residuals**2, axis=1) / (num_bvals - 2)  # Potential division by zero!
        
        # Compute standard errors
        adc_error = np.sqrt(XTX_inv[0, 0] * sigma_sq).reshape(shape)
        lnS0_error = np.sqrt(XTX_inv[1, 1] * sigma_sq).reshape(shape)
        S0_error = S0_map * lnS0_error # propogate error from ln(S0) to S0




    # Step 7: Return the maps of ADC and S0 values
    return adc_map, S0_map, adc_error, S0_error







    
    
    
def calc_quick_f(dwi_data, intercept):
    
    S0_measured = dwi_data[:,:,:,0] # note that this is also changing the array I pass in
    
    
    epsilon = 1e-8  # Small epsilon value to avoid division by zero
    S0_measured[S0_measured < epsilon] = epsilon
    
    f = (S0_measured - intercept) / S0_measured # checked this with chatGPT and equations agree
    

    # removing this, and do with the maps that are returned
    #f[f < 0] = -1 # set -ve fractions to -1 to proces later
    #f[f > 1] = 2 # set fractions > 1 to 2, to remove later
    
    return f














# Helper functions, but not actually fitting


def average_of_first_n_b_values(volume, n=6):
    """
    Create an average image of the first n b-values from a 4D volume.

    Parameters:
    - volume: 4D numpy array with shape (x, y, slice, b-value)
    - n: Number of b-values to average over (default is 6)

    Returns:
    - average_image: 3D numpy array with shape (x, y, slice) representing the average image
    """
    # Ensure the volume has at least n b-values
    if volume.shape[3] < n:
        raise ValueError("The volume does not have enough b-values.")

    # Slice the volume to get the first n b-values
    sliced_volume = volume[:, :, :, :n]

    # Compute the average along the b-value dimension (axis=3)
    average_image = np.mean(sliced_volume, axis=3)

    return average_image




