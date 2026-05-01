# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:11:07 2024

Functions used for registration of images.

To see default input to itkelastix look at this:
    ?itk.ParameterObject.GetDefaultParameterMap
    
    
    


@author: MarzettM
"""



import itk
from scipy.ndimage import label, find_objects
from scipy.ndimage import binary_dilation
import skimage
import numpy as np





def dilate_mask(mask, scale_factor=1):
    
    """
    Dilates a binary lesion mask by an amount proportional to lesion size.

    This is intended to create a surrounding margin (e.g. for registration
    or contextual analysis) while avoiding a fixed kernel size that would
    over-dilate small lesions or under-dilate large ones.
    """

    # Calculate the target kernel size for dilation/erosion
   
        
    target_size = int(calculate_mask_size(mask) * scale_factor / 2)
    if target_size % 2 == 0: # kernel must be odd number (needs a central pixel)
        target_size += 1
    
    kernel = np.ones((target_size, target_size, target_size))
    
    # Adjust mask size
    dilated_mask_np = binary_dilation(mask, structure=kernel) 
    
    
    dilated_mask = itk.GetImageFromArray(dilated_mask_np.astype(np.uint8))
    # Copy metadata (spacing, origin, direction) from the original mask
    dilated_mask.SetSpacing(mask.GetSpacing())
    dilated_mask.SetOrigin(mask.GetOrigin())
    dilated_mask.SetDirection(mask.GetDirection())
    
    
    return dilated_mask



def calculate_mask_size(mask):
    """
    Calculate the size of the largest connected component in a binary mask.

    The function labels connected components in the binary mask, identifies 
    the largest connected component, and calculates its size as the mean 
    of its major and minor axis lengths.

    Parameters:
    -----------
    mask : np.ndarray
        A binary mask (2D or 3D) where foreground is represented by 1 and 
        background by 0.

    Returns:
    --------
    float
        The size of the largest connected component, calculated as the 
        mean of its major and minor axis lengths.

    Notes:
    ------
    - Uses `scipy.ndimage.label` to label connected components.
    - Relies on `skimage.measure.regionprops` to calculate properties of the components.
    - Only considers the largest connected component for size calculation.
    - Assumes a single dominant connected component per mask.
    - properties[0] is therefore treated as the lesion of interest.

    """
    


    labeled_mask, num_features = label(mask)
    properties = skimage.measure.regionprops(labeled_mask)
    mask_size = np.mean([properties[0].axis_major_length, properties[0].axis_minor_length])
    
    
    
    return mask_size


def DWI_reg_old(DWI_4D_image):
    '''
    Groupwise registration with PCA metric, takes all images simultaneously

    This one doens;t clip max and min values to max pre reg and 0 respectively

    Parameters
    ----------
    DWI_4D_image : itkImagePython
        DESCRIPTION.

    Returns
    -------
    registered_DWI_volume : itkImagePython
        DESCRIPTION.
    DWI_transform_parameters : elxParameterObjectPython
        DESCRIPTION.

    '''
    
    gridSpace =  (DWI_4D_image.shape[2]) * 0.2
    gridSpace = str(gridSpace)
    
    parameter_file = "F:/PhD/Prospective Study/Registration/itkElastix/ParameterFiles/"
    p_file = parameter_file + "par_groupwise_ADC-ABDOMEN.txt"
    #p_file = parameter_file + "par_groupwise_ADC-ABDOMEN_PCA1.txt" # comment out this line
    
    print("\nStarting registration")
    
    #print(f"Wooo, new gridSpace - it is {gridSpace}")
    
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterFile(p_file)
    parameter_object.SetParameter(0, "FinalGridSpacingInPhysicalUnits", gridSpace) # This can't be set as a dictionary like other parameters, not sure why but there is a method to do it
    a = parameter_object.GetParameter(0, "FinalGridSpacingInPhysicalUnits") # takes self (implicit) then index (0 as one file) and then key
    print(f"Grid space: {a}")
    registered_DWI_volume, DWI_transform_parameters = itk.elastix_registration_method(
        DWI_4D_image, DWI_4D_image,
        parameter_object=parameter_object,
        log_to_console=False)

    

    print("finished registration \n")
    
    #check_details(registered_DWI_volume, "Registered image")
    
    
    return registered_DWI_volume, DWI_transform_parameters    





def DWI_reg(DWI_4D_image):
    '''
    Groupwise registration with PCA metric, takes all images simultaneously

    Clamp min value to 0 and max to the max pre registration

    Parameters
    ----------
    DWI_4D_image : itkImagePython
        DESCRIPTION.

    Returns
    -------
    registered_DWI_volume : itkImagePython
        DESCRIPTION.
    DWI_transform_parameters : elxParameterObjectPython
        DESCRIPTION.

    '''
    
    gridSpace =  (DWI_4D_image.shape[2]) * 0.2
    gridSpace = str(gridSpace)
    
    parameter_file = "F:/PhD/Prospective Study/Registration/itkElastix/ParameterFiles/"
    p_file = parameter_file + "par_groupwise_ADC-ABDOMEN.txt"
    
    # stick with max as in some smaller tumours values in tumour can be > 99.9th percentile
    max_val = np.max(DWI_4D_image, axis = (1,2,3))#np.percentile(DWI_4D_image, 99.9, axis = (1,2,3)) 
    #print(f"maximum values in each b-value: {max_val}")
    #print(f"99th percentile values in each b-value: {np.percentile(DWI_4D_image, 99, axis = (1,2,3))}")
    #print(f"99.5th percentile values in each b-value: {np.percentile(DWI_4D_image, 99.5, axis = (1,2,3))}")
    #print(f"99.9th percentile values in each b-value: {np.percentile(DWI_4D_image, 99.9, axis = (1,2,3))}")
    #print(f"100th percentile values in each b-value: {np.percentile(DWI_4D_image, 100, axis = (1,2,3))}")

    
    print("\nStarting registration")
    
    #print(f"Wooo, new gridSpace - it is {gridSpace}")
    
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterFile(p_file)
    parameter_object.SetParameter(0, "FinalGridSpacingInPhysicalUnits", gridSpace) # This can't be set as a dictionary like other parameters, not sure why but there is a method to do it
    #a = parameter_object.GetParameter(0, "FinalGridSpacingInPhysicalUnits") # takes self (implicit) then index (0 as one file) and then key
    #print(f"Grid space: {a}")
    registered_DWI_volume, DWI_transform_parameters = itk.elastix_registration_method(
        DWI_4D_image, DWI_4D_image,
        parameter_object=parameter_object,
        log_to_console=False)
    
    
    DWI_array = itk.GetArrayFromImage(DWI_4D_image)
    
    print(f"Shape of array: {DWI_array.shape}")
    
    for i in range(DWI_array.shape[0]):  
        DWI_array[i] = np.clip(DWI_array[i], a_min = 0, a_max = max_val[i])
        
    clipped_reg_DWI_volume = itk.GetImageFromArray(DWI_array)
    clipped_reg_DWI_volume.SetOrigin(DWI_4D_image.GetOrigin())
    clipped_reg_DWI_volume.SetSpacing(DWI_4D_image.GetSpacing())
    clipped_reg_DWI_volume.SetDirection(DWI_4D_image.GetDirection())
    

    print("finished registration \n")
    
    #check_details(registered_DWI_volume, "Registered image")
    
    
    return clipped_reg_DWI_volume, DWI_transform_parameters    
    



def anatomical_reg_rigid(reg_DWI_b0_img, anatomical_img, mask = None):
    # note all inputs are itk objects, as is the mask
    # rigid registration only
    

    parameter_object = itk.ParameterObject.New()
    default_rigid_parameter_map = parameter_object.GetDefaultParameterMap('rigid')
    default_rigid_parameter_map['Metric'] = ['AdvancedMattesMutualInformation']
    parameter_object.AddParameterMap(default_rigid_parameter_map)



    if mask == None:
        result_image, result_transform_parameters = itk.elastix_registration_method(
            reg_DWI_b0_img, anatomical_img,
            parameter_object=parameter_object,
            log_to_console=False
        )
    else:
        result_image, result_transform_parameters = itk.elastix_registration_method(
            reg_DWI_b0_img, anatomical_img,
            parameter_object = parameter_object,
            fixed_mask = mask,  # Only the fixed mask is provided#
            log_to_console=False
        )
        
        

    return result_image, result_transform_parameters


def anatomical_reg_rigid_retro(fixed, moving):
    # note all inputs are itk objects, as is the mask
    # rigid registration only
    
    
    parameter_file = "F:/PhD/Prospective Study/Registration/itkElastix/ParameterFiles/"
    p_file = parameter_file + "retro_registration.txt"
    
    parameter_object = itk.ParameterObject.New()
    parameter_object.AddParameterFile(p_file)

    #parameter_object = itk.ParameterObject.New()
    #default_rigid_parameter_map = parameter_object.GetDefaultParameterMap('rigid')
    #default_rigid_parameter_map['Metric'] = ['AdvancedMattesMutualInformation']
    #parameter_object.AddParameterMap(default_rigid_parameter_map)



    result_image, result_transform_parameters = itk.elastix_registration_method(
            fixed, moving,
            parameter_object=parameter_object,
            log_to_console=False
    )

        

    return result_image, result_transform_parameters




def anatomical_reg_affine(reg_DWI_b0_img, anatomical_img, mask = None):
    # Need to update with mask
    # note both of these are itk objects
    

    # B-spline interpolation - start with an affine
    parameter_object_bSpline = itk.ParameterObject.New()
    
    default_affine_parameter_map = parameter_object_bSpline.GetDefaultParameterMap('affine',4)
    default_affine_parameter_map['FinalBSplineInterpolationOrder'] = ['1'] #was 3, but that can introduce the big max and min values. 1 wll cause blurring, but not a huge issue as downsampling anyway to match DWI reg
    default_affine_parameter_map['Metric'] = ['AdvancedMattesMutualInformation']
    default_affine_parameter_map['MaximumNumberOfIterations'] = ['512']
    parameter_object_bSpline.AddParameterMap(default_affine_parameter_map)


    if mask == None:
        result_image, result_transform_parameters = itk.elastix_registration_method(
            reg_DWI_b0_img, anatomical_img,
            parameter_object=parameter_object_bSpline,
            log_to_console=False)
    else:
        result_image, result_transform_parameters = itk.elastix_registration_method(
            reg_DWI_b0_img, anatomical_img,
            parameter_object=parameter_object_bSpline,
            fixed_mask = mask,  # Only the fixed mask is provided#
            log_to_console=False)
    

    return result_image, result_transform_parameters





def anatomical_reg_bspline(reg_DWI_b0_img, anatomical_img, mask = None, gridSpace = 0): # Thought about passing in the reg parameters but decided against it now, just calculate from DWI image best
    '''
    Following discussion with Stefan 15/11/24 have decided to go rigid and then 
    b-spline, rather than affine - then bspline
    # Need to update with mask
    '''    




    if gridSpace == 0:
        gridSpace =  (reg_DWI_b0_img.shape[2]) * 0.2 # in dwi image each pixel is 1 mm, so this means we use 32 for a 160 x 160 image, and more for vigger images
    
    
    #print(f"Calc grid space: {gridSpace}")
    
    # B-spline interpolation - start with a rigid first
    parameter_object_bSpline = itk.ParameterObject.New()

    default_rigid_parameter_map = parameter_object_bSpline.GetDefaultParameterMap('rigid')
    default_rigid_parameter_map['FinalBSplineInterpolationOrder'] = ['1']
    default_rigid_parameter_map['Metric'] = ['AdvancedMattesMutualInformation']
    default_rigid_parameter_map['MaximumNumberOfIterations'] = ['1000']
    parameter_object_bSpline.AddParameterMap(default_rigid_parameter_map)
    

    default_bspline_parameter_map = parameter_object_bSpline.GetDefaultParameterMap('bspline', 4, gridSpace) # crashes if this number (4) above 4, despite what ChatGPT says
    default_bspline_parameter_map['Metric'] = ['AdvancedMattesMutualInformation']
    default_bspline_parameter_map['FinalBSplineInterpolationOrder'] = ['1']
    default_bspline_parameter_map['MaximumNumberOfIterations'] = ['1000']
    parameter_object_bSpline.AddParameterMap(default_bspline_parameter_map)
    
    
    
    
    #a = parameter_object_bSpline.GetParameter(1, "FinalGridSpacingInPhysicalUnits") # takes self (implicit) then index (1 as second map) and then key
    #print(f"Grid space from Elastix: {a}")
   

    
    
    if mask == None:
        
        result_image, result_transform_parameters = itk.elastix_registration_method(
            reg_DWI_b0_img, anatomical_img,
            parameter_object=parameter_object_bSpline,
            log_to_console=False
        )
    
        #check_details(registered_DWI_volume, "Registered anatomical image")
    
    else:
        result_image, result_transform_parameters = itk.elastix_registration_method(
            reg_DWI_b0_img, anatomical_img,
            parameter_object = parameter_object_bSpline,
            fixed_mask = mask,  # Only the fixed mask is provided#
            log_to_console=False
        )
      


     

    
    return result_image, result_transform_parameters





def resize_anatomical(source_image, target_image):
    """
    Resamples the source image to match the dimensions, spacing, and origin of the target image.

    Args:
        source_image: The itk image to be resampled.
        target_image: The itk image whose dimensions, spacing, and origin will be matched.

    Returns:
        resampled_image: The resampled itk image with the same size and spacing as the target_image.
        """

    # Get the size, spacing, origin, and direction of the target image
    target_size = target_image.GetLargestPossibleRegion().GetSize()
    target_spacing = target_image.GetSpacing()
    target_origin = target_image.GetOrigin()
    target_direction = target_image.GetDirection()
    
    # Set up the resample filter
    resample_filter = itk.ResampleImageFilter.New(Input=source_image)

    # Set parameters of the resample filter to match the target image
    resample_filter.SetSize(target_size)
    resample_filter.SetOutputSpacing(target_spacing)
    resample_filter.SetOutputOrigin(target_origin)
    resample_filter.SetOutputDirection(target_direction)

    # Set interpolator (you can choose Linear, NearestNeighbor, etc.)
    resample_filter.SetInterpolator(itk.LinearInterpolateImageFunction.New(source_image))

    # Update the filter and get the resampled image
    resample_filter.Update()
    resampled_image = resample_filter.GetOutput()

    return resampled_image