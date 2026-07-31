# -*- coding: utf-8 -*-
"""
Extracting all features from retrospective data using automatic PyRadiomics 
parameters - saved in params_file_auto



Created on Wed Aug 13 09:44:50 2025

@author: marzettm
"""


import pandas as pd
from radiomics import featureextractor
import os
from tqdm import tqdm 
import itk
import numpy as np


params_file_auto = r"D:\Radiomics\RetrospectiveData\MyVersions\PyRadiomics_ParamFiles\PyRadiomicsParams_AutoPreProcessing.yaml"
extractor_auto = featureextractor.RadiomicsFeatureExtractor(params_file_auto)


image_folder = r"D:\RetrospectiveData\Registered_T1_T2FS"
mask_folder = r"D:\RetrospectiveData\Registered_T1_T2FS\Final_masks"

info_df = pd.read_csv(r"F:\PhD\Retrospective data\radiomics_clin_info_final.csv")


all_results_auto = []

exclude_list = []


for _, row in tqdm(info_df.iterrows(), total = len(info_df), desc = "Extracting radiomics features"):
    pid = row["PseudoPatientID"]
    
    print(pid)

    t1_img_path = os.path.join(image_folder, pid+"_0000.nii.gz")
    t2_img_path = os.path.join(image_folder, pid+"_0001.nii.gz")
    mask_path = os.path.join(mask_folder, pid + ".nii.gz")
    
    #print(t1_img_path)
    #print(mask_path)
    
    mask = itk.imread(mask_path)
    
    if np.sum(mask) < 30:
        print(f"{pid} has no mask data, as nnUnet didn't work and there is no pathology report to speak of, so skipping")
        exclude_list.append(pid)
        continue
    
    
    result_t1_auto = extractor_auto.execute(t1_img_path, mask_path)
    result_t2_auto = extractor_auto.execute(t2_img_path, mask_path)
    
    
    
    
    
    
    # Clean metadata keys (remove diagnostics etc.)
    def clean_results(d):
        for key in list(d.keys()):
            if key.startswith("diagnostics") or key in ["general_info", "image", "mask"]:
                d.pop(key, None)
        return d
    
    result_t1_auto = clean_results(result_t1_auto)
    result_t2_auto = clean_results(result_t2_auto)
    
    features_auto = {"PseudoPatientID": pid}
    features_auto.update({f"T1_auto_{k}": v for k, v in result_t1_auto.items()})
    features_auto.update({f"T2_auto_{k}": v for k, v in result_t2_auto.items()})

    all_results_auto.append(features_auto)
    
    
output_csv_auto = r"D:\Radiomics\RetrospectiveData\MyVersions\radiomics_features_auto.csv"
df_results_auto = pd.DataFrame(all_results_auto)
df_results_auto.to_csv(output_csv_auto, index=False)









