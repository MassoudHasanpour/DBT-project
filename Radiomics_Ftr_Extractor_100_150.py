
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 02:05:21 2022

@author: User
"""

import pandas as pd
# import radiomics
from radiomics import featureextractor
# import re
# import pydicom
# import numpy as np
# import SimpleITK as sitk
# from skimage import draw
import os
from pathlib import Path




def featureVector_extractor(imageName, maskName, paramsFile):
    # Something went wrong, in this case PyRadiomics will also log an error
    if imageName is None or maskName is None:
        print('Error getting testcase!')
        exit()
    extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile)
    featureVector = extractor.execute(imageName, maskName)
    return featureVector



PATHs = pd.read_csv('PATHs.csv')
Unqs = PATHs['descriptive_path'].unique()
Unqs_classic_path = PATHs['classic_path'].unique()
Coulums = pd.read_csv('Coulumns.csv')

j = 0

features_list =[]
features_list_breast =[]

RadiomicFeatures_Norm_0_done = pd.read_csv('RadiomicFeatures_Norm_0.csv')
patients_done = RadiomicFeatures_Norm_0_done["('Patient_ID',)"]

for i in range(116, 150): # len(Unqs)):
    features =[]
    features_breast =[]
    
    try:
    
        indices = PATHs.descriptive_path[PATHs.descriptive_path == Unqs[i]].index.tolist()
        
        pt = "D:/Projects/DBT project/"+ "Patients with benign cancer" + Unqs[i][27:]
        pt = pt.replace('/', '\\' )
        pt = pt.replace('0000-', '0000-NA-' )
        
        ptt = Path(pt)
        # dcm_path = os.path.join("E:\\" , ptt.replace('/', '\\' ))
        view = PATHs['View'][indices[0]]
        # pt_Img = Path(pt.replace('.dcm', '_crrct.dcm' ))        
        # pt_mask = Path(ptt.replace('.dcm', '_mask.dcm' ))
        
        if 'cc' in view: # not (any(Unqs[i][28:38] in p for p in patients_done)):
            pt_dir = Path(pt[:-8]) 
            ptt_dir = Path(pt_dir)
            os.chdir(ptt_dir)
            
            features.append(PATHs['PatientID'][indices[0]])
            features_breast.append(PATHs['PatientID'][indices[0]])
            
            featureVector = featureVector_extractor('1-1_crrct.dcm', '1-1_mask.dcm', 'D:/Projects/DBT project/Patients with benign cancer/Pyradiomics_Params.yaml')
            featureVector_breast = featureVector_extractor('1-1_crrct.dcm', '1-1_Breast_mask.dcm', 'D:/Projects/DBT project/Patients with benign cancer/Pyradiomics_Params.yaml')
            for j in range(1,len(Coulums)):
                features.append(featureVector[Coulums['Coloumns'][j]])
                features_breast.append(featureVector_breast[Coulums['Coloumns'][j]])
            
            features_list.append(features)
            features_list_breast.append(features_breast)
        
        print(PATHs['PatientID'][indices[0]] + "....... done")
        j += 1
        Num = j%2
        df = pd.DataFrame (features_list, columns = Coulums)
        F_N = "D:/Projects/DBT project/Patients with benign cancer/" + 'RadiomicFeatures_100_150_2' + str(Num) + '.csv'
        df.to_csv(F_N)
        df = pd.DataFrame (features_list_breast, columns = Coulums)
        F_N = "D:/Projects/DBT project/Patients with benign cancer/" + 'RadiomicFeatures_100_150_breast_2' + str(Num) + '.csv'
        df.to_csv(F_N)
    except Exception as e:
        print(e)
        try:
            indices = PATHs.descriptive_path[PATHs.descriptive_path == Unqs[i]].index.tolist()
            pt = "D:/Projects/DBT project/"+ "Patients with benign cancer" + PATHs['classic_path'][indices[0]][27:]
            pt = pt.replace('/', '\\' )
            pt = pt.replace('0000-', '0000-NA-' )
            
            ptt = Path(pt)
            # dcm_path = os.path.join("E:\\" , ptt.replace('/', '\\' ))
            view = PATHs['View'][indices[0]]
            # pt_Img = Path(pt.replace('.dcm', '_crrct.dcm' ))        
            # pt_mask = Path(pt.replace('.dcm', '_mask.dcm' ))
            if 'cc' in view: # not (any(Unqs[i][28:38] in p for p in patients_done)):
            
                pt_dir = Path(pt[:-8]) 
                ptt_dir = Path(pt_dir)
                os.chdir(ptt_dir)
                
                features.append(PATHs['PatientID'][indices[0]])
                features_breast.append(PATHs['PatientID'][indices[0]])
                
                featureVector = featureVector_extractor('1-1_crrct.dcm', '1-1_mask.dcm', 'D:/Projects/DBT project/Patients with benign cancer/Pyradiomics_Params.yaml')
                featureVector_breast = featureVector_extractor('1-1_crrct.dcm', '1-1_Breast_mask.dcm', 'D:/Projects/DBT project/Patients with benign cancer/Pyradiomics_Params.yaml')
                for j in range(1,len(Coulums)):
                    features.append(featureVector[Coulums['Coloumns'][j]])
                    features_breast.append(featureVector_breast[Coulums['Coloumns'][j]])
                
                features_list.append(features)
                features_list_breast.append(features_breast)
            
            print(PATHs['PatientID'][indices[0]] + "....... done")
            j += 1
            Num = j%2
            df = pd.DataFrame (features_list, columns = Coulums)
            F_N = "D:/Projects/DBT project/Patients with benign cancer/" + 'RadiomicFeatures_100_150_2' + str(Num) + '.csv'
            df.to_csv(F_N)
            df = pd.DataFrame (features_list_breast, columns = Coulums)
            F_N = "D:/Projects/DBT project/Patients with benign cancer/" + 'RadiomicFeatures_100_150_breast_2' + str(Num) + '.csv'
            df.to_csv(F_N)
        except Exception as e:
            print(e)
            print("D:\\Projects\\DBT project\\"+ "Patients with benign cancer" + PATHs['classic_path'][indices[0]][27:])