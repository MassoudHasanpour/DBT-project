# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 14:06:53 2022

@author: hasan
"""

import pandas as pd
import numpy as np
import pickle


Data = pd.read_csv(r'newData_No6_Ssn.csv', encoding= 'unicode_escape',header=None,  na_values="?" ,low_memory=False) # , error_bad_lines=False)
Data = np.array(Data)

Dataa = []
for i in range(len(Data)):
    if i>-2: # (Data[i,1]==6) or (Data[i,1]==100):
        Dataa.append(Data[i,:])
Dataa = np.array(Dataa)
Input_Stats_MR = []
Stats = []

for i in range(0,len(Dataa),2):

    Lbl = Dataa[i,3]
    Ssn = Dataa[i,2]
    Zn = Dataa[i,0]
    
    
    for j in range(50):
        try:
            
            # AST = Dataa[i,:]
            AS = Dataa[i, 4+j*50:3+j*50+1000]
            AS = AS[~np.isnan(AS)]
            AS = AS[0:min([1000,len(AS)])]
            
            # AST = Dataa[i+1,:]
            
            AS2 = Dataa[i+1, 4+j*50:3+j*50+1000]
            AS2 = AS2[~np.isnan(AS2)]
            AS2 = AS2[0:min([1000,len(AS2)])]
            
            Stats = []
            
            if len(AS2) > 300 and len(AS)  > 300:
            
                Stats.append(Lbl)
                Stats.append(np.mean(AS))
                Stats.append(np.std(AS))
                Stats.append(np.mean(AS2))
                Stats.append(np.std(AS2))
                
                for ij in range(1,100):
                    Stats.append(np.percentile(AS, ij))
                    Stats.append(np.percentile(AS2, ij))
                
                Stats.append(Zn)
                Stats.append(Ssn)
                
                Input_Stats_MR.append(Stats)

        except:
             print("!!!!!!")
    
df = pd.DataFrame(Input_Stats_MR)
df.to_csv('Input_Stats_Ocp_Main_Sub_5All.csv')

Input_Statss_MR = pd.DataFrame(Input_Stats_MR)
with open('Input_Statsss_Ocp_Main_Sub_5All.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(Input_Statss_MR, f)


Data = pd.read_csv(r'newData_No6_Ssn.csv', encoding= 'unicode_escape',header=None,  na_values="?" ,low_memory=False) # , error_bad_lines=False)
Data = np.array(Data)

Dataa = []
for i in range(len(Data)):
    if i>-2: # (Data[i,1]==6) or (Data[i,1]==100):
        Dataa.append(Data[i,:])
Dataa = np.array(Dataa)
Input_Stats_MR = []

for i in range(0,len(Dataa),2):

    Lbl = Dataa[i,3]
    Ssn = Dataa[i,2]
    Zn = Dataa[i,0]
    
    for j in range(50):
        try:
            
            # AST = Dataa[i,:]
            AS = Dataa[i, 4+j*50:3+j*50+1000]
            AS = AS[~np.isnan(AS)]
            AS = AS[0:min([1000,len(AS)])]
            
            # AST = Dataa[i+1,:]
            
            AS2 = Dataa[i+1, 4+j*50:3+j*50+1000]
            AS2 = AS2[~np.isnan(AS2)]
            AS2 = AS2[0:min([1000,len(AS2)])]
            
            Input_Stats_MR.append([Lbl, np.mean(AS), np.std(AS),
                                    np.percentile(AS, 1), np.percentile(AS, 99),
                                    np.percentile(AS, 5), np.percentile(AS, 95),
                                    np.percentile(AS, 50),
                                    np.percentile(AS, 45), np.percentile(AS, 55),
                                    np.percentile(AS, 40), np.percentile(AS, 60),
                                    np.percentile(AS, 30), np.percentile(AS, 70),
                                    np.percentile(AS, 15), np.percentile(AS, 85),
                                    np.percentile(AS, 10), np.percentile(AS, 90),
                                    np.mean(AS2), np.std(AS2),
                                    np.percentile(AS2, 1), np.percentile(AS2, 99),
                                    np.percentile(AS2, 5), np.percentile(AS2, 95),
                                    np.percentile(AS2, 50),
                                    np.percentile(AS2, 45), np.percentile(AS2, 55),
                                    np.percentile(AS2, 40), np.percentile(AS2, 60),
                                    np.percentile(AS2, 30), np.percentile(AS2, 70),
                                    np.percentile(AS2, 15), np.percentile(AS2, 85),
                                    np.percentile(AS2, 10), np.percentile(AS2, 90),
                                    Zn, Ssn])
        except:
             print("!!!!!!")
    
df = pd.DataFrame(Input_Stats_MR)
df.to_csv('Input_Stats_Ocp_Main_Sub_4.csv')

Input_Statss_MR = pd.DataFrame(Input_Stats_MR)
with open('Input_Statsss_Ocp_Main_Sub_4.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(Input_Statss_MR, f)


Data = pd.read_csv(r'newData_No6_Ssn.csv', encoding= 'unicode_escape',header=None,  na_values="?" ,low_memory=False) # , error_bad_lines=False)
Data = np.array(Data)

Dataa = []
for i in range(len(Data)):
    if i>-2: # (Data[i,1]==6) or (Data[i,1]==100):
        Dataa.append(Data[i,:])
Dataa = np.array(Dataa)
Input_Stats_MR = []

for i in range(0,len(Dataa),2):

    Lbl = Dataa[i,3]
    Ssn = Dataa[i,2]
    Zn = Dataa[i,0]
    
    for j in range(50):
        try:
            
            # AST = Dataa[i,:]
            AS = Dataa[i, 4+j*50:3+j*50+1000]
            AS = AS[~np.isnan(AS)]
            AS = AS[0:min([1000,len(AS)])]
            
            # AST = Dataa[i+1,:]
            
            AS2 = Dataa[i+1, 4+j*50:3+j*50+1000]
            AS2 = AS2[~np.isnan(AS2)]
            AS2 = AS2[0:min([1000,len(AS2)])]
            
            Input_Stats_MR.append([Lbl, np.mean(AS), np.std(AS),
                                    np.percentile(AS, 1), np.percentile(AS, 99),
                                    np.percentile(AS, 5), np.percentile(AS, 95),
                                    np.percentile(AS, 50),
                                    np.percentile(AS, 40), np.percentile(AS, 60),
                                    np.percentile(AS, 30), np.percentile(AS, 70),
                                    np.percentile(AS, 15), np.percentile(AS, 85),                                    np.mean(AS2), np.std(AS2),
                                    np.percentile(AS2, 1), np.percentile(AS2, 99),
                                    np.percentile(AS2, 5), np.percentile(AS2, 95),
                                    np.percentile(AS2, 50),
                                    np.percentile(AS2, 40), np.percentile(AS2, 60),
                                    np.percentile(AS2, 30), np.percentile(AS2, 70),
                                    np.percentile(AS2, 15), np.percentile(AS2, 85),
                                    Zn, Ssn])
        except:
             print("!!!!!!")
    
df = pd.DataFrame(Input_Stats_MR)
df.to_csv('Input_Stats_Emp_Main_Sub_4.csv')

Input_Statss_MR = pd.DataFrame(Input_Stats_MR)
with open('Input_Statsss_Emp_Main_Sub_4.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(Input_Statss_MR, f)

                        

# In_St = Input_Statss_MR.to_numpy()  
# Zones = In_St[: , 35]
# groups = In_St[: , 36]
# Unq_Zs = np.unique(Zones)
# Nan_Z = []
# Mn_Empty = []
# for i in Unq_Zs:
#     N_Em = []
#     Nr = 0
#     for j in range(len(Zones)):
#         if (Zones[j] == i and Nr < 11 and In_St[j , 0]==0):
#             N_Em.append(In_St[j, 1:36]) 
#             Nr+=1
#         elif Nr > 10:
#             break
#     try:   
#         Mn_Empty.append(np.mean(np.array(N_Em),axis=0))   
#     except:
#         Nan_Z.append(In_St[j, 35])
# MnE = []           
# for i in range(len(Unq_Zs)):
#     if Mn_Empty[i].size>2:
#         MnE.append(Mn_Empty[i])
# MnE = np.mean(np.array(MnE),axis=0)

# for i in range(len(Unq_Zs)):
#     if not Mn_Empty[i].size>2:
#         # MnE[13] = Unq_Zs[i]
#         # print(MnE[13])
#         Mn_Empty[i] = MnE
#         Mn_Empty[i][34] = Unq_Zs[i]
# Mn_Empty = np.array(Mn_Empty)
        
# for i in range(len(Unq_Zs)):

#     Mn_Empty[i,34] = Unq_Zs[i]        


# Input_Statss_MR_Norm = np.array(Input_Statss_MR)
# for i in range(len(Input_Statss_MR)):
#     for k in range(len(Unq_Zs)):
#         if Mn_Empty[k,34] == Input_Statss_MR_Norm[i,35]:
#             for j in range(1,35):
#                 Input_Statss_MR_Norm[i,j] = Input_Statss_MR_Norm[i,j]/Mn_Empty[k,j-1]

# Input_Statss_MR_Norm = pd.DataFrame(Input_Statss_MR_Norm)
# with open('Input_Statss_Norm_Main_Sub_2.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#     pickle.dump(Input_Statss_MR_Norm, f)
    

# Input_Statss_MR_Normm = pd.DataFrame(Input_Statss_MR_Norm)
# Input_Statss_MR_Normm.to_csv('Input_Statss_MR_Normm_Main_Sub_2.csv')    
    