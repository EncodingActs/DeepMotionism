# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 13:52:57 2021

@author: fadel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections
import pickle


file_path = "./data/extracted/Tung-Kong-Fodei-Daai-Paal-20141021_rot.csv"

#loaded_data = np.genfromtxt(file_path,delimiter=',',skiprows=1)
data = pd.read_csv(file_path)

dummys = []
features_1 = []
features_2 = []
features_3 = []
features_4 = []
all_columns = []
features_to_delete = ['Mesh.x','Mesh.y','Mesh.z','time',
                      'New_kungfumaster.z', 'New_kungfumaster.x',
                      'New_kungfumaster.y', 'Reference.z', 'Reference.x',
                      'Reference.y']

#-----Feature selection 

for col in data.columns :
    
    all_columns.append(col)
    
    if col.count("Dummy") > 0:
        dummys.append(col)
        features_to_delete.append(col)
    
    elif col.count("Roll") > 0 :
        features_to_delete.append(col)
    
    elif col.count("1") > 0 :
        features_1.append(col)
        
    elif col.count("2") > 0 :
        features_2.append(col)
    
    elif col.count("3") > 0 :
        features_3.append(col)
    
    elif col.count("4") > 0 :
        features_4.append(col)
       

indx = []
highest_frequency = []
selected_features = []

#--Sorting features by variances
for i in range(len(features_1)) :
    variances = np.array([data[features_1[i]].var(),data[features_2[i]].var(),
                 data[features_3[i]].var(),data[features_4[i]].var()])
    best = np.argmax(variances) + 1
    indx.append(best)

#-- Selecting the feature with highest variance between xx1 xx2 xx3 xx4
#and along the three axes
for i in range(len(indx)//3 ):    
    frequency = collections.Counter( [indx[::3][i], indx[1::3][i], indx[2::3][i]] )
    highest_frequency = sorted(frequency)[-1]
        
    j = i*3
    if highest_frequency == 1 :
        selected_features.append(features_1[j])
        selected_features.append(features_1[j+1])
        selected_features.append(features_1[j+2])
        
    if highest_frequency == 2 :
        selected_features.append(features_2[j])
        selected_features.append(features_2[j+1])
        selected_features.append(features_2[j+2])
    
    if highest_frequency == 3 :
        selected_features.append(features_3[j])
        selected_features.append(features_3[j+1])
        selected_features.append(features_3[j+2])
        
    if highest_frequency == 4 :
        selected_features.append(features_4[j])
        selected_features.append(features_4[j+1])
        selected_features.append(features_4[j+2])

#--Extract unwanted features names and update
features_1_2_3_4 = features_1 + features_2 + features_3 + features_4
unwanted_1234 = list(set(features_1_2_3_4) ^ set(selected_features) )
features_to_delete = features_to_delete + unwanted_1234

#--Update Selected features
result = set(all_columns) ^ set(features_to_delete) 
selected_features = sorted(list(result))

# Uncomment the line below to add more features with high variance
#selected_features = sorted(list(result) + features_to_delete[60:] )

print("There are :",len(selected_features)," selected features")

#with open("selected_features_labels_highestFrequency.txt","wb") as file :
    #pickle.dump(selected_features,file)

"""
fig, ax1 = plt.subplots(2,1,figsize=(10,10))
ax1[0].plot(data[selected_features].var(axis=0).to_numpy())
ax1[0].set_title("Selected")

ax1[1].plot(data[features_to_delete[60:]].var(axis=0).to_numpy())
ax1[1].set_title("Deleted")


fig, ax = plt.subplots(2,2,figsize=(10,10))
ax[0,0].plot(data[features_1].var(axis=0).to_numpy())
ax[0,0].set_title("1s")
ax[0,0].set(ylabel = 'Variance' )

ax[0,1].plot(data[features_2].var(axis=0).to_numpy())
ax[0,1].set_title("2s")
ax[0,1].set(ylabel='Variance')

ax[1,0].plot(data[features_3].var(axis=0).to_numpy())
ax[1,0].set_title("3s")
ax[1,0].set(ylabel='Variance')

ax[1,1].plot(data[features_4].var(axis=0).to_numpy())
ax[1,1].set_title("4s")
ax[1,1].set(ylabel='Variance')
"""