"""
Created on Mon Mar 15 17:47:07 2021

@author: fadel
"""

from src.deepArchitecture import train_model, initialize_model,transform_data
import torch
from src.dataPreparation import data_preparation
import time
import matplotlib.pyplot as plt
import os
import pickle


            
#--Creating an instance of data_preparation
data = data_preparation()

#---- Setting parameters
data.set_imaging_parameters(do_recurrenceP = False, RP_threshold = 'point',
                               RP_timeDelay = 1, RP_dimension = 0.25,
                               do_GAF = False, GAF_method = 'summation',
                               do_MTF = False,
                               square_reshape = True,
                               vertical_stacking = False,
                               horizontal_stacking = False) 
 
data.set_motionWords_parameters(number_of_frames=16,overlaping_degree=0.75)
data.set_featureExpansion_parameters(do_PCA=True,number_of_components=50)

#--Mining csv and computing and storing motion words
data.csv_files_mining(select_high_variance_features=True,expand_features=True)
data.compute_motion_words(divide_sequence=0)

#-------------------------- Initialize ---------------------------------
model_name = 'resnext50'  # It can be 'resnet18' 'resnext50' or 'inceptionV3'.
do_feature_extract=False
print('Only training last FC layer :', do_feature_extract, "; Architecture : " ,model_name)
model , input_size = initialize_model(model_name, num_classes=700,
                                      feature_extract= do_feature_extract, use_pretrained=True)

#-------------------------- Training -------------------------------
"""
We are training the network 3 times using differents ways for computing the training dataset.
e trainings will follow up on the previous learned weights.
"""


paths_to_training_dataset = ['./../data/datasets/training/training_dataset_5similarity_soft_dtw',
                             './../data/datasets/training/training_dataset_5similarity_gak',
                             './../data/datasets/training/training_dataset_5similarity_dtw' ]

transformer = transform_data(input_size,is_ts_images=False)
batch_size = 10

#--Do this if previous_model_path exists otherwise use --> empty ""
previous_model_path = "" #'../data/models/model_resnet18_training_dataset_5similarity_gak.pth'
labels =[]
training_losses = []
use_multi_gpus = False #True

for indx,path in enumerate(paths_to_training_dataset) :
    print(f"\n Training with : {path}")
    start = time.time()

    #--Loading training path
    data.path_to_training_dataset = path
    dataloader = data.load_training_dataset()
    
    if os.path.exists(previous_model_path) : 
        #--Loading the weights previously learned if applicable
        #--Here we reload a new model on the CPU in case one is training on multiple GPUs
        #--The model will be transfered on the GPUS in train_model
        model , _ = initialize_model(model_name,700,do_feature_extract,False)
        model.load_state_dict(torch.load(previous_model_path,map_location=torch.device("cpu")),strict=True)
        print(f"Model : {previous_model_path} --> loaded.")
	
            
    model,loss_history = train_model(model, model_name, dataloader, data, transformer,
                                    learning_rate = 0.001,
                                    feature_extract = do_feature_extract,
                                    batch_size = batch_size,
                                    num_epochs = 10,
                                    do_ts_images = False,
                                    use_multi_gpus=use_multi_gpus)
    
    training_losses.append(loss_history)
    #---------------------------- Saving after each training
    previous_model_path = f'./../data/models/model_{model_name+"_"+os.path.basename(path)}.pth'
    labels.append(os.path.basename(path))
    
    if use_multi_gpus and torch.cuda.device_count() > 1:
        torch.save(model.module.state_dict(),previous_model_path)
    else :
        torch.save(model.state_dict(),previous_model_path)

    print(f'It took :{time.time()-start:.2f} seconds to train. ')
    
    plt.plot(loss_history)
    plt.ylabel('Loss')
    plt.legend(labels)
    plt.title('Training Loss',fontsize=10)
    plt.savefig(f'{model_name+"_"+os.path.basename(path)}')

#--------Saving the losses for analysis
with open(f'./training_motionWords_losses_{model_name}','wb') as file:
    pickle.dump([training_losses,labels])
