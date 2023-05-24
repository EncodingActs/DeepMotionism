from deepArchitecture import train_model, initialize_model,transform_data
import torch
from dataPreparation import data_preparation
import time
import matplotlib.pyplot as plt
#import os
import pickle

start = time.time()
       
#--Creating an instance of data_preparation
data = data_preparation(mode='train',include_noisy_data=True)

#---- Setting parameters
do_ts_images = False
print(f'Transforming data into images : {do_ts_images}')
print('Include all noisy data : ', data.include_noisy)

data.set_imaging_parameters(do_recurrenceP = False, RP_threshold = 'point',
                               RP_timeDelay = 1, RP_dimension = 0.25,
                               do_GAF = False, GAF_method = 'summation',
                               do_MTF = False,
                               square_reshape = True,
                               vertical_stacking = False,
                               horizontal_stacking = False) 
 
data.set_motionWords_parameters(number_of_frames=16,overlaping_degree=0.75)
data.set_featureExpansion_parameters(do_PCA=True, number_of_components=50)

#--Mining csv and computing and storing motion words
data.csv_files_mining(select_high_variance_features=None,expand_features=True)
data.compute_motion_words(divide_sequence=0)

print('----Noisy data')
#-------------------------- Initialize ---------------------------------
model_name = 'resnet18'  # It can be 'resnet18' 'resnext50' or 'inceptionV3'.
do_feature_extract=False
print('Only training last FC layer :', do_feature_extract, "; Architecture : " , model_name)
model , input_size = initialize_model(model_name, num_classes=700,
                                      feature_extract= do_feature_extract, use_pretrained=True)

#-------------------------- Training -------------------------------

"""
The training scheme with no videos but noisy MoCap.
paths_to_training_dataset = ['./../data/datasets/training/training_dataset_5similarity_soft_dtw',
                             './../data/datasets/training/training_dataset_5similarity_gak',
                             './../data/datasets/training/training_dataset_10similarity_dtw',
                             './../data/datasets/training/noisy_no_videos_dataset_10similarity_dtw',
                             './../data/datasets/training/noisy_no_videos_dataset_5similarity_gak',
                             './../data/datasets/training/noisy_no_videos_dataset_5similarity_soft_dtw']
"""

#---Training scheme with noisy MoCap, videos of 12 keypoints, cleaned MoCap.
paths_to_training_dataset = ['./../data/datasets/training/training_dataset_10_Noisy_True_similarity_gak',
                             './../data/datasets/training/training_dataset_10_Noisy_True_similarity_soft_dtw',
                             './../data/datasets/training/training_dataset_10_Noisy_True_similarity_dtw']
    
print("\nWe're training on these merged files : ", paths_to_training_dataset)
#---Merging data
data.path_to_training_dataset ='./../data/training_datasets_noisy_True'
dataloader = data.load_training_dataset()
print('-'*20,"\n Merging training datasets \n")

transformer = transform_data(input_size, is_ts_images = do_ts_images)
batch_size = 2
use_mutli_gpus = True

model,loss_history = train_model(model, model_name, dataloader, data, transformer,
                                    learning_rate = 0.001,
                                    feature_extract = do_feature_extract,
                                    batch_size = batch_size,
                                    num_epochs = 30,
                                    do_ts_images = do_ts_images,
                                    use_multi_gpus=use_mutli_gpus)
    
    #---------------------------- Saving after each training
model_path = f'./../data/models/model_{model_name+"_MergedData_noisy_videos"}.pth'
    
if use_mutli_gpus and torch.cuda.device_count() > 1:
    torch.save(model.module.state_dict(),model_path)
else :
    torch.save(model.state_dict(),model_path)

print(f'It took :{time.time()-start:.2f} seconds to train. ')

plt.plot(loss_history)
plt.ylabel('Loss')
plt.title('Training Loss',fontsize=10)
plt.savefig(f'{model_name+"_MergedData_noisy_videos"}')

#--------Saving the losses for analysis
with open(f'./training_motionWords_losses_{model_name}_MergedData_noisy_videos','wb') as file:
    pickle.dump(loss_history)
    
