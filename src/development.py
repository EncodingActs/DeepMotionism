"""
Created on Tue Mar 16 14:43:27 2021

@author: fadel
"""
import cProfile,pstats
from dataPreparation import data_preparation
#from deepArchitecture import train_model, initialize_model,transform_data
#from memory_profiler import profile
import time
#import matplotlib.pyplot as plt
#import torch
#from deepArchitecture import initialize_model
import numpy as np

from nearpy import Engine
from nearpy.hashes import RandomBinaryProjections
from nearpy.filters import NearestFilter

def run():
    
    
    data = data_preparation(mode='train',include_noisy_data=True)
    data.set_motionWords_parameters(number_of_frames=16,overlaping_degree=0.75)
    data.set_featureExpansion_parameters(do_PCA=True,number_of_components=50)
    data.csv_files_mining(select_high_variance_features=None,expand_features=True)
    data.compute_motion_words(divide_sequence=0)
    
    #----imaging
    #---- Setting parameters
    data.set_imaging_parameters(do_recurrenceP = False, RP_threshold = 'point',
                               RP_timeDelay = 1, RP_dimension = 0.25,
                               do_GAF = False, GAF_method = 'summation',
                               do_MTF = False,
                               square_reshape = True,
                               vertical_stacking = False,
                               horizontal_stacking = False)
    data.load_motion_word_dataset('../data/embeddings_retrieval_Noisy_True')
        
    start = time.time()
    # Dimension of our vector space
    dimension = 700
    
    # Create a random binary hash with 10 bits
    rbp = RandomBinaryProjections('rbp', 10)
    
    # Create engine with pipeline configuration
    engine = Engine(dimension, lshashes=[rbp],vector_filters=[NearestFilter(10)])
    
    # Index 1000000 random vectors (set their data to a unique string)
    for key,val in data.motion_word_dataset.items():
        engine.store_vector(val.reshape((1,-1)).squeeze(), key)
    
    # Create random query vector
    
    for  query in data.motion_word_dataset.values():
        # Get nearest neighbours
        N = engine.neighbours(query)
        names = [data[1] for data in N]
        print(names)
    
    print(time.time() - start)
    """
    data = data_preparation(mode='test',include_noisy_data=True)
    data.set_motionWords_parameters(number_of_frames=16,overlaping_degree=0.75)
    data.set_featureExpansion_parameters(do_PCA=True,number_of_components=50)
    data.csv_files_mining(select_high_variance_features=None,expand_features=True)
    data.compute_motion_words(divide_sequence=0)
    
    #----imaging
    #---- Setting parameters
    data.set_imaging_parameters(do_recurrenceP = False, RP_threshold = 'point',
                               RP_timeDelay = 1, RP_dimension = 0.25,
                               do_GAF = False, GAF_method = 'summation',
                               do_MTF = False,
                               square_reshape = True,
                               vertical_stacking = False,
                               horizontal_stacking = False)
    
    for k in [10]:
        for i in ['dtw','gak','soft_dtw']:
            func = data.similarity_functions[i]
            data.compute_training_dataset(func,k)
    
    
    paths = ['./../data/datasets/training/training_dataset_10_Noisy_True_similarity_gak',
             './../data/datasets/training/training_dataset_10_Noisy_True_similarity_soft_dtw',
             './../data/datasets/training/training_dataset_10_Noisy_True_similarity_dtw']
    
    data.merge_training_data(paths)
    """
    
    
    
"""
    # mode can be 'train' or 'test'
    data = data_preparation(mode = 'train')
    data.set_motionWords_parameters(number_of_frames=16,overlaping_degree=0.75)
    data.set_featureExpansion_parameters(do_PCA=True,number_of_components=50)
    data.csv_files_mining(select_high_variance_features=True,expand_features=True)
    data.compute_motion_words(divide_sequence=0)
    
    paths_to_positive_negative_samples = ['./../data/datasets/training/training_dataset_10similarity_dtw',
                                         './../data/datasets/training/training_dataset_10similarity_gak',
                                         './../data/datasets/training/training_dataset_10similarity_soft_dtw']
    dataset = data.merge_training_data(paths_to_positive_negative_samples)
    
    positive = []
    negative = []
    dict_vals =  data.motion_word_dataset
    length = len(dict_vals)
    count = 0
    for name,value in dataset.items():
        func = data.similarity_functions[0] #gak:0, softDTW:1 ,dtw:2
        dummy_pos,dummy_neg = [],[]
        #---
        for i in range(len(value[0])):
            dummy_pos.append(func(dict_vals[value[0][i]],
                                  dict_vals[name]))
            
            dummy_neg.append(func(dict_vals[value[0][1]],
                                  dict_vals[name]))
        #---
        positive.append(np.mean(np.array(dummy_pos)))
        negative.append(np.mean(np.array(dummy_neg)))
        
        if count % 50 == 0 : 
            print("Progress: %s" % str(100*count/length))
        count += 1
    
    plt.boxplot(positive)
    plt.boxplot(negative)
    plt.ylabel("Sigma")
    plt.title("Global alignment kernel sigmas")
"""
"""
#-----------------------------------------------
data = data_preparation(mode = 'train')
data_embeddings = data_preparation(mode='train')
#------------------------------------------------
data.set_motionWords_parameters(number_of_frames=16,overlaping_degree=0.75)
data.set_featureExpansion_parameters(do_PCA=True,number_of_components=50)
data.csv_files_mining(select_high_variance_features=True,expand_features=True)
data.compute_motion_words(divide_sequence=0)

#--Loading the model parameters
num_classes = 700
model , input_size = initialize_model(model_name='resnet18', num_classes = num_classes,
                                          feature_extract= False, use_pretrained=False)
model_path = './../data/models/model_resnet18_training_dataset_10similarity_dtw.pth' 
model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')),strict=True)
print(f"Model : {model_path} --> loaded.")
    
#--Loading the transformer
transformer = transform_data(input_size,is_ts_images = False)

#--Running the evaluation
evaluation(model, 
           data,
           transformer,
           embedding_dimension = num_classes,
           batch_size = 100, 
           do_ts_images = False,
           is_inception=False)

#--- 
data_embeddings.load_motion_word_dataset(path='./../data/embeddings')
names,values_emd, values_sim = embedding_quality_assessment_similarityFuncVsNN(data, 
                                                                               data_embeddings, 
                                                                               k = 10, 
                                                                               similarityFunc = 'dtw')

"""

def profiling():
    profiler = cProfile.Profile()
    profiler.enable()
    run()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats()


#profiling()

run()
