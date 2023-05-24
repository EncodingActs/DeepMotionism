import time
from dataPreparation import data_preparation
from deepArchitecture import initialize_model,transform_data
import torch
from evaluation import evaluation, motion_signatures,embedding_quality_assessment_OneVsRest,embedding_quality_assessment_similarityFuncVsNN
from retrieval_system import retrieval_engine
from  copy import deepcopy
import os

start = time.time()

engine = retrieval_engine(num_subsequences=4,select_high_variance_features=None,include_noisy=True)

engine.compute_embeddings()
engine.compute_cluster_predictor(save=True) 
#engine.load_embedding()
#engine.load_predictor()    

top_k = 20
query_names = list(engine.data.mapping_Of_motionWordsNames.keys())[:8]

for name in query_names :
    print('-'*50)
    #retrieved_sequences= engine.retrieve_reference(query_name=name,sequence_name=None,start_frame_index=None,end_frame_index=None, top_k = top_k,load_emdeddings=True,load_clusters=True)

    #--Other methods below : uncomment to use !
    #retrieved_sequences = engine.retrieve_tree(query_name=name,sequence_name=None,start_frame_index=None,end_frame_index=None,top_k = top_k, load_tree=False)
    retrieved_sequences = engine.retrieval_lsh(query_name=name,sequence_name=None,start_frame_index=None,end_frame_index=None,top_k=top_k) #tart_frame_index=1080,end_frame_index=2080,
    
    print("\nQuery name : ",name)
    for key,value in retrieved_sequences.items():
        print('\nSubsequence or sequences retrieved :',key,
              '\nParent sequence name and frames info :',value,'\n')
   

print(f"It took : {time.time() - start : 0.2f} seconds.")
        

#-------------------------  Demo Evaluation --------------------------------------------
if "__main__" == "0__main__" :
    
    start = time.time()
    
    #--Creating an object of data_preparation
    embedding_dimension = 700
    data = data_preparation(mode='train',include_noisy_data=False)
    data.set_motionWords_parameters(number_of_frames=32,overlaping_degree=0.25) #ym: change number_of_frames=16->32 , overlaping_degree=0.75->0.25
    data.set_featureExpansion_parameters(do_PCA=True,number_of_components=50)
    
    motion_word_dataset = deepcopy(data.motion_word_dataset)
    #--Mining csv and computing and storing motion words 
    #Uncomment  1) or 2)
    
    #-------------1)
    data.csv_files_mining(select_high_variance_features=True,expand_features=True)
    data.compute_motion_words(divide_sequence=7)
    #------------
    
    #-------------2)
    #data.load_motion_word_dataset()
    #data.load_motion_words_context()
    #-------------
    
    #--Set imaging parameters
    data.set_imaging_parameters(do_recurrenceP=False,do_GAF=False,do_MTF=False)
      
    #--Loading the model parameters
    model , input_size = initialize_model(model_name='resnet18', num_classes = embedding_dimension,
                                          feature_extract= False, use_pretrained=False)
    
    model_path = './../data/models/model_resnet18_training_dataset_10similarity_dtw.pth' 
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')),strict=True)
    print(f"Model : {model_path} --> loaded.")
    
    #--Loading the transformer
    transformer = transform_data(input_size,is_ts_images = False)
    
    #--Loading dataloader
    #dataloader = data.motion_word_dataset
    
    #--Running the evaluation
    evaluation(model, 
               data,
               transformer,
               embedding_dimension = embedding_dimension,
               batch_size = 100, 
               do_ts_images = False,
               is_inception=False)
    
    data.load_motion_word_dataset(path='./../data/embeddings')
    
    signatures_dict = motion_signatures(data,
                                        plot_embeddings=True,
                                        num_clusters=100,
                                        plot_signatures=True,
                                        plot_pairwise_distances=True)
    
    embedding_quality_assessment_OneVsRest(signatures_dict)
    sim_func = ['dtw','soft_dtw','gak'] 
    print(f'\n The similarity function used is : {sim_func}')
    plot_title =f"{os.path.splitext(os.path.basename(model_path))[0]}"
    values_emd, values_sim = embedding_quality_assessment_similarityFuncVsNN(motion_word_dataset, 
                                                                             data, 
                                                                             similarityFunc = sim_func,
                                                                             plot_title=plot_title, 
                                                                             mode=data.mode)
            

    print(f'\n Running time: {time.time()-start:.3f} seconds')

#-------------------------------Demo compute training dataset------------------------------------
if "main" == 0 : #"main":
    
    data = data_preparation()
    
    data.set_motionWords_parameters(number_of_frames=16,overlaping_degree=0.75)
    data.set_featureExpansion_parameters(do_PCA=True,number_of_components=50)
    data.csv_files_mining(select_high_variance_features=True,expand_features=True)
    
    data.compute_motion_words(divide_sequence=0)
    
    for k in [5,10]:
        func = data.similarity_functions['soft_dtw'] # it can be : 'soft_dtw';'dtw';'gak'
        data.compute_training_dataset(func,k)
        
#--------------------------------Demo Retrieval -------------------------------------------------
if "main" == "0main" :
    
    start = time.time()
    
    engine = retrieval_engine(num_subsequences=10,select_high_variance_features=None,include_noisy=True)
    engine.load_embedding()
    engine.load_predictor()    
    
    top_k = 10
    query_names = list(engine.data.mapping_Of_motionWordsNames.keys())[:4]
    
    for name in query_names :
        print('-'*50)
        retrieved_sequences= engine.retrieve_reference(query_name=name,sequence_name=None,start_frame_index=None,end_frame_index=None, top_k = top_k,load_emdeddings=True,load_clusters=True)
        
        #--Other methods below : uncomment to use !
        #retrieved_sequences = engine.retrieve_tree(query_name=name,sequence_name=None,start_frame_index=None,end_frame_index=None,top_k = top_k, load_tree=True)
        #retrieved_sequences = engine.retrieval_lsh(query_name=name,sequence_name=None,start_frame_index=None,end_frame_index=None,top_k=top_k)
        
        print("\nQuery name : ",name)
        for key,value in retrieved_sequences.items():
            print('\nSubsequence or sequences retrieved :',key,
                  '\nParent sequence name and frames info :',value,'\n')
       
    
    print(f"It took : {time.time() - start : 0.2f} seconds.")
        


