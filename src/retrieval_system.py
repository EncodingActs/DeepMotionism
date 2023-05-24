# -*- coding: utf-8 -*-

from evaluation import  evaluation, motion_signatures, cluster_dataset
from deepArchitecture import transform_data,initialize_model
from dataPreparation import data_preparation
import torch
import numpy as np
from joblib import dump, load
import os
import copy
from sklearn.neighbors import BallTree 
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from re import split
from nearpy import Engine
from nearpy.hashes import RandomBinaryProjectionTree
from nearpy.filters import NearestFilter
from nearpy.distances import CosineDistance

class retrieval_engine():
    """
    
    
    Methods
    -------
    
    
    
    
    """
    
    def __init__(self, 
                 model_name='resnet18',
                 path_to_model_param='./../data/models/model_resnet18_RecordingParam_tsImages_False.pth', 
                 embedding_dimension=700, 
                 do_ts_images = False,
                 is_inception=False,
                 num_subsequences=4,
                 select_high_variance_features = True,
                 include_noisy=False ):
        """
        
        Parameters
        ----------
        model_name : string, optional
            It is the name of the desired backbone. The default is 'resnet18'.
            
        path_to_model_param : string, optional
            It is the path to model parameters meaning the weights and biases of the network. The default is './../data/models/model_resnet18_RecordingParam_tsImages_False.pth'.
            
        embedding_dimension : int, optional
            The embedding dimension of the motion words.Give the embedding dimension for which the model was trained. The default is 700.
            
        do_ts_images : boolean, optional
            It states if the motion words should be transformed into images.If set to 'True' one should run "self.data.set_imaging_parameters(**)" . The default is False.
            
        is_inception : boolean, optional
            It states if the model is an inception model. The default is False.
            
        num_subsequences : int, optional
            The default is 4.

        Returns
        -------
        None.
        

        """
        
        
        self.data = None
        self.select_high_variance_features = select_high_variance_features
        self.include_noisy = include_noisy
        
        if self.include_noisy :
            path_to_model_param = './../data/models/model_resnet18_RecordingParam_tsImages_False_includeNoisy_True.pth'
            self.select_high_variance_features = None
            
        for indx, mode in enumerate(['train','test']):
            
            dummy = data_preparation(mode = mode,include_noisy_data=include_noisy)
            
            #-----------------------Parameters-------------------------
            dummy.set_motionWords_parameters(number_of_frames=32,overlaping_degree=0.75)
            dummy.set_featureExpansion_parameters(do_PCA=True, number_of_components=50)
            dummy.csv_files_mining(select_high_variance_features=self.select_high_variance_features,
                                   expand_features=True)
            
            dummy.compute_motion_words(divide_sequence=num_subsequences)
            
            #----imaging parameters
            dummy.set_imaging_parameters(do_recurrenceP = False, RP_threshold = 'point',
                                       RP_timeDelay = 1, RP_dimension = 0.25,
                                       do_GAF = False, GAF_method = 'summation',
                                       do_MTF = True,#False
                                       square_reshape = True,
                                       vertical_stacking = False,
                                       horizontal_stacking = False)
            
            if indx == 0 :
                self.data = copy.deepcopy(dummy)
                
            else :
                self.data.motion_word_dataset.update(dummy.motion_word_dataset)
                self.data.mapping_Of_motionWordsNames.update(dummy.mapping_Of_motionWordsNames)
                
        #---   
        self.do_ts_images = do_ts_images
        self.model_name = model_name
        self.path_to_model_param = path_to_model_param
        
        self.is_inception = is_inception
        self.batch_size = 180
        self.embedding_dimension = embedding_dimension
        self.num_clusters = 200
        
        #---Signatures
        self.signatures = None
        
        #---Predict clusters
        self.cluster_predictor = None
        self.path_to_predictor = f"./../data/cluster_predictor_Noisy_{self.include_noisy}.joblib"
        self.cluster_assignments = None
        self.cluster_sizes = None
        self.cluster_centers = None

        #---Flag
        self.data_contains_embeddings = False
        
        #---Windows parameters
        self.windows_overlaping_degree = 0.6
        self.ball_tree_window_size = 40 # number of words 20
        
        self.lsh_engine = None
        
    def compute_embeddings(self) :
        """Computes the embeddings.
        

        Returns
        -------
        None.

        """
        model,input_size =  initialize_model(model_name = self.model_name,
                                       num_classes=self.embedding_dimension,
                                       feature_extract=False,use_pretrained=False)
        
        #model.load_state_dict(torch.load(self.path_to_model_param,
                                                      #map_location=torch.device('cpu')),
                                                      #strict=True)
        #--Loading last checkpoint and model parameters                                           
        checkpoint = torch.load(self.path_to_model_param, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)  
        
        print("\nThere are : ",len(self.data.motion_word_dataset), "motions words in the dataset.")
        print(f"\nModel : {self.path_to_model_param} --> loaded.")
        
        self.transformer = transform_data(input_size,is_ts_images=self.do_ts_images)
        
        #--Computing embeddings
        evaluation(model,
                   self.data,
                   self.transformer, 
                   self.embedding_dimension,
                   self.batch_size,
                   self.do_ts_images,
                   self.is_inception,
                   saving_file_ending=f'retrieval_Noisy_{self.include_noisy}')
        
        #---Loading embeddings        
        self.data.load_motion_word_dataset(path=f'./../data/embeddings_retrieval_Noisy_{self.include_noisy}')
        
        #--Setting the flag to True
        self.data_contains_embeddings = True
        
    def compute_cluster_predictor(self,save=True):
        """Computes the clusters and assignments.
        

        Parameters
        ----------
        save : boolean, optional
            It states if the computed clusters. The default is True.

        Returns
        -------
        None.

        """
        if self.data_contains_embeddings == False : 
            self.compute_embeddings()
        
        embeddings = np.stack(list(self.data.motion_word_dataset.values()),axis=0)
        self.cluster_assignments , self.cluster_sizes , self.cluster_centers, self.cluster_predictor  = cluster_dataset(embeddings ,
                                                                                                                        motion_names =  list(self.data.motion_word_dataset.keys()), 
                                                                                                                        num_clusters = self.num_clusters)   
        if save :
            #--Saving fitted model
            dump(self.cluster_predictor, self.path_to_predictor) 
            dump(self.cluster_assignments,f"./../data/cluster_assignments_Noisy_{self.include_noisy}.joblib")
            dump(self.cluster_sizes,f"../data/cluster_sizes_Noisy_{self.include_noisy}.joblib")
            dump(self.cluster_centers,f"../data/cluster_center_Noisy_{self.include_noisy}.joblib")
        
                
    def load_predictor(self):
        """Loads clusters information.
        

        Returns
        -------
        None.

        """
        if os.path.exists(self.path_to_predictor):
            self.cluster_predictor = load(self.path_to_predictor)
        
            self.cluster_assignments =  load(f"./../data/cluster_assignments_Noisy_{self.include_noisy}.joblib")
            self.cluster_sizes = load(f"../data/cluster_sizes_Noisy_{self.include_noisy}.joblib")
            self.cluster_centers = load(f"../data/cluster_center_Noisy_{self.include_noisy}.joblib")
            
        else : print("Make sure the predictor has been saved to : ", self.path_to_predictor)
        
        return None
    
    def load_embedding(self):
        """Loads embeddings.
        

        Returns
        -------
        None.

        """
        path = f'./../data/embeddings_retrieval_Noisy_{self.include_noisy}'
            
        #---Loading embeddings        
        self.data.load_motion_word_dataset(path)
        
        #--Setting the flag to True
        self.data_contains_embeddings = True
        
        return None
        
    def compute_motion_signatures(self,query_sequence_name=None,sequence_decomposition = None):
        """Computes motion signatures.
        

        Parameters
        ----------
        query_sequence_name : string, optional
            The default is None.
            
        sequence_decomposition : dictionary, optional
            It is a dictionary where the values. The default is None.

        Returns
        -------
        signatures : dictionary

        """
        
        #---Signatures
        signatures = motion_signatures(self.data,
                                                  num_clusters=self.num_clusters,
                                                  plot_embeddings=False,
                                                  plot_signatures=False,
                                                  plot_pairwise_distances=False,
                                                  names_and_assignments=self.cluster_assignments,
                                                  size_of_clusters=self.cluster_sizes,
                                                  cluster_centers = self.cluster_centers,
                                                  sequence_decomposition = sequence_decomposition  )
        
        return signatures
    
    
    def compute_ball_tree(self,save=True,distance_metric='pyfunc', window_size=None):
        """Computes the ball tree and optionally save it.
        

        Parameters
        ----------
        save : boolean, optional
            The default is True.
            
        distance_metric : string, optional
            We recommend the following distance metrics are 'euclidean', 'minkowski','manhattan', 'cityblock', 'l1', 'seuclidean', 'canberra', 'pyfunc'. The default is 'pyfunc'. When we use 'pyfunc', we can define our own function which is here the earth mover distance..
            
        window_size : int, optional
            The number of motion words to consider as a window. The default is None.
              
        Returns
        -------
        list : [tree, signatures, all_sequences_names, windows_maping, all_windows]
            For more info on how to perform queries on the tree visit :  https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html.

        """
        
        if self.data_contains_embeddings == False:
            self.compute_embeddings()
        
        if window_size is not None:
            self.ball_tree_window_size = window_size
        
        #--
        all_windows, windows_maping = self.retrieve_windows(query_name=None,window_size = self.ball_tree_window_size)
        
        #--Compute clusters if necessary
        condition = [val is None for val in [self.cluster_assignments,self.cluster_centers,self.cluster_sizes,self.cluster_predictor]]
        if any(condition) :
            print('\nComputing clusters...')
            self.compute_cluster_predictor(save=False)
        #--  
        print('Computing motion signatures...')
        all_sequences_signatures = self.compute_motion_signatures(sequence_decomposition =  all_windows)
        
        #--Earth mover distance for distributions
        signatures = np.vstack(list(all_sequences_signatures.values()))
        tree = BallTree(signatures, leaf_size = 40, metric = distance_metric,func=wasserstein_distance)
         
        #--           
        if save :
            file = f'./../data/ball_tree_signatures_Noisy_{self.include_noisy}.joblib' 
            dump([tree,signatures,list(all_sequences_signatures.keys()),windows_maping,all_windows], file)
           
        return [tree, signatures, list(all_sequences_signatures.keys()), windows_maping, all_windows]
    
    def compute_dendrogram(self):
        """Computes the dendrogram of the embeddings.
        

        Returns
        -------
        None.

        """
        if self.data_contains_embeddings:
            embeddings = np.vstack(list(self.data.motion_word_dataset.values()))
        else :
            self.compute_embeddings()
            embeddings = np.vstack(list(self.data.motion_word_dataset.values()))
        
        X = pdist(embeddings,metric='euclidean')
        Z = linkage(X, 'ward')
        _ = plt.figure(figsize=(70, 70))
        _ = dendrogram(Z)
        plt.show()
                
        print(Z)

    def compute_lsh_engine(self,top_k=10):
        """Computes the lsh engine.
        

        Parameters
        ----------
        top_k : int, optional
            The default is 10.

        Returns
        -------
        None.

        """
        all_windows, windows_maping = self.retrieve_windows(query_name=None,window_size = self.ball_tree_window_size)
        all_sequences_signatures = self.compute_motion_signatures(sequence_decomposition =  all_windows)
                
        rbpt = RandomBinaryProjectionTree('rbpt', 2*top_k, 2*top_k)
        engine = Engine(self.num_clusters, lshashes=[rbpt],distance=CosineDistance(),vector_filters=[NearestFilter(top_k)])
        
        for key,val in all_sequences_signatures.items():
            engine.store_vector(val.reshape((1,-1)).squeeze(), key)

        self.lsh_engine = [engine,all_sequences_signatures,all_windows]
        
        return None
   #---------------------------------------- Retrieving data--------------------
    def retrieve_windows(self,query_name=None,window_size=None):
        """This function retrieves the windows. Make sure that either query_name and window_size is defined. Both shouldn't be 'None'.
        

        Parameters
        ----------
        query_name : string or None, optional
            The default is None.
            
        window_size : int or None, optional
            The default is None.

        Returns
        -------
        all_windows : dictionary
            
        windows_mapping : dictionary

        """
        
        all_windows = dict()
        windows_mapping = dict()
        
                
        if query_name is not None  :
            window_size = len(self.data.mapping_Of_motionWordsNames[query_name]) 
        
        #--Number of frames to skip
        skip = int(self.windows_overlaping_degree * window_size)
        
        for sequence_name, motions_in_sequence in self.data.mapping_Of_motionWordsNames.items() :
            
            #--length of the array
            n = len(motions_in_sequence)
            #---------
            if n > window_size : 
                windows_names = []
                for i in range(int((n - window_size)/skip)):
                    window = motions_in_sequence[i*skip:i*skip + window_size]
                    #windows.append(window)
                    #---
                    name = sequence_name + f"_window_{i}"
                    all_windows[name] = window
                    windows_names.append(name)
                
                windows_mapping[sequence_name] = windows_names
            #else :
            #---------
            all_windows[sequence_name] = motions_in_sequence
            windows_mapping[sequence_name] = [sequence_name]
            
        return all_windows, windows_mapping
        
    
    def retrieve_reference(self, query_name=None,sequence_name=None,start_frame_index=None,end_frame_index=None,top_k = 10, load_emdeddings = True, load_clusters = True, windows_overlaping_degree = None):
        """Reference retrieval engine.
        

        Parameters
        ----------
        query_name : string
            Name of the sequence to retrieve. The default is None.
            
        sequence_name : string or None, optional
            This parameter should be given if one wants to use "start_frame_index" & "end_frame_index" . The default is None.
            
        start_frame_index : int, optional
            Starting frame index. The default is None.
            
        end_frame_index : int, optional
            Ending frame index. The default is None.
            
        top_k : int, optional
            Top closest sequences to retrieve. The default is 10.
            
        load_emdeddings : boolean, optional
            The default is True.
            
        load_clusters : boolean, optional
            The default is True.
            
        windows_overlaping_degree : float or None, optional
            The window overlaping degree. It should be between 0 and 1. The default is None.

        Returns
        -------
        Dictionary
            The keys of this dictionary are the retrieved sequences or subsequences names. And the values are lists of the following form [parent_sequence_name,(starting frame indice, ending frame indice)]. "parent_sequence_name" is the sequence name for which the frame indices are given.

        """

        
        #--
        if sequence_name is not None :
            assert None not in [start_frame_index,end_frame_index],"Please provide valid values for 'start_frame_index' & 'end_frame_index.' "
            query_name = self.data.add_subsequence(sequence_name, start_frame_index, end_frame_index)
        
        assert query_name is not None, "query_name can't be 'None'.Give valid query_name."
        
        if load_emdeddings:
            self.load_embedding()
            
        if self.data_contains_embeddings == False:
            self.compute_embeddings()
        
        if load_clusters :
            self.load_predictor()
        
        if windows_overlaping_degree is not None :
            self.windows_overlaping_degree = windows_overlaping_degree
            
        #--Compute clusters if necessary
        condition = [val is None for val in [self.cluster_assignments,self.cluster_centers,self.cluster_sizes,self.cluster_predictor]]
        if any(condition) :
            print('\nComputing clusters...')
            self.compute_cluster_predictor(save=True)
        
        #--Retrieving all windows
        print('\nRetrieving windows...')
        all_windows, windows_maping = self.retrieve_windows(query_name)
        print(f'There are {len(all_windows)} windows to analyze.')
        
        #--Inverse document frequency function
        idf = lambda x : np.log( len(all_windows)/(1+self.cluster_sizes[x]) )
        cluster_densities = [1/np.exp(idf(x)) for x in range(self.num_clusters)]
        
        #--Earth mover distance for distributions
        func = lambda x,y: wasserstein_distance(x,y, u_weights = cluster_densities,v_weights=cluster_densities)
        distances = []
        print('Computing motion signatures...')
        all_windows_signatures = self.compute_motion_signatures(sequence_decomposition = all_windows)
                
        #----
        for _, sequence_windows in windows_maping.items():
            distance = [func(all_windows_signatures[window],all_windows_signatures[query_name]) for window in  sequence_windows]
            distances.append(np.mean(np.array(distance)))
            
        #----
        sequences_names = list(windows_maping.keys())
        sequences_names = filter(lambda x: not x.endswith('_rot'), sequences_names)#filter out whole sequence
        sorted_names_by_score = [[name,score] for score,name in sorted(zip(distances,sequences_names))]
        top_k = min(top_k,len(distances))
        selected_names  = [sorted_names_by_score[i][0] for i in range(top_k)]
        
        #----
        func = lambda x : split(r"\B_window_",x)[0]
        func_1 = lambda x : split(r"_sub_\B",x)[0]
        parentNames_framesIndices = []
        for name in selected_names :
            #--Getting sequence name and updating dictionary.
            if name.count("_sub_") > 0 :
                parentNames_framesIndices.append([func_1(name),self.data.get_frames_indices(all_windows[name])]) #get_frames_indices_second
            else :
                parentNames_framesIndices.append([func(name),self.data.get_frames_indices(all_windows[name])])
                    
        return dict(zip(selected_names,parentNames_framesIndices))
        
    def retrieve_tree(self,query_name=None,sequence_name=None,start_frame_index=None,end_frame_index=None,top_k=10,load_tree=True ):
        """Tree-based retrieval engine.
        

        Parameters
        ----------
        query_name : string
            Name of the sequence to retrieve. The default is None.
            
        sequence_name : string or None, optional
            This parameter should be given if one wants to use "start_frame_index" & "end_frame_index" . The default is None.
            
        start_frame_index : int, optional
            Starting frame index. The default is None.
            
        end_frame_index : int, optional
            Ending frame index. The default is None.
            
        top_k : int, optional
            Top closest sequences to retrieve. The default is 10.
            
        load_tree : boolean, optional
            The default is True.

        Returns
        -------
        Dictionary
            The keys of this dictionary are the retrieved sequences or subsequences names. And the values are lists of the following form [parent_sequence_name,(starting frame indice, ending frame indice)]. "parent_sequence_name" is the sequence name for which the frame indices are given.

        """
        
        tree_path=f'./../data/ball_tree_signatures_Noisy_{self.include_noisy}.joblib'
        
        #--Retrieve subsequence if frames indexes are given.
        if sequence_name is not None :
            assert None not in [start_frame_index,end_frame_index],"Please provide valid values for 'start_frame_index' & 'end_frame_index.' "
            query_name = self.data.add_subsequence(sequence_name, start_frame_index, end_frame_index)
            load_tree = False      
        assert query_name is not None, "query_name can't be 'None'.Give valid query_name."
        
        #--Loading or computing data
        if load_tree and os.path.exists(tree_path):
            tree,all_windows_signatures,all_windows_names, windows_maping,all_windows = load(tree_path)
        else:
            print('Computing ball tree....')
            tree,all_windows_signatures,all_windows_names, windows_maping,all_windows = self.compute_ball_tree()

        
        #---Retrieval
        all_indices = []
        for window in windows_maping[query_name]:
            window_index = all_windows_names.index(window)
            window_signature = all_windows_signatures[window_index].reshape(1,-1)
            dist, inds = tree.query(window_signature, k=top_k)
            all_indices.append(inds)
        
        all_indices= np.vstack(all_indices)
        unique, counts = np.unique(all_indices, return_counts=True)
        out = [ind for _,ind in sorted(zip(counts,unique),reverse=True)][:top_k]
        
        retrieved_sequences = dict()
        func = lambda x : split(r"\B_window_",x)[0]
        func_1 = lambda x : split(r"_sub_\B",x)[0]
        
        for ind in out : 
            selected = all_windows_names[ind]
            
            #--Getting sequence name and updating dictionary.
            if selected.count("_sub_") > 0 :
                retrieved_sequences[selected] = [func_1(selected),self.data.get_frames_indices(all_windows[selected])] #get_frames_indices_second
            else :
                retrieved_sequences[selected] =[func(selected),self.data.get_frames_indices(all_windows[selected])]
            
            
        return retrieved_sequences
    
    def retrieval_lsh(self,query_name=None,sequence_name=None,start_frame_index=None,end_frame_index=None,top_k=10):
        """Lsh-based retrieval engine.
        

        Parameters
        ----------
        query_name : string or None, optional
            The default is None.
            
        sequence_name : string or None, optional
            This parameter should be given if one wants to use "start_frame_index" & "end_frame_index" . The default is None.
            
        start_frame_index : int, optional
            Starting frame index. The default is None.
            
        end_frame_index : int, optional
            Ending frame index. The default is None.
            
        top_k : int, optional
            Top closest sequences to retrieve. The default is 10.

        Returns
        -------
        Dictionary
            The keys of this dictionary are the retrieved sequences or subsequences names. And the values are lists of the following form [parent_sequence_name,(starting frame indice, ending frame indice)]. "parent_sequence_name" is the sequence name for which the frame indices are given.One should divide by the frame rate 'fps' to get location in 'seconds' in the parent sequence.

        """
  
        #--
        if sequence_name is not None:
            assert None not in [start_frame_index,end_frame_index],"Please provide valid values for 'start_frame_index' & 'end_frame_index.' "
            query_name = self.data.add_subsequence(sequence_name, start_frame_index, end_frame_index)
        
        #assert query_name is not None, "query_name can't be 'None'.Give valid query_name."
        assert query_name is not None, "query_name can't be 'None'.Give valid query_name."

        if self.lsh_engine is None :
            self.compute_lsh_engine(top_k=top_k)
        
        engine,signatures_dataset,all_windows = self.lsh_engine
        query = signatures_dataset[query_name]
        results = engine.neighbours(query)
        names = [data[1] for data in results]
        
        #--Thses functions will get the parent sequence of a given subsequence
        func = lambda x : split(r"\B_window_",x)[0]
        func_1 = lambda x : split(r"_sub_\B",x)[0]
        parentNames_framesIndices = []
        
        for name in names :
            #--Getting sequence name and updating dictionary.
            if name.count("_sub_") > 0 :
                parentNames_framesIndices.append([func_1(name),self.data.get_frames_indices(all_windows[name])])
            else :
                parentNames_framesIndices.append([func(name),self.data.get_frames_indices(all_windows[name])])

        return dict(zip(names,parentNames_framesIndices))
    
#----------------------------------------- Demo --------------------------------
"""Example
engine = retrieval_engine(num_subsequences=0,select_high_variance_features=None,include_noisy=True)
engine.load_embedding() 
engine.load_predictor()
engine.ball_tree_window_size = 25
engine.lsh_engine = None
for i in range(10) :
    name = list(engine.data.mapping_Of_motionWordsNames.keys())[i]
    retrieved_info_1 = engine.retrieve_tree(query_name=None,
                                            sequence_name=name,
                                            start_frame_index=1,
                                            end_frame_index=67,
                                            top_k = 10, 
                                            load_tree=True)

    print(retrieved_info_1.values(),'\n\n')
 """   

if "main" == "0main" :
    """This code runs queries and compare the 3 differents retrieval techniques in terms of speed,
    and relative similarity with respect to the reference method.
    
    
    PS :One should either load embeddings or compute new embeddings. If new data is data then, one should compute new embeddings.
    
    """
    import time
    init = time.time()
    
    engine = retrieval_engine(num_subsequences=0,select_high_variance_features=None,include_noisy=True)
    
    #-- Either load or compute embeddings.
    #If new csv files are added then, compute new embeddings
    #engine.load_embedding()
    engine.compute_embeddings()
    
    #--If one computes the predictor then it should compute cluster predictor
    engine.compute_cluster_predictor(save=True) 
    #engine.load_predictor()
    
    #--Other parameters.
    engine.ball_tree_window_size = 20
    engine.lsh_engine = None
    
    query_names = list(engine.data.mapping_Of_motionWordsNames.keys())
    print('Number of sequences : ', len(query_names))
    distances = []
    distances_2 = []
    distances_3 = []
    times = []
    times_1 = []
    times_2 = []
    top_k= 10
    
    for name in query_names[:-1] :
        start = time.time()
        print('-'*50)
        retrieved_info_1 = engine.retrieve_tree(query_name=name,sequence_name=None,start_frame_index=None,end_frame_index=None,top_k = top_k, load_tree=True)
        names_tree = list(retrieved_info_1.keys())
        print("\nQuery name : ",name,'\n')
        times_1.append(time.time() - start)
        
        retrieved_info_2 = engine.retrieval_lsh(query_name=name,sequence_name=None,start_frame_index=None,end_frame_index=None,top_k=top_k)
        names_lsh = list(retrieved_info_2.keys())
        times_2.append(time.time()- times_1[-1] - start)
        
        retrieved_info= engine.retrieve_reference(query_name=name,sequence_name=None,start_frame_index=None,end_frame_index=None, top_k = top_k,load_emdeddings=True,load_clusters=True)
        names_reference = list(retrieved_info.keys())
        #print(name,'\n\n',names_,'\n\n',names_1,'\n\n',names)

        times.append(time.time() - start - times_2[-1] - times_1[-1])

        similarity_between_methods_2 = 100 - 100*len(set(names_reference + names_lsh))/(2*top_k)
        similarity_between_methods = 100 - 100*len(set(names_reference + names_tree))/(2*top_k) 
        similarity_between_methods_3 = 100 - 100*len(set(names_tree + names_lsh))/(2*top_k)
        
        distances.append(similarity_between_methods)
        distances_2.append(similarity_between_methods_2)
        distances_3.append(similarity_between_methods_3)
    print(f'\n Total time : {time.time()-init:0.2f} seconds.')
    
    #--Plotting
    fig, ax = plt.subplots(4,1,sharex=True,figsize=(15,15))
    ax[0].plot(distances,label=f'Tree vs Reference, median = {np.median(np.array(distances)):0.2f}')
    ax[0].plot(distances_2,label=f'LSH vs Reference, median = {np.median(np.array(distances_2)):0.2f}')
    ax[0].plot(distances_3,label=f'LSH vs Tree, median = {np.median(np.array(distances_3)):0.2f}')
    ax[0].set_ylabel('%')
    ax[0].legend()
    
    ax[1].plot(times,'-g',label=f'Median : {np.median(np.array(times)):0.6f}')
    ax[1].set_ylabel('Time in seconds')
    ax[1].legend()
    
    ax[2].plot(times_1,'-r',label=f'Median : {np.median(np.array(times_1)):0.6f}')
    ax[2].set_xlabel("Sequences")
    ax[2].set_ylabel('Time in seconds')
    ax[2].legend()
    
    ax[3].plot(times_2,'-m',label=f'Median : {np.median(np.array(times_2)):0.6f}')
    ax[3].set_ylabel("Time in seconds")
    ax[3].legend()

    #--
    ax[0].set_title('Methods similarity in %')
    ax[1].set_title(f'Time for sliding window technique . Top: {top_k}.')
    ax[2].set_title(f'Time for ball tree technique, window size :{engine.ball_tree_window_size}')
    ax[3].set_title(f'Time for lsh technique, window size :{engine.ball_tree_window_size}')
    
    
    #print(distances)
    