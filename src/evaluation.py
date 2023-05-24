from deepArchitecture import initialize_model, transform_data, build_batch
import torch
from dataPreparation import data_preparation
import matplotlib.pyplot as plt
from sklearn import cluster, metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate,RepeatedKFold
import numpy as np
import pickle
from pandas import read_csv
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance
import os
import time
import copy


def evaluation(model, data_object,
               transformer, 
               embedding_dimension , batch_size = 100,
               do_ts_images = False,is_inception=False,saving_file_ending=None):
    """
    

    Parameters
    ----------
    model : pytorch neural network model
        
    data_object : data_preparation
        This is an instance of the class data_preparation. 
        It should contain all the motion words so be sure to have them.You can run 'data_object.load_motion_word_dataset(path)'
        
    transformer : function
        This is the function that transforms the input to meet the network requirement.
        
    embedding_dimension : int
                
    batch_size : int, optional
        The default is 100.
        
    do_ts_images : boolen, optional
        States if the motion words should be transformed into images. The default is False.
        
    is_inception : boolean, optional
        States if the backbone is an inception model. The default is False.

    Returns
    -------
    None
    The embeddings will be save at './../data/embeddings' so one can load them easily.

    """
    #--Setting up model parameters
    model = model.double()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    #---Evaluation
    all_names = []
    dataloader = data_object.motion_word_dataset
    output = dict()
    
    embeddings = torch.zeros((len(dataloader),embedding_dimension)) #((len(dataloader), embedding_dimension))
    count = 0   
                 
    for b in range(0,len(dataloader), batch_size):
        
        data, motion_names = build_batch(dataloader, data_object, batch_size, b, 'eval', do_ts_images)

        #--Data transform
        if not do_ts_images or data.shape[1]==1:
            data  =  torch.tensor(data).clone().detach().expand(-1,3,-1,-1)
        
        else :
            data = transformer(torch.tensor(data).clone().detach())
        
        #--Forward pass
        with torch.no_grad():
            
            data = data.to(device)
            
            if is_inception:
                
                outputs, aux_outputs = model(data)
                
            else :
                
                outputs = model(data)
                
        #--Motion names       
        all_names = all_names + motion_names
        
        #--Embeddings
        embeddings[b:b+batch_size] = outputs
        
        count += batch_size
        print(f"\n The process is at : {100*count/len(dataloader):.2f}%")
        
        #---Updating the    
        for indx, name in enumerate(motion_names):
            output[name] = outputs[indx].cpu().numpy()
            #output[name] = outputs[indx].numpy()
    
    #--------- saving
    file_name = f'./../data/embeddings_{saving_file_ending}'if saving_file_ending is not None else './../data/embeddings'
    print(f'Saving the embeddings at {file_name}')
    with open(file_name,'wb') as file :
        pickle.dump(output,file)
        
    return None 


def cluster_dataset(embeddings, motion_names,num_clusters=200,display_silhouette_score=False):
    """This function will compute the clusters in the datasets using kmeans algorithm.
    

    Parameters
    ----------
    embeddings : 2D numpy array
        It represents all the motion words embeddings.
    
    motion_names : list of strings
        A list of the motion words names with the same ordering as the embeddings i.e embeddings[k] is linked to motion_names[k]
                
    num_clusters : int, optional
        Number of clusters to compute. The default is 200.
        
    display_silhouette_score : boolean, optional
        The default is True.

    Returns
    -------
    names_and_assignments : dictionary
        It maps all the motion words to the cluster they are assigned to.
        
    size_of_clusters : list
        This list gives the number of motion words assigned to each cluster i.e size_of_clusters[k] give the number of assignements to the cluster k.
        
    centers : numpy arrays
        It stores the cluster centers.

    """
        
    #----Kmeans
    kmeans_mini_batch = cluster.MiniBatchKMeans(n_clusters = num_clusters, batch_size=100)
    
    #----Retrieve cluster assignments
    assignments = kmeans_mini_batch.fit_predict(embeddings).tolist()
    
    #---Retrieve cluster centers
    centers = kmeans_mini_batch.cluster_centers_
    
    #---Compute size of clusters
    size_of_clusters = [assignments.count(i) for i in range(num_clusters)]
    
    #---Compute silhouette scores
    silhouette_scores = metrics.silhouette_samples(embeddings, assignments)
                   
    #---Updating output dictionary
    names_and_assignments = {motion_names[indx]:assign for indx, assign in enumerate(assignments) } 
    
    if display_silhouette_score :
        
        fig, ax = plt.subplots(2,1,figsize=(15,15) )
        ax[0].plot(silhouette_scores,'*b')
        ax[0].set_xlabel('word motion indice',fontsize=20)
        ax[0].set_ylabel('Score',fontsize=20)
        ax[1].hist(silhouette_scores,width=0.1)
        ax[1].set_xlabel('word motion indice',fontsize=20)
        ax[1].set_ylabel('Score',fontsize=20)
        fig.suptitle('Silhouette scores',fontsize=30)
        plt.show()
        
       
        
    return names_and_assignments , size_of_clusters , centers, kmeans_mini_batch


def motion_signatures(data_object_embeddings,
                      num_clusters=200,
                      plot_embeddings=True,
                      plot_signatures=True,
                      plot_pairwise_distances=True,
                      names_and_assignments=None,
                      size_of_clusters=None,
                      cluster_centers = None,
                      sequence_decomposition=None):
    """
    

    Parameters
    ----------
    data_object_embeddings : data_preparation
        This is an instance of the class data_preparation that has the embedding values stored at "data_object_embeddings.motion_word_dataset".
     
    num_clusters : int, optional
        Number of clusters to compute. The default is 200.
        
    plot_embeddings : boolean, optional
        If True, we run PCA on the embeddings and plot motion words along with the cluster center. The default is False.
        
    plot_signatures : boolean, optional
        The default is True.
        
    plot_pairwise_distances : TYPE, optional
        DESCRIPTION. The default is True.
        
    names_and_assignments : dictionary, optional
        It maps all the motion words to the cluster they are assigned to. The default is None.
        
    size_of_clusters : list of ints
        This list gives the number of motion words assigned to each cluster i.e size_of_clusters[k] give the number of assignements to the cluster k.
       
    cluster_centers : numpy arrays, optional
        The default is None.
        
    sequence_decomposition : dictionary, optional
        By default, "sequence_decomposition = data_object_embeddings.mapping_Of_motionWordsNames". The default is None.

    Returns
    -------
    sequences_signatures : dictionary
        It maps each sequence to its computed signature.

    """

    #----
    sequences_signatures = dict()
    all_names = list(data_object_embeddings.motion_word_dataset.keys())

    #--Computing clusters if necessary
    condition = [val is None for val in [names_and_assignments, size_of_clusters, cluster_centers]]
    if any(condition)  :
        all_names = list(data_object_embeddings.motion_word_dataset.keys())
        embeddings = np.stack(list(data_object_embeddings.motion_word_dataset.values()),axis=0)
        
        names_and_assignments , size_of_clusters , cluster_centers, _  = cluster_dataset(embeddings,
                                                                              motion_names =  all_names, 
                                                                              num_clusters = num_clusters)
    
    #--Loading the context
    #data_object_embeddings.load_motion_words_context()
    if sequence_decomposition is None :
        sequence_decomposition = data_object_embeddings.mapping_Of_motionWordsNames
        
      
    #--Inverse document frequency function
    idf = lambda x : np.log( len(all_names)/(1+size_of_clusters[x]) )
    
       
    #--Computing all sequences signatures in the dataset
    #count = 0
    for sequence_name , motions_in_sequence in sequence_decomposition.items():
        #count += 1 
        
        #Fetching the clusters assignments of the motion words of this sequence
        assignments = [ names_and_assignments[name] for name in motions_in_sequence ]
                        
        #--Computing the signature of the sequence    
        signature = [idf(i)*assignments.count(i)/len(motions_in_sequence) for i in range(num_clusters)]
                    
        #--Updating
        sequences_signatures[sequence_name] = np.array(signature)
        
        #if count % 50 == 0 :
            #print(f"Signature computing progress is : {100*count/len(sequence_decomposition):0.2f} %.")
               
    
       
    #--------------------------------Plotting---------------------------------------------------------------------
    if plot_signatures:
        
        labels = list(sequences_signatures.keys())
        num_cols = 4 if len(labels) > 4 else 1
        num_rows = len(labels)//4 
        num_rows = num_rows + 1 if len(labels)%4 != 0 else  num_rows 
        
        fig,_ = plt.subplots(num_rows,num_cols,figsize=(60,60))#,constrained_layout=True)
        fig.tight_layout(pad=4.0)
        
        print(f'There are {len(labels)} signatures to be plotted. ')
        
        for indx,label in enumerate(labels):
            plt.subplot(num_rows,num_cols,indx+1)
            plt.bar(np.arange(num_clusters),sequences_signatures[label],width=0.35) #ax[indx%num_rows,indx%num_cols]
            plt.title(label,fontsize=20)
            plt.ylabel('Frequency',fontsize=20)
            plt.xlabel('Cluster indice',fontsize=20)
        
        plt.show()
        
    if plot_embeddings:
        
        pca = PCA(n_components=2)
        data = pca.fit_transform(embeddings)
        cluster_centers = pca.fit_transform(cluster_centers)
                
        plt.scatter(data[:,0],data[:,1],cmap='g',label='Data')
        plt.scatter(cluster_centers[:,0],cluster_centers[:,1],cmap='r',label='cluster centers')
        plt.title('Embeddings')
        plt.legend()
        plt.show()
        
    if plot_pairwise_distances :
        
        #--Computing the earth mover distances between the sequences
        X = np.vstack(list(sequences_signatures.values()))
        cluster_densities = [1/np.exp(idf(x)) for x in range(num_clusters)]
        func = lambda x,y: wasserstein_distance(x,y, u_weights= cluster_densities,v_weights=cluster_densities)
        distances = pdist(X,func) 
        #-----
        plt.boxplot(distances)
        plt.title('Pairwise distance - Earth mover distance')
        plt.show()
        
    return sequences_signatures


def embedding_quality_assessment_OneVsRest(signatures_dict,
                                           path_meta_data='./../data/csv/metadata/metadata_train.csv'):
    """This function runs a 7-fold cross validation using a OneVsRest strategy to assess the quality of the computed embeddings.
    
    It uses sklearn OneVsRest classifier : https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html .
    

    Parameters
    ----------
    signatures_dict : dictionary
        A dictionary mapping sequences to their signatures.
        
    path_meta_data : string, optional
        path to metadata of the sequences. The default is './../data/csv/metadata/metadata_train.csv'.

    Returns
    -------
    None.

    """
    #---Reading the metadata
    df = read_csv(path_meta_data,header=0,sep=';',encoding='utf-8')
    
    #---Renaming the filenames to match the sequence's
    for i in range(df.shape[0]):
        renaming = str.split(df['filename'].iloc[i],sep='.')[0]+'_rot'
        df.loc[i,'filename'] = renaming
        
    #---Changing the indexing --> indexing by filename
    df.to_csv('./../data/csv/metadata/metadata_modified.csv')
    df = read_csv('./../data/csv/metadata/metadata_modified.csv',header=0,sep=',',encoding='utf-8',index_col='filename')

    #---Fetching interesting columns
    labels = ['kungfu_style_1','kungfu_style_2','weapon','master']
    df = df[labels]
    
    #---Initializing the onehot encoder
    encoder = OneHotEncoder(sparse=True)
    
    #---Stacking the signature values
    X = np.vstack(list(signatures_dict.values()))
   
    #---Running the cross validation for each label using OneVsRest classifier
    for label in labels :
        
        y = []
        print('-'*50,'\nLabel --> ',label,'\n')
        
        #---Fetching the class of each sequence for a given label
        for sequence_name in signatures_dict.keys() :
            name = sequence_name.split(sep='+')[0]  #--In case we have subsequences that ende with '+sub_k'
            dummy = df.loc[name][label]
            y.append([dummy])  
        
        #---Encoding categorial data in y
        y = encoder.fit_transform(y)
        
        #---Setting the scoring methods
        scoring = ['f1_micro','recall_micro','accuracy']
        cv = RepeatedKFold(n_splits=7,random_state=1)
        
        #---loading classifier
        #kernels can 'rbf' , 'linear' , 'poly', 'sigmoid'
        classifier = OneVsRestClassifier(SVC(kernel='rbf', C=1, random_state=0),n_jobs=-1).fit(X, y)
        
        #---Computing cross validation scores and printings
        scores = cross_validate(classifier, X, y, cv=cv,scoring=scoring)
        for key in scores.keys():
            if key not in ['fit_time','score_time'] :
                print(key)
                print("%0.2f score with a standard deviation of %0.2f" % (scores[key].mean(), scores[key].std()),"\n")
     
    return None



def embedding_quality_assessment_similarityFuncVsNN(motion_word_dataset,
                                                    data_object_embedding,
                                                    similarityFunc = ['dtw','soft_dtw','gak'],
                                                    plot_title='',
                                                    mode='train',
                                                    include_noisy_data=True):
    """Embedding quality assessment by comparison with a similarity function. 
    For simplicity, we load training/testing datasets and we check how well the embeddings perform to separate positive samples from negative samples.
    

    Parameters
    ----------
    motion_word_dataset : dictionary
        It is a dictionary that maps motion words to their values.
        
    data_object_embedding : data_preparation
        An instance of data_preparation that has the embeddings at 'data_object_embedding.motion_word_dataset'.

    similarityFunc : string, optional
        The name of the similarity function to consider. The allowed values are : 'gak','soft_dtw','dtw'. The default is 'dtw'.
    
    Returns
    -------
    values_emd : 2D numpy array
        Positive and negative distances computed using the embeddings. The first column relates to the positive distances and the second column to the negative distances.
    
    values_sim : 2D numpy array
        Positive and negative distances computed using 'similarityFunc'. The first column relates to the positive distances and the second column to the negative distances.

    """
    
    #-- square L2 norm
    def square_l2_norm(X,Y):
        return np.linalg.norm((X-Y),ord=2)**2
        
    #-- Scaler
    def scaler(X):
        X = np.array(X)
        mean_ = np.mean(X)
        std = np.std(X)
        return (X-mean_)/std
        
    #--Datasets
    embeddings = data_object_embedding.motion_word_dataset
    num_motion_words =len(list(data_object_embedding.motion_word_dataset.keys()))    

    if mode == 'train':
        #--Load positive and negative samples dataset
        paths_to_positive_negative_samples = ['./../data/datasets/training/training_dataset_10similarity_dtw',
                                             './../data/datasets/training/training_dataset_10similarity_gak',
                                             './../data/datasets/training/training_dataset_10similarity_soft_dtw']
        
        if include_noisy_data :
            paths_to_positive_negative_samples = ['./../data/datasets/training/training_dataset_10_Noisy_True_similarity_gak',
                                                  './../data/datasets/training/training_dataset_10_Noisy_True_similarity_dtw',
                                                  './../data/datasets/training/training_dataset_10_Noisy_True_similarity_soft_dtw']
            
            
    if mode == 'test':
        paths_to_positive_negative_samples = ['./../data/datasets/testing/testing_dataset_10similarity_gak',
                                              './../data/datasets/testing/testing_dataset_10similarity_dtw',
                                             './../data/datasets/testing/testing_dataset_10similarity_soft_dtw']        
        if include_noisy_data :
            paths_to_positive_negative_samples = ['./../data/datasets/testing/testing_dataset_10_Noisy_True_similarity_dtw',
                                                 './../data/datasets/testing/testing_dataset_10_Noisy_True_similarity_gak',
                                                 './../data/datasets/testing/testing_dataset_10_Noisy_True_similarity_soft_dtw']
    
    #-------
    dataset = data_object_embedding.merge_training_data(paths_to_positive_negative_samples)
    #data_object.path_to_training_dataset = path_to_positive_negative_samples
    #dataset = data_object.load_training_dataset()
    print('\nThe dataset considered for evaluation is : ', paths_to_positive_negative_samples,'\n')
    
    #--------Plotting  
    distances_emdedding = dict()
    fig, axs = plt.subplots(1, 3,figsize=(30,10), constrained_layout=True) 
    colors = ['#3b78db', '#bc7f2d', '#122a3f']
    count = 0

    for func_indx, function in enumerate(similarityFunc) :
        
        #--Retrieving the similarity function
        func = data_object_embedding.similarity_functions[function] 
        
        #--Dictionary
        distances_similarity_func = dict()
        dist_pos_embed = []
        dist_neg_embed = []
        
        for anchor_name, values in dataset.items():
            
            dist_pos_1 = []
            dist_neg_1 = []
            
            for pos,neg in zip(values[0],values[1]):
                
                if func_indx < 1 :  #--We only compute the embedding distances once for a given model.
                    #---L2 distance for embeddings
                    dist_pos_embed.append(square_l2_norm(embeddings[anchor_name],embeddings[pos]))
                    dist_neg_embed.append(square_l2_norm(embeddings[anchor_name],embeddings[neg]))
                
                #---Similarity score
                dist_pos_1.append(func(motion_word_dataset[pos],motion_word_dataset[anchor_name]))
                dist_neg_1.append(func(motion_word_dataset[neg],motion_word_dataset[anchor_name]))
            
            #---Distances
            #--- We are computing the coefficients of variation.
            if func_indx < 1 :
                distances_emdedding[anchor_name] =[np.std(dist_pos_embed)/(1+np.mean(dist_pos_embed)),
                                                   np.std(dist_neg_embed)/(1+np.mean(dist_neg_embed))]
            
            distances_similarity_func[anchor_name] = [np.std(dist_pos_1)/(1+np.mean(dist_pos_1)),
                                                      np.std(dist_neg_1)/(1+np.mean(dist_neg_1))]
            
            if count % 99 == 0 :
                progress = 100*(count+1)/(num_motion_words*len(similarityFunc))
                print(f"The computation is at : {progress:.2f}%.")
            count += 1
            
        #if func_indx < 1 :
        values_emd = np.vstack(list(distances_emdedding.values()))
        axs[func_indx].scatter(values_emd[:,0],
                            values_emd[:,1],
                            c='#ff0000')
            
        values_sim = np.vstack(list(distances_similarity_func.values()))
        #-------------
        axs[func_indx].scatter(values_sim[:,0],values_sim[:,1],c=colors[func_indx])
        axs[func_indx].legend(["Embeddings",function])
        axs[func_indx].set_xlabel('Positive samples')
        axs[func_indx].set_ylabel('Negative samples')

    
    #--Save figure
    title = plot_title
    plt.suptitle(title)
    plt.savefig(plot_title)
    
    return values_emd, values_sim    

#-------------------------------------- Demo for evaluation---------------------
if 'main' == '0main':
    start = time.time()
    mode = 'train' # 'test'
    data = data_preparation(mode = mode,include_noisy_data=True)
    data_embeddings = data_preparation(mode= mode,include_noisy_data=True)
    
    #-----------------------Parameters-------------------------
    data.set_motionWords_parameters(number_of_frames=16,overlaping_degree=0.75)
    data.set_featureExpansion_parameters(do_PCA=True,number_of_components=50)
    data.csv_files_mining(select_high_variance_features=None,expand_features=True)
    data.compute_motion_words(divide_sequence=5)
    motion_word_dataset = copy.deepcopy(data.motion_word_dataset)
    #----imaging
    #---- Setting parameters
    data.set_imaging_parameters(do_recurrenceP = False, RP_threshold = 'point',
                               RP_timeDelay = 1, RP_dimension = 0.25,
                               do_GAF = False, GAF_method = 'summation',
                               do_MTF = False,
                               square_reshape = True,
                               vertical_stacking = False,
                               horizontal_stacking = False) 
    
    #--Loading the model parameters
    num_classes = 700
    model_name = 'resnet18'
    model , input_size = initialize_model(model_name = model_name, num_classes = num_classes,
                                              feature_extract= False, use_pretrained=False)
    #--Iterating over models
    
    for k in [1] :
    
        """
        ['./../data/models/model_resnet18_training_dataset_5similarity_dtw.pth',
               './../data/models/model_resnet18_training_dataset_10similarity_dtw.pth',
               './../data/models/model_resnet18_MergedData.pth'] :
        """
        
        model_path = './../data/models/model_resnet18_RecordingParam_tsImages_False_includeNoisy_True.pth'
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict']) 
        print('Loss :',checkpoint['loss'].item(),'; Num of epoch',checkpoint['epoch'])
        
        #model_resnet18_training_dataset_10similarity_dtw
        #model_resnet18_MergedData
        #data.MTF = True
        #data.recurrence_plot = False
        ts_images = False

        #if os.path.exists(model_path):
            #model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')),strict=True)
            #print(f"Model : {model_path} --> loaded.")
            
        #--Loading the transformer
        transformer = transform_data(input_size, is_ts_images = ts_images)
        
        #--Running the evaluation
        evaluation(model, 
                   data,
                   transformer,
                   embedding_dimension = num_classes,
                   batch_size = 180, 
                   do_ts_images = ts_images,
                   is_inception=False)
        
        #--- Load embeddings
        data_embeddings.load_motion_word_dataset(path='./../data/embeddings')
        
        sim_func = ['dtw','soft_dtw','gak'] 
        print(f'\n The similarity function used is : {sim_func}')
        plot_title =f"{os.path.splitext(os.path.basename(model_path))[0]}"
        
        values_emd, values_sim = embedding_quality_assessment_similarityFuncVsNN(motion_word_dataset, 
                                                                                 data_embeddings, 
                                                                                 similarityFunc = sim_func,
                                                                                 plot_title=plot_title,
                                                                                 mode=mode,
                                                                                 include_noisy_data=True)
            
        
        print(f" It tooks {time.time()-start:0.2f} seconds.")
        
