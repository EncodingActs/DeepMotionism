import numpy as np
import tslearn.metrics   
#---------------------------------Similarity calculation---------------------------------------------------------------
def similarity_dtw(input1,input2,metric="cosine",itakura=False,itakura_max_slope=2.,sakoe_chiba=True,sakoe_chiba_radius=10):
    """This method will compute the similarity between the two videos.
    
    Parameters
    ----------
    video1 : 2D numpy array
        the pre-processed features of input1 is a numpy array of shape (size1*d) where each column is a frame.
        
    video2 : 2D numpy array
        the pre-processed features of input2 with is a numpy array of shape (size2*d).
        
    metric : String, optional
        specifies the metric to use for calculating the similarity using DTW
        if metric is not given then it assumes an euclidean distance.
        The default is "cosine".
        We advise to use one of the these: ‘chebyshev’, ‘cityblock’,
        ‘correlation’, ‘cosine’, ‘seuclidean’, ‘sqeuclidean’.
        it follows  'scipy.spacial.distance.pdist' as it is the underlying function being run.
        
    itakura : Boolean, optional
        Specifies if the itakura algorithm is to be used. The default is False.
    
    itakura_max_slope : float, optional
        The default is 2..
        
    sakoe_chiba : Boolean, optional
        specifies if the sakoe_chiba algorithm is to be used. The default is True.
        
    sakoe_chiba_radius : Int, optional
        The restriction window size along the diagonal. The default is 10.

    Returns
    -------
    cost : float
        The similarity value.

    """
    
    #--Restricting the path to resemble a parallelogram --> itakura method
    if itakura :
        _,cost = tslearn.metrics.dtw_path_from_metric(input1,input2,metric,global_constraint="itakura",
                                   itakura_max_slope=itakura_max_slope)
    
    #--Restricting the path to resemble a band --> sakoe_chiba method
    if sakoe_chiba :
        _,cost = tslearn.metrics.dtw_path_from_metric(input1,input2,metric,global_constraint="sakoe_chiba",
                                   sakoe_chiba_radius=sakoe_chiba_radius)
        
    #--Using the traditional formulation that goes through the whole distance matrix
    else :
        _,cost = tslearn.metrics.dtw_path_from_metric(input1, input2)
    
    
    return cost



def similarity_soft_dtw (input1, input2, gamma=1):
    """This method uses a novel algorithm that uses a soft-min approach to dynamic time wrapping 

    
    Parameters
    ----------
    input1 : 2D numpy array of floats
        The first input
        
    input2 : 2D numpy array of floats
        The second input.
        
    gamma : int, optional
        When gamma = 0, then it is equivalent to the standard DTW. The default is 1.

    Returns
    -------
    score : float
        The lowest the score, the highest similarity.

    """
                
    score = tslearn.metrics.soft_dtw(input1, input2,gamma)
        
    return score

def similarity_gak(input1,input2):
    """This function computes the distance between two inputs using the global alignment kernel. For more information visit : https://tslearn.readthedocs.io/en/stable/gen_modules/metrics/tslearn.metrics.gak.html#tslearn.metrics.gak
    

    Parameters
    ----------
    input1 : 2D numpy array of floats
        The first input
        
    input2 : 2D numpy array of floats
        The second input.
    
    Returns
    -------
    score : float
        The lowest the score, the highest similarity.

    """
    #--Computing sigma
    sigma = 7.5 #tslearn.metrics.sigma_gak(np.vstack([input1,input2]),n_samples=input1.shape[0],random_state=0)
    
    #--Computing the score
    score = 1/(1+tslearn.metrics.gak(input1, input2,sigma))
    
    return score
#---------------------------------Retrieving top k--------------------------------------------------------------------------

def retrieve_top_k(input_ , input_name, lookUp_dataset, k = 10, func = similarity_gak):
    """This function retrieves the top k closest and 'farest' motions words.
    
    Parameters
    ----------
    input_ : 2D numpy array
        The shape is : (num_samples,num_features).
        
    input_name : string
        The name of the motion word for which we want to compute the closest and farest motion words.
        
    lookUp_dataset : Dictionary
        The dataset that is to be used to look for the closest and farest motion words.
        
    k : unsigned int, optional
        The number of modion words to retrieve. The default is 10.
        The 'farest' represent hard negatives so they are not the top k with worst score but the k motion words having scores coming just after the median score. 
    
    func : string.
        {"similarity_soft_dtw","similarity_dtw","similarity_gak"}.

    Returns
    -------
    closest_values : list of numpy arrays
        K closest values.
        
    closest_names : list of strings
        Names of k closest values.
        
    farest_values : list of numy arrays
        k 'farest' values.
        
    farest_names : list of strings
        Names of k 'farest' values.

    """
    
    assert type(k) is int, "k must be an integer"
    assert abs(k) <= len(lookUp_dataset.values())," k is too large, use a smaller value for k."
    
       
    #--iterating over values in lookUp_dataset
    scores = []
    names = []
    reference = func(input_,input_)
    
    for name, value in lookUp_dataset.items():
        if name != input_name :
            score = abs(reference - func(input_, value)) #/reference
            scores.append(score)
            names.append(name)
        
    #--sorting
    sorted_k = [[name,score] for score,name in sorted(zip(scores,names))] 
    
    #--Getting the top k closest
    closest_names = [sorted_k[i][0] for i in range(k)]      
    closest_values = [lookUp_dataset[name]  for name in closest_names]
    closest_values = np.stack(closest_values,axis=0)
    
    #--Getting the hard negative samples with scores coming after the median.
    size = len(scores)//2
    farest_names = [sorted_k[i][0] for i in range(size, size+k)]
    farest_values = [lookUp_dataset[name]  for name in farest_names]
    farest_values = np.stack(farest_values,axis=0)
    
    #--Storing results in a dictionary
    return closest_values, closest_names, farest_values,farest_names
    


        

  



    
    
    














