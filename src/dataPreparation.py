import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pyts.image import RecurrencePlot,  GramianAngularField, MarkovTransitionField
import os, fnmatch
import pickle
import time
from pandas import read_csv
from similarity_calculation  import retrieve_top_k,similarity_soft_dtw,similarity_dtw,similarity_gak
from re import split
import gc
import math
from copy import deepcopy

class data_preparation:
    """The class data_preparation allows to gather all the tools necessary to do the following :
            
            - mine extracted joint rotation angles
            - compute motion words
            - transform motion words into images (recurrence plot, markov transational fields, gramian angular fields)
            - compute training datasets with different distance metrics : soft-DTW, DTW, global alignment kernel (gak) available at self.similarity_functions
        
        The two most important class members are : 
            - self.motion_word_dataset --> is a dictionary that maps each computed motion word to its value as follow {name:value}.
            - self.mapping_Of_motionWordsNames --> is a dictionary that maps each sequence to its motion words as follows {sequence_1:[motionword_1,motionword_2,...,motionword_n]}.The motion words have the same name as the sequence except that we add "_k" where k specifies its location in the sequence
    
        Methods
        -------
        
        
    """
    
    def __init__(self,  path_to_training_dataset=None, mode = 'train',include_noisy_data=False):
        """Constructor. Do not forget the mode which is either 'train' or 'test'. In case it is mode='test',  the csv at 'data/csv/Test' will be used. If mode='train', the csv at 'data/csv/Train' will be used
            
        

        Parameters
        ----------
        path_to_training_dataset : string, optional
            DESCRIPTION. The default is None.
            
        path_csv_directory : string, optional
            DESCRIPTION. The default is './../data/csv/Train/'.
            
        mode : string, optional
            It can be 'test' or 'train'. The default is 'train'.

        Returns
        -------
        None.

        """
        
        #--- Mode
        assert mode in ['test','train'], "The mode should be 'test','train'."
        self.mode = mode
        self.include_noisy = include_noisy_data
        
        #--- Paths to files to extract or  datasets
        if self.mode == 'train':
            self.paths_to_csv_directory = './../data/csv/Train/'  
            self.path_to_noisy = './../data/csv/Train/noisy/' if include_noisy_data == True else None
        else:
            self.paths_to_csv_directory = './../data/csv/Test/'
            self.path_to_noisy = './../data/csv/Test/noisy/' if include_noisy_data == True else None  
            
        #---
        self.path_to_dataset = None
        self.path_to_motion_words = f'./../data/motion_words_{self.mode}'
        
        #--- Path to positive and negative examples
        self.path_to_training_dataset = path_to_training_dataset
        
        #---Dictionaries to hold motion words
        self.motion_word_dataset = dict()
        self.mapping_Of_motionWordsNames = dict()
        self.mapping_of_subsequences = dict()
        
        #---Motion words segmentation
        self.number_of_frames  = 16
        self.overlaping_degree = 0.75
        
        #---Feature expansion and PCA
        self.do_PCA = True
        self.number_of_components = 50
        
        #--- Functions for similarity calculation
        self.similarity_functions = dict()
        self.similarity_functions['gak']= similarity_gak
        self.similarity_functions['dtw']= similarity_dtw
        self.similarity_functions['soft_dtw']= similarity_soft_dtw
       
    #------------------------------- Setters-------------------------------------------------
    def set_motionWords_parameters(self,number_of_frames = 16, overlaping_degree = 0.75):
        """This function set the parameters of computing motion words. One specifies the number of frames and the overlaping degree between motion words.
        

        Parameters
        ----------
        number_of_frames : int, optional
            The default is 16. We recommend to give a number for which the squre root is an integer to guarantee a square reshaping if one plans to transform the motion words into images.
            
        overlaping_degree : int, optional
            The overlaping degree between two consecutive motion words. The default is 0.75.

        Returns
        -------
        None.

        """
        
        self.number_of_frames  = number_of_frames
        self.overlaping_degree = overlaping_degree
        

    def set_featureExpansion_parameters(self,do_PCA=True,number_of_components = 50 ):
        """
        

        Parameters
        ----------
        do_PCA : Boolean, optional
            It says if we should do PCA on the motion word. The default is True.
            
        number_of_components : int, optional
            The default is 50.

        Returns
        -------
        None.

        """
        
        self.do_PCA = do_PCA
        self.number_of_components = number_of_components
        
        return None
        
    def set_imaging_parameters(self, do_recurrenceP = True, RP_threshold = 'point',
                               RP_timeDelay = 1, RP_dimension = 0.25,
                               do_GAF = False, GAF_method = 'summation',
                               do_MTF = False,
                               square_reshape = True,
                               vertical_stacking = False,
                               horizontal_stacking = False ):
        
        """This method sets the timeseries imaging parameters.If more than one method is chosen you get, images concatenated along the first axis.
            
        Choose between : do_recurrenceP, do_GAF, do_MTF and set only one or all of them to "True".
        
        For more information on time series imaging visit : https://pyts.readthedocs.io/en/stable/api.html#module-pyts.image
        
        Parameters
        ----------
        do_recurrenceP : Boolean, optional
            States if a recurrence plot should be computed. The default is True.
            
        RP_threshold : string, optional
            Threshold for the minimum distance. If None, the recurrence plots are not binarized.
            If ‘point’, the threshold is computed such as percentage percents of the points are smaller than the threshold.
            If ‘distance’, the threshold is computed as the percentage of the maximum distance. The default is 'point'.
            
        RP_timeDelay : int or float, optional
            Time gap between two back-to-back points of the trajectory.
            If float, If float, it represents a percentage of the size of each time series and must be between 0 and 1. The default is 1.
            
        RP_dimension : int or float, optional
            Dimension of the trajectory. If float, If float, it represents a percentage of the size of each time series and must be between 0 and 1. The default is 0.25.
            
        do_GAF : Boolean, optional
            States if the gramian angular field should be computed. The default is False.
            
        GAF_method : string, optional
            Type of Gramian Angular Field. ‘s’ can be used for ‘summation’ and ‘d’ for ‘difference’. The default is 'summation'.
            
        do_MTF : Boolean, optional
            States if the Markov transition field should be computed. The default is False.

        NB: if do_MTF == True , do_GAF == True, do_recurrenceP == True, 
            we will get an RGB-like image where the resulting images from these methods have been stacked along the first channel i.e. the shape will be [3,n,m].
        
        Returns
        -------
        None.

        """
        #assert [do_recurrenceP,do_GAF, do_MTF].count(True) == 1, 'Their should be only one method selected.'
        
        self.recurrence_plot = do_recurrenceP
        self.RP_threshold = RP_threshold
        self.RP_timeDelay = RP_timeDelay
        self.RP_dimension = RP_dimension
        self.GAF = do_GAF
        self.GAF_method = GAF_method
        self.MTF = do_MTF
        self.square_reshape = square_reshape
        self.vertical_stacking = vertical_stacking
        self.horizontal_stacking = horizontal_stacking
        self.image_size = 50

        return None
       
    #------------------------------------------------------------------------------------------------------
    def csv_files_mining(self, select_high_variance_features=True, feature_labels=None, expand_features = True):
        """This function runs the feature selection on all the converted bvh files for which the csv files are provided in '../data/csv/'.
        It writes the result to an external file that is available "self.path_to_dataset".
        

        Parameters
        ----------
        select_high_variance_features : Boolean or None, optional
                The default is True. If it is *None* then all features are considered. no selection is done.
                
        feature_labels : None or array of strings, optional
            The strings should be the features names to be selected. We use pandas to read the csv files so it's important to have to correct columns names. The default is None.
            
        expand_features : Boolean, optional
            States if features should be expanded. The default is True. 

        Returns
        -------
        None.

        """
        
        csv_paths = [self.paths_to_csv_directory+file for file in os.listdir(self.paths_to_csv_directory) if fnmatch.fnmatch(file, "*.csv")]
        
        if self.include_noisy and os.path.exists(self.path_to_noisy) :
            csv_paths_noisy = [self.path_to_noisy+file for file in os.listdir(self.path_to_noisy) if fnmatch.fnmatch(file, "*.csv")]
            csv_paths = csv_paths + csv_paths_noisy
            
        print("\n","-"*20,"\n CSV files mining has started !")
        print(f" We're mining the file at : {self.paths_to_csv_directory}.")
        
                    
        if select_high_variance_features :
            labels_path = './selected_features_labels_highestFrequency.txt'
            dataset_name='dataset'+'_HighVariance'+f'_{self.mode}'
            
        else:
            labels_path = './selected_features_labels.txt'
            dataset_name='dataset'+'_NotHighVariance'+f'_{self.mode}'
    
        if select_high_variance_features is not None :
            #--Retrieve labels
            with open(labels_path,'rb') as file:
                labels = pickle.load(file)
            
        #--Create the holder for all extracted features
        joint_angle_values = dict()
        
        #--Iterating through the paths 
        for _,path in enumerate(csv_paths):
            file_name = os.path.basename(path)
            file_name = os.path.splitext(file_name)[0]
            
            #--Extract the values
            loaded_data =  read_csv(path)
            
            #--Selecting wanted features from the csv files
            if select_high_variance_features is None :
                labels = loaded_data.columns.tolist() 
                
            #--Getting labels if available
            if feature_labels is not None :
                labels = feature_labels
                
            #--Retrieve selected features
            if expand_features :
                joint_angle_values.update({file_name : self.featureExpansion_angles(loaded_data[labels].to_numpy())})
            else:
                joint_angle_values.update({file_name : loaded_data[labels].to_numpy()})

        #--Saving
        self.path_to_dataset =f'./../data/datasets/{dataset_name}'
        
        #--Opening or creating the dataset
        with open(self.path_to_dataset,"wb") as file:
            pickle.dump(joint_angle_values, file)
            
          
        print(f"\n The data has been successfully processed and is located at : {self.path_to_dataset}")
        
        
        return None
    
    
    
    def compute_motion_words(self,divide_sequence=0):
        """This function computes the motion words from the extracted joint-rotations.
        The motion words are stored in 'self.motion_word_dataset'.
        

        Parameters
        ----------
        divide_sequence : int, optional
            This will emulate having more data which can be useful for the validation stage.Each sequence will be divided in 'divide_sequence' parts.  If divide_sequence=0, nothing is done. The default is 0.

        Returns
        -------
        None.

        """

        
        print("\n","-"*20,"\n Computing motion words !")
               
        #-- number of frames to skip
        skip = int(self.overlaping_degree * self.number_of_frames)
        
        #-- Loading the extracted csv
        with open(self.path_to_dataset,"rb") as file :
            data = pickle.load(file)
        
        #--Iterating through the dataset
        for key, value in data.items():
            file_name = key
            motion_words_names = []
            
            indx = 0
            #--Computing the motion words
            while( indx*skip + self.number_of_frames <= value.shape[0]):
                word_value = np.copy(value[indx*skip : indx*skip + self.number_of_frames])
                word_name = file_name + f"_{indx}"
                               
                self.motion_word_dataset.update({word_name : word_value})
                motion_words_names.append(word_name)
                indx += 1
                
            self.mapping_Of_motionWordsNames.update({file_name : motion_words_names})
            
            print("\n Number of words : ", len(motion_words_names), f" in {file_name}")
        
        #---Artificially adding subsequences from data
        if divide_sequence > 1 :
            
            dummy = dict()
            for key,value in self.mapping_Of_motionWordsNames.items():
                subsequence_length = len(value)//divide_sequence 
                prev_ = 0
                subsequences_names = []
                for i in range(divide_sequence):
                    
                    new_key = key + f"_sub_{i}"
                    
                    if (i < divide_sequence-1):
                        next_ = prev_ + subsequence_length
                    else :
                        next_ = len(value)
                    
                    dummy.update({new_key : value[prev_:next_]})
                    prev_ = next_
                    subsequences_names.append(new_key)
                    
                #--Subsequences mapping to original file
                self.mapping_of_subsequences.update({key: subsequences_names})
                    
            #--Adding subsequences
            self.mapping_Of_motionWordsNames.update(dummy)
                    
        #----Saving the file 
        with open(self.path_to_motion_words,"wb") as file :
            pickle.dump(self.motion_word_dataset, file)
        
        with open(f'./../data/motionWord_context_{self.mode}','wb') as f :
            pickle.dump(self.mapping_Of_motionWordsNames,f)
          
        print(f'\n the motion_word_dataset is avaible at : {self.path_to_motion_words} \n',"-"*20)
        return None
    
    #--Frames indices
    def get_frames_indices(self,list_of_motion_words):
        """It will return the indices of the first and last frames given a list of consecutive motion words.
        

        Parameters
        ----------
        list_of_motion_words : list of strings
            The motion words should be reference by their name and not their values.

        Returns
        -------
        tuple
            It returns a tuple of length 2. 

        """
        #--
        if len(list_of_motion_words) > 0 :
            first_motion_word = list_of_motion_words[0]
            last_motion_word = list_of_motion_words[-1]
        
            #- start_index = 1 + motionWord_index*(1-overlaping degree)*number_of_frames
            starting_index = 1 + self.number_of_frames*int(split("_",first_motion_word)[-1])*(1-self.overlaping_degree)
        
            #- ending_index = number_of_frames + motionWord_index*(1-overlaping degree)*number_of_frames
            ending_index= self.number_of_frames + self.number_of_frames*int(split("_",last_motion_word)[-1])*(1-self.overlaping_degree)
        else:
            starting_index = -1 #-1 denotes error
            ending_index = -1 #-1 denotes error
        return int(starting_index), int(ending_index)

        #--Frames indices
    def get_frames_indices_second(self,list_of_motion_words):
        """It will return the indices of the first and last frames given a list of consecutive motion words.
        

        Parameters
        ----------
        list_of_motion_words : list of strings
            The motion words should be reference by their name and not their values.

        Returns
        -------
        tuple
            It returns a tuple of length 2. 

        """
        #--
        first_motion_word = list_of_motion_words[0]
        last_motion_word = list_of_motion_words[-1]
        
        #- start_index = 1 + motionWord_index*(1-overlaping degree)*number_of_frames
        starting_index = 1 + self.number_of_frames*int(split("_",first_motion_word)[-1])*(1-self.overlaping_degree)
        starting_index = math.floor(starting_index/120)

        #- start_index = number_of_frames + motionWord_index*(1-overlaping degree)*number_of_frames
        ending_index= self.number_of_frames + self.number_of_frames*int(split("_",last_motion_word)[-1])*(1-self.overlaping_degree)
        ending_index = math.ceil(ending_index/120)
        
        return int(starting_index), int(ending_index)
            
    def add_subsequence(self,sequence_name,starting_frameIndex,ending_frameIndex):
        """Creates a new subsequence from  starting and ending frame indices.
        

        Parameters
        ----------
        sequence_name : string
            
            
        starting_frameIndex : int
            Starting frame index. It should be at least 1.
            
        ending_frameIndex : int
            Ending frame index. If greater than the actual max frame index, it will take the last frame.

        Returns
        -------
        query_name : string
            The name of the created subsequence to query.

        """
        #--Check that it's not empty
        if bool(self.mapping_of_subsequences):
            assert sequence_name in self.mapping_of_subsequences.keys(), "Sequence_name is not correct ! Make sure that it's not a subsequence of sequence."
        
        assert starting_frameIndex < ending_frameIndex, "Starting frame index can't be superior to ending frame index"
        assert starting_frameIndex >= 1, "Starting frame index can't be negative."
        
        #--Motion words indexes
        starting_motion_word_index = (starting_frameIndex-1)/((1-self.overlaping_degree)*self.number_of_frames)
        starting_motion_word_index = int(np.floor(starting_motion_word_index))
        
        ending_motion_word_index = (ending_frameIndex - self.number_of_frames)/((1-self.overlaping_degree)*self.number_of_frames)
        ending_motion_word_index = int(np.ceil(ending_motion_word_index))
        
        #---Getting the subsequence
        num_motion_words = len(self.mapping_Of_motionWordsNames[sequence_name])

        if ending_motion_word_index > num_motion_words : 
            ending_motion_word_index =  num_motion_words - 1
            
        #--Updating the dictionary
        query_name = sequence_name+f"_queryFrames_{str(starting_frameIndex)}_{str(ending_frameIndex)}"
        motion_words_in_query = self.mapping_Of_motionWordsNames[sequence_name][starting_motion_word_index:ending_motion_word_index]
        self.mapping_Of_motionWordsNames[query_name] = deepcopy(motion_words_in_query)
        
        return query_name
    
    #--------------------------------Loading methods----------------------------------------------
    def load_motion_words_context(self,path='./../data/motionWord_context_train'):
        """This function loads the motion word context which is a dictionary that records all the motion words extracted from a sequence.
        

        Parameters
        ----------
        path : string, optional
            The default is './../data/motionWord_context'.

        Returns
        -------
        None.

        """
        if os.path.exists(path):
            with open(path,'rb') as file :
                self.mapping_Of_motionWordsNames = pickle.load(file)
        else : 
            print(f'{path} : does not exist.')
        return None
    
    
    def load_motion_word_dataset(self,path= None):
        """This function will load the motion word dataset located at self.path_to_motion_words
        

        Parameters
        ----------
        path : string, optional
            The default is None.

        Returns
        -------
        None.

        """
        if path == None and os.path.exists(self.path_to_motion_words):
            with open(self.path_to_motion_words,'rb') as file :
                self.motion_word_dataset = pickle.load(file)
        else:
            with open(path,'rb') as file:
                self.motion_word_dataset = pickle.load(file)
        print('\n The motion words dataset has been loaded to self.motion_word_dataset\n')
        return None

    def load_training_dataset(self,path=None):
        """This function loads the training dataset located at self.path_to_training_dataset.One defines  'path_to_training_dataset' when creating an instance of the class or by doing  'data_object.path_to_training_dataset = your_PATH'.

        Parameters
        ----------
        path : string, optional
            Path to training dataset can also be given directly here to be loaded. The default is None.

        Returns
        -------
        data : TYPE
            DESCRIPTION.

        """

        if path is None :
            #--Loading data
            with open(self.path_to_training_dataset,"rb") as file :
                data = pickle.load(file)
                
        elif os.path.exists(path) :
            with open(self.path_to_training_dataset,"rb") as file :
                data = pickle.load(file)
        else : 
            print("The given path doesn't exist.")
            return None
        
        return data
    
    def merge_training_data(self,paths):
        """Merge training datasets
        

        Parameters
        ----------
        paths : list of strings
            paths to training datasets.

        Returns
        -------
        output : dictionary
            merged training datasets.

        """
        output = dict()
        
        for indx, path in enumerate(paths):
            self.path_to_training_dataset = path
            data = self.load_training_dataset()
            
            if indx == 0 :
                output.update(data)
            else :
                for name,value in data.items():
                    positive, negative = value[0],value[1]
                    
                    if name in output.keys():
                        values_old = output[name]
                        output[name] = [positive+values_old[0],negative+values_old[1]]
                        
                    else :
                        output[name] =[positive,negative]
        
        with open(f'./../data/training_datasets_noisy_{self.include_noisy}','wb') as file:
            pickle.dump(output,file)
        return output
    #-------------------------------------------------------------------------------------


    def featureExpansion_angles(self,input_motion_word):
        """This function computes feature augmentation by adding velocity and acceleration across frames.
    
        Parameters
        ----------
            input_motion_word : 2D numpy array 
                Input motion_word of size [word_zise,num_features].
    
        Returns
        -------
            augmented_motion_word : 2D numpy array
    
        """
        
        #---Copying the input
        sample = input_motion_word.copy()
      
        #---Standardization
        sample = StandardScaler().fit_transform(sample)
        
        #--- Feature expansion
        sample_velocity =  np.gradient(sample, axis=0)  #np.zeros_like(sample)
        sample_acceleration =  np.gradient(sample_velocity, axis = 0 )  #np.zeros_like(sample)
        
        #--horizontal stacking the acceleration and velocites --> it gives shape=(n,m)
        augmented_motion_word = np.hstack([sample, sample_velocity,sample_acceleration])
        
        #-- Doing dimensionality reduction if requested
        if self.do_PCA :
            augmented_motion_word = self.dimensionality_reduction(augmented_motion_word)
        
        print(augmented_motion_word.shape)
        
        return augmented_motion_word
    
    
    
    def dimensionality_reduction(self,augmented_motion_word):
        
        """This function computes the dimensionality reduction on the input. The parameters are defined in 'set_featureExpansion_parameters'
    
        Parameters
        ----------
            augmented_motion_word : 2D numpy array

        Returns
        -------
            output : 2D numpy array
    
        """
        
        pca = PCA(n_components= self.number_of_components)
        output = pca.fit_transform(augmented_motion_word)
            
        
        return output
    
    
    def timeseries_to_images(self, motion_word):
        
        """This function returns a number of "word_size" images.
        Each row of the motion word is converted into an image. The parameters should be defined using self.set_imaging_parameters.
        
                
        
        Parameters
        ----------
            motion_word : 2D numpy array.
            
        Returns
        -------
            images : 3D or 2D numpy array
                It is of shape [words_size,n,m] if all imaging methods are set to True i.e self.GAF, self.MTF,self.recurrence_plot.
                Otherwise it is of shape [n,m]. 
                By default the images are reshaped as squares which is guaranteed to work when the square root of motion word size is a integer.
                
        """
        
        assert([self.GAF,self.MTF,self.recurrence_plot].count(True) > 0) , " There should be at least one method of imaging selected"
        
        #-----------------------------------
        if(all([self.MTF, self.GAF, self.recurrence_plot])):
            transfo_rp = RecurrencePlot(threshold = self.RP_threshold,
                                        dimension= self.RP_dimension,
                                        time_delay = self.RP_timeDelay)

            image_3 = transfo_rp.fit_transform(motion_word)
            
            #--Chaning image size so that they all have the same size
            self.image_size = image_3.shape[-1]
            
            #--Doing other transforms           
            transfo_gaf = GramianAngularField( method=self.GAF_method,image_size=self.image_size)
            transfo_mtf = MarkovTransitionField(image_size=self.image_size,n_bins=5,strategy='normal')
            image_1 = transfo_gaf.fit_transform(motion_word)
            image_2 = transfo_mtf.fit_transform(motion_word)
            
            #--Stacking the images to get an RGB-like
            images = np.stack([image_1,image_2,image_3] ,axis=0)
            
            #--Reshaping as square.
            _,numb_images, dimension1, dimension2 = images.shape
            size = np.sqrt(numb_images*dimension1*dimension2).astype(int)
            images = images.reshape((-1,size,size))
            
            #--Scaling the image along the axis=0
            max_ = np.max(images,axis=0)
            min_ = np.min(images,axis=0)
            images = (images - min_)/(max_ - min_)
            
            return images

        #--------------Doing other types of imaging -----------------------
        
        elif self.recurrence_plot == True :
            transformer = RecurrencePlot(threshold = self.RP_threshold ,
                                            dimension= self.RP_dimension,
                                            time_delay = self.RP_timeDelay)
            images = transformer.fit_transform(motion_word)

        elif self.GAF == True :
            transformer = GramianAngularField( method=self.GAF_method,image_size=self.image_size)
            images = transformer.fit_transform(motion_word)
        
        elif self.MTF == True :
            
            transformer = MarkovTransitionField(image_size=self.image_size)
            images = transformer.fit_transform(motion_word)
        
        #----------------------------------------------------------------------
        numb_images, dimension1, dimension2 = images.shape
        
        if(self.vertical_stacking or self.horizontal_stacking ):
            all_images = []
            for i in range(numb_images):
                all_images.append(images[i])
            
        if self.vertical_stacking :
            images = np.vstack(all_images)
        
        elif self.horizontal_stacking :
            images = np.hstack(all_images)
        
        elif self.square_reshape :
            size = int(np.sqrt(numb_images*dimension1*dimension2)//2)*2
            return images.reshape((1,-1,size))
        
        #--Scaling
        max_ = np.max(images)
        min_ = np.min(images)
        images = (images - min_)/(max_ - min_)
        
        
        return images
    
    
    def get_positive_and_negative(self, anchor_name, k,similarity_function):
        """This function computes the positive and negative examples given an anchor motion word. It is to be reminded that dynamic timpe wrapping is used to get the positive and negative examples. The negative examples are the hard negatives.
            
            PS: Ensure that the motion words haven't been transformed into images before running this function.
        
        Parameters
        ----------
        anchor_name : string
            The name of the motion word for which we want to get the positive and negative examples.
            
        k : int
            k is the number of closest or farest motion words to retrieve.
            
        similarity_function : function 
        {similarity_soft_dtw,similarity_dtw,similarity_gak}
      
        Returns
        -------
        positive_names : list of strings
            The list has a length of k.
        negative_names : list of strings
            The list has a length of k.

        """
        #assert self.imaging == False, "The current dataset has been converted into images! Run without this transformation on the dataset."
        
        #-- Retrieve the context of the motion word
        parent_name = split(r"\B_rot",anchor_name)[0]+"_rot"
        context = self.mapping_Of_motionWordsNames[parent_name]
        dataset = {context[i]:self.motion_word_dataset[context[i]] for i in range (len(context))}
        
        #-- Retrieve the positive and negative
        positive_values, positive_names, negative_values, negative_names = retrieve_top_k(self.motion_word_dataset[anchor_name],
                                            anchor_name, dataset, k, similarity_function)
        
        #--- Deleting the retrieved values for positive and negative samples
        del positive_values
        del negative_values
        #---

        return positive_names, negative_names

    

    def compute_training_dataset(self,similarity_function, k = 5 ):
        """This function computes the training dataset.
            
        Parameters
        ----------
        similarity_function : the similarity function to use
        
        k : int
            k is the number of closest or farest motion words to retrieve.
            
        
        Returns
        -------
        None.
        
        """
        start = time.time()
        
        print("\n","-"*20,"\n The positive-negative dataset computation has started :")
        
        output = dict()
        
        self.path_to_training_dataset = f"./../data/datasets/{self.mode+'ing'}/{self.mode+'ing_dataset_'+str(k)+'_Noisy_'+str(self.include_noisy)+'_'+similarity_function.__name__}"
        num_motion_words =len(list(self.motion_word_dataset.keys()))
        
        #--Iterating through the motion word dataset
        for indx, anchor_name in enumerate(list(self.motion_word_dataset.keys())):
            
            #--Fetching the positive and negative examples
            output.update({anchor_name : self.get_positive_and_negative(anchor_name,k,similarity_function)})
            
            if indx % 100 == 0 :
                progress = 100*(indx+1)/num_motion_words
                print(f"The computation is at : {progress:.2}%.")
        
        #--Saving the training dataset    
        with open(self.path_to_training_dataset,"wb") as file :
            pickle.dump(output,file)
            
        print(f"The computation has finished after {round(time.time() - start)} seconds! The file is available at : {self.path_to_training_dataset}")
        
        return None

    def all_motion_word_to_images(self):
        """This function will transform all motion words into images depending on the imaging method(s) selected.The motion words will be stored in self.motion_word_dataset.

        Returns
        -------
        None.

        """
        #---Transforming the motion words dataset into images
        count = 1
        length = len(self.motion_word_dataset)

        for name,value in self.motion_word_dataset.items():
            self.motion_word_dataset[name] = self.timeseries_to_images(value)

            if  count % 100 == 0:
                print(f"The process is at : {100*count/length:0.2f}%")
            count += 1

            
        gc.collect()
            
        return None

    def training_dataset_to_images(self,save_data=False):
        """It will save the training dataset as a dictionary --> {motion_word_name : [positve,negative]}.

        Parameters
        ----------
        save_data : Boolean, optional
            The default is False.
            
        Returns
        -------
        output : dictionary
            Training datset.

        """
        
        
        output = dict()
        data = self.load_training_dataset()
        
        print('-'*20,'Transforming all the motion words into images','-'*20)
        self.all_motion_word_to_images()

        #--Iterating through the dataset and transforming
        print('-'*20,'Transforming training dataset to images','-'*20)
        count = 1
        num_files = len(data)

        for anchor_name, value in data.items():
            positive_images = []
            negative_images = []
            anchor_value = self.motion_word_dataset[anchor_name]

            if(count % 50 == 0):
                print(f"The process is at : {100*count/num_files:.2f}%")

            for i in range(len(value[0])):
                positive_images.append(self.motion_word_dataset[value[0][i]])
                negative_images.append(self.motion_word_dataset[value[1][i]])
            
            count += 1
            
            
            #--Updating the output dictionary
            positives = np.stack(positive_images,axis=0)
            negatives = np.stack(negative_images,axis=0)
            output.update({anchor_name:[anchor_value,positives,negatives]})
            
                        
        #--Saving the data if needed
        if save_data:
            file_name = os.path.basename(self.path_to_training_dataset)
            file_name = os.path.splitext(file_name)[0]
            
            if all([self.MTF, self.GAF, self.recurrence_plot]):
                file_name = file_name + "_3Channel" 
                
            else:
                file_name = file_name + "_1Channel"
                
            file_path = f'./../data/datasets/training/ts_images/{file_name}'
            with open(file_path,'wb') as file:
                pickle.dump(output, file)
                
            print(f'The training time series images have been saved to {file_path}.')
            
            
        return output
        