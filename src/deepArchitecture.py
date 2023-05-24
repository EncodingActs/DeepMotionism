from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import models, transforms
import gc

def transform_data(input_size,is_ts_images=False):
    """This function will transform the input data to meet the requirements of the neural network. It performs : normalization, random erasing , random resize cropping.The normalization uses hardcoded values given at https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html .
    

    Parameters
    ----------
    input_size : int
        The desired input size of the network. The value to be included will be given by the function 'initialize_model'.
        
    is_ts_images : boolean, optional
        This states whether the data to be transformed consists of images or they have 3 channels. The default is False.

    Returns
    -------
    transformer : function
        This function will transform the data to fit the network requirement.


    """

    if is_ts_images :
        transformer = transforms.Compose( [transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0), ratio=(0.8, 1.33)),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                           transforms.RandomErasing(p=0.1)])
    else:
        transformer = transforms.Compose( [transforms.RandomCrop(input_size, padding=[0,0,0,0], pad_if_needed=True, fill=0, padding_mode='constant'),
                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                     transforms.RandomErasing(p=0.1)])
        
    return transformer 
       

def set_parameter_requires_grad(model, feature_extracting):
    """This function will freeze the parameters of model if feature_extracting == True.
    
    Parameters
    ----------
    model : pytorch neural network object
        
    feature_extracting : Boolean
        This should be 'True' if we just want to train the last fully connected layer.

    Returns
    -------
    None.

    """
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            

def initialize_model(model_name='resnet18', num_classes=1000, feature_extract=True, use_pretrained=True):
    """
    

    Parameters
    ----------
    model_name : string, optional
        It can be "resnet18","resnext50" or "inceptionV3". The default is 'resnet18'.
        
    num_classes : int, optional
        It represents the desired output embedding dimension. The default is 1000.
        
    feature_extract : Boolean, optional
        Tells if we will only be training on the last fully connected layer or not. The default is True.
        
    use_pretrained : Boolean, optional
        If True, a pretrained model will be loaded for transfer learning purposes. The default is True.

    Returns
    -------
    model_ft : torchvision neural network
        
    input_size : int
        The adequate input size for the desired network. The networks take as inputs of shape (3,input_size,input_size).

    """
    
    # variables a
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        #---Resnet18
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    
    elif model_name == "inceptionV3":
        #--Inception v3
        
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        
        #--Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        
        #--Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299
    
    elif model_name == 'resnext50':
        #--ResNeXT50
        
        model_ft = models.resnext50_32x4d(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        
        input_size = 224
        
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


def build_batch(dataloader, data_object, batch_size, indx, mode = 'train', do_ts_images=False):
    """This function will build the training or evaluation batch provided a dataloader.
    

    Parameters
    ----------
    dataloader : dictionary
        It is the training dataset.
        
    data_object : data_preparation object
        This object is an instance of the data_preparation class.

    batch_size : int
        
    indx : int
        Starting indice of the batch.
    
    mode : string
        It should be 'train' or 'eval' for training and evaluation respectively.

    Returns
    -------
    triplet_size : int
        The size of the triplets. It can be 5 or 10 and it will mean that we consider 5 or 10 positive and negative samples for training.
    
    anchors : numpy array of dimension  4
        The anchors are stacked and look like (triplet_size,d,m,n) where d is the number of channels.
    
    positives : numpy array of dimension  4
        The positive samples are stacked and look like (triplet_size*batch_size,d,m,n) where d is the number of channels.
    
    negatives : numpy array of dimension  4
        The negative samples are stacked and look like (triplet_size*batch_size,d,m,n) where d is the number of channels.

    """
    
    assert mode in ['eval','train'], "The mode has to be 'train' or 'val' ! Chech the entries to the function 'build_batch'."

      
    #--Training mode
    if mode=='train' :
    
        #--
        anchors =   []
        positives = []
        negatives = []
        #--
        positive_names =  []
        negative_names =  []
        
        #------------------------------Building batch--------------------------------------
        
        for i, anchor_name in enumerate(list(dataloader.keys())):
            
            #--Getting the triplet size
            if i == 0:
                triplet_size = len(dataloader[anchor_name][0])
        	
            #--Fetching the data   
            if  (i >= indx and i < batch_size+indx):
                positive_names = positive_names + dataloader[anchor_name][0]
                negative_names = negative_names + dataloader[anchor_name][1]
                
                #--Transforming the anchor to images if required
                if do_ts_images: 
                    dummy = data_object.timeseries_to_images(data_object.motion_word_dataset[anchor_name])
                    anchors.append(dummy)
                
                else :     
                    anchors.append(data_object.motion_word_dataset[anchor_name][None,:,:])
                        
            if  i >= batch_size+i: break #-- Breaking when we get out of the mini batch range
        
        
        #--Transforming positive and negative samples ot images if required.
        if do_ts_images:
            for i in range(len(positive_names)):
                positive = data_object.timeseries_to_images(data_object.motion_word_dataset[positive_names[i]])
                negative = data_object.timeseries_to_images(data_object.motion_word_dataset[negative_names[i]])
                positives.append(positive)
                negatives.append(negative) 
         
        #--In this case, the data is 2D, we need to add another axis at the begining i.e shape will be (1,m,n).
        else :
            for i in range(len(positive_names)):
                positive = data_object.motion_word_dataset[positive_names[i]]
                negative = data_object.motion_word_dataset[negative_names[i]]
                positives.append(positive[None,:,:])
                negatives.append(negative[None,:,:])   
          
                
        anchors   = np.stack(anchors,axis=0)
        positives = np.stack(positives,axis=0)
        negatives = np.stack(negatives,axis=0)
        
        #--Free up some memory
        gc.collect()
        return triplet_size, anchors, positives, negatives
    
    #--- eval mode
    if mode == 'eval':
        data = []
        names = []
        #--Building batch
        for i,motion_name in enumerate(list(dataloader.keys())):
                
            if  (i >= indx and i < batch_size+indx):
                
                if  do_ts_images:
                    ts_images = data_object.timeseries_to_images(data_object.motion_word_dataset[motion_name])
                    data.append(ts_images)
                    names.append(motion_name)
                else:
                    data.append(data_object.motion_word_dataset[motion_name][None,:,:])
                    names.append(motion_name)
                
            if  i >= batch_size+i: break #-- Breaking when we get out of the mini batch range

                
        return np.stack(data,axis=0), names
    
    
            
def train_model(model, model_name, dataloader, data_object, transformer,
                learning_rate=0.001,
                feature_extract=False,
                criterion=nn.TripletMarginLoss(margin=0.2),
                batch_size=15,
                num_epochs=15,
                do_ts_images=False,
                use_multi_gpus=False):
    """
    

    Parameters
    ----------
    model : pytorch neural network object
                
    dataloader : dictionary
        It is the training dataset.
        
    data_object : data_preparation object
        This object is an instance of the data_preparation class.
        
    transformer : function
        This function will transform the data to fit the network requirement.
        
    learning_rate : float, optional
        The default is 0.001.
        
    feature_extract : Boolean, optional
        Tells whether we're only training the last fully connected layer or not. The default is True.
        
    criterion : loss function, optional
        We train using a triplet loss function ie nn.TripletMarginLoss(margin=0.2).
        
    batch_size : int, optional
        The default is 15.
        
    num_epochs : int, optional
        The default is 15.
        
    do_ts_images : Boolean, optional
        It says whether the input should be transformed into images or not. The default is False.

    Returns
    -------
    model : trained pytorch neural network object
    loss_history : array of floats
        All losses for each epoch

    """
    
    loss_history = []
    model = model.double()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    #--Setting the flag is_inception
    if model_name == 'inceptionV3' : 
        is_inception =True
        use_multi_gpus = False #-Do not use multiple gpus with inception
    else : 
        is_inception = False
    
    #--Doing parallel training with requirements are met
    if use_multi_gpus and  torch.cuda.device_count() > 1:
        print(f'\n\n We are using {torch.cuda.device_count()} GPUs.')
        model = nn.DataParallel(model)
    
    #--Model to device
    model.to(device)
    
    #---Get params to update
    params_to_update = model.parameters()
    
    if feature_extract:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

    #---Set model to training mode
    model.train()
    optimizer = optim.Adam(params_to_update, lr = learning_rate)
    
    for epoch in range(num_epochs):
        print('\n Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 50)

        count = 0
        
        #--Iterate over data using a mini_batch.
        for b in range(0,len(dataloader), batch_size):
            
            triplet_size, anchors, positives, negatives = build_batch(dataloader = dataloader,data_object = data_object,
                                                                      batch_size = batch_size , indx = b,  mode ='train',
                                                                      do_ts_images = do_ts_images)
            
            #print(triplet_size,anchors.shape,positives.shape,negatives.shape)
                          
            #--If len(dataloader) is not divisible by batch_size,
            #--there will be an issue when b reaches len(dataloader) so we reset the batch_size
            batch_size = anchors.shape[0] 
        
            #--- Expanding across channels if it has one channel i.e anchors.shape[1]==1 or do_ts_images==False
            if not do_ts_images or anchors.shape[1]==1 :

                anchors  =  torch.tensor(anchors).clone().detach().expand(-1,3,-1,-1)
                positives = torch.tensor(positives).clone().detach().expand(-1,3,-1,-1)
                negatives = torch.tensor(negatives).clone().detach().expand(-1,3,-1,-1)
                
            else:
                anchors  =  torch.tensor(anchors).clone().detach()
                positives = torch.tensor(positives).clone().detach()
                negatives = torch.tensor(negatives).clone().detach()
            
            #--Getting transformed data
            anchors   =  transformer(anchors)
            positives =  transformer(positives)
            negatives =  transformer(negatives)

            # zero the parameter gradients
            optimizer.zero_grad()

            #--------------------------forward pass-----------------------------------------
            inputs = torch.cat((anchors, positives, negatives), dim=0).double()
            inputs = inputs.to(device)
            
            if is_inception :
                if use_multi_gpus and  torch.cuda.device_count() > 1:
                    outputs, aux_outputs = model(inputs).values()
                    
                else :
                    outputs, aux_outputs = model(inputs)

                #----Repeating the anchor value to have triplets of same size (anchor,positive,negative)
                anchor_embedding = torch.repeat_interleave(aux_outputs[:batch_size],
                                                           repeats=triplet_size,dim=0)
                
                positives_embedding = aux_outputs[batch_size:(triplet_size+1)*batch_size]
                negatives_embedding = aux_outputs[-triplet_size*batch_size:]
                loss1 = criterion(anchor_embedding,positives_embedding,negatives_embedding)   
                
                #-----
                anchor_embedding = torch.repeat_interleave(outputs[:batch_size],
                                                           repeats=triplet_size,dim=0)
                
                positives_embedding = outputs[batch_size:(triplet_size+1)*batch_size]
                negatives_embedding = outputs[-triplet_size*batch_size:]
                loss2 = criterion(anchor_embedding,positives_embedding,negatives_embedding)   
                #-----
                
                loss = loss1 + 0.4*loss2

            else:
                outputs = model(inputs)
                
                #--
                anchor_embedding = torch.repeat_interleave(outputs[:batch_size],
                                                           repeats=triplet_size,dim=0)
                
                positives_embedding = outputs[batch_size:(triplet_size+1)*batch_size]
                negatives_embedding = outputs[-triplet_size*batch_size:]

                #print(anchor_embedding.shape,positives_embedding.shape,negatives_embedding.shape)

                loss = criterion(anchor_embedding,positives_embedding,negatives_embedding)  
            
            #--appending the loss
            loss_history.append(loss.item())
            
            #--backward pass + optimize
            loss.backward()
            optimizer.step()
            
            #---------------------- Outputting progress ---------------
            count += batch_size
            if count % 100*batch_size == 0 :
            	print(f"\n The process is at : {100*count/len(dataloader):.2f}%")
            
            #--Collecting garbage to free up memory.
            gc.collect()
            
        #--------------- End epoch ----------------------------------------   
        #--Saving a checkpoint  state dictionary
        path = f'./../data/models/model_{model_name}_RecordingParam_tsImages_{do_ts_images}_includeNoisy_{data_object.include_noisy}.pth'
        if use_multi_gpus and torch.cuda.device_count() > 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                            }, path)
             
        else :
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                            }, path)   
             
    return model, loss_history
































