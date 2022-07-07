# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 09:29:20 2020

@author: user
"""
import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, Dataset
import copy
from sklearn.metrics import accuracy_score
import numpy as np
import time
#ourself libs
from model_initiation import model_init
from data_preprocess import data_set

from FL_base import fedavg, global_train_once, FL_Train, FL_Retrain



def federated_learning_unlearning(init_global_model, client_loaders, test_loader, FL_params):
    
    
    # all_global_models, all_client_models 为保存起来所有的old FL models
    print(5*"#"+"  Federated Learning Start"+5*"#")
    std_time = time.time()
    old_GMs, old_CMs = FL_Train(init_global_model, client_loaders, test_loader, FL_params)
    end_time = time.time()
    time_learn = (std_time - end_time)
    print(5*"#"+"  Federated Learning End"+5*"#")
    
    
    print('\n')
    """4.2 unlearning  a client，Federated Unlearning"""
    print(5*"#"+"  Federated Unlearning Start  "+5*"#")
    std_time = time.time()
    #Set the parameter IF_unlearning =True so that global_train_once skips forgotten users and saves computing time
    FL_params.if_unlearning = True
    #Set the parameter forget_client_IDx to mark the user's IDX that needs to be forgotten
    FL_params.forget_client_idx = 2
    unlearn_GMs = unlearning(old_GMs, old_CMs, client_loaders, test_loader, FL_params)
    end_time = time.time()
    time_unlearn = (std_time - end_time)
    print(5*"#"+"  Federated Unlearning End  "+5*"#")
    
    
    
    print('\n')
    """4.3 unlearning a client，Federated Unlearning without calibration"""
    print(5*"#"+"  Federated Unlearning without Calibration Start  "+5*"#")
    std_time = time.time()
    uncali_unlearn_GMs = unlearning_without_cali(old_GMs, old_CMs, FL_params)
    end_time = time.time()
    time_unlearn_no_cali = (std_time - end_time)
    print(5*"#"+"  Federated Unlearning without Calibration End  "+5*"#")
    
    
    
    # if(FL_params.if_retrain):
    #     print('\n')
    #     print(5*"#"+"  Federated Retraining Start  "+5*"#")
    #     std_time = time.time()
    #     # FL_params.N_client = FL_params.N_client - 1
    #     # client_loaders.pop(FL_params.forget_client_idx)
    #     # retrain_GMs, _ = FL_Train(init_global_model, client_loaders, test_loader, FL_params)
    #     retrain_GMs = FL_Retrain(init_global_model, client_loaders, test_loader, FL_params)
    #     end_time = time.time()
    #     time_retrain = (std_time - end_time)
    #     print(5*"#"+"  Federated Retraining End  "+5*"#")
    # else:
    #     print('\n')
    #     print(5*"#"+"  No Retraining "+5*"#")
    #     retrain_GMs = list()
    
    print(" Learning time consuming = {} secods".format(-time_learn))
    print(" Unlearning time consuming = {} secods".format(-time_unlearn)) 
    print(" Unlearning no Cali time consuming = {} secods".format(-time_unlearn_no_cali)) 
    # print(" Retraining time consuming = {} secods".format(-time_retrain)) 
    
    
    return old_GMs, unlearn_GMs, uncali_unlearn_GMs, old_CMs

    
    
    


def unlearning(old_GMs, old_CMs, client_data_loaders, test_loader, FL_params):
    """
    

    Parameters
    ----------
    old_global_models : list of DNN models
        In standard federated learning, all the global models from each round of training are saved.
    old_client_models : list of local client models
        In standard federated learning, the server collects all user models after each round of training.
    client_data_loaders : list of torch.utils.data.DataLoader
        This can be interpreted as each client user's own data, and each Dataloader corresponds to each user's data
    test_loader : torch.utils.data.DataLoader
        The loader for the test set used for testing
    FL_params : Argment（）
        The parameter class used to set training parameters

    Returns
    -------
    forget_global_model : One DNN model that has the same structure but different parameters with global_moedel
        DESCRIPTION.

    """
    
    
    if(FL_params.if_unlearning == False):
        raise ValueError('FL_params.if_unlearning should be set to True, if you want to unlearning with a certain user')
        
    if(not(FL_params.forget_client_idx in range(FL_params.N_client))):
        raise ValueError('FL_params.forget_client_idx is note assined correctly, forget_client_idx should in {}'.format(range(FL_params.N_client)))
    if(FL_params.unlearn_interval == 0 or FL_params.unlearn_interval >FL_params.global_epoch):
        raise ValueError('FL_params.unlearn_interval should not be 0, or larger than the number of FL_params.global_epoch')
    
    old_global_models = copy.deepcopy(old_GMs)
    old_client_models = copy.deepcopy(old_CMs)
    
    
    forget_client = FL_params.forget_client_idx
    for ii in range(FL_params.global_epoch):
        temp = old_client_models[ii*FL_params.N_client : ii*FL_params.N_client+FL_params.N_client]
        temp.pop(forget_client)#During Unlearn, the model saved by the forgotten user pops up
        old_client_models.append(temp)
    old_client_models = old_client_models[-FL_params.global_epoch:]
        
    
    
    GM_intv = np.arange(0,FL_params.global_epoch+1, FL_params.unlearn_interval, dtype=np.int16())
    CM_intv  = GM_intv -1
    CM_intv = CM_intv[1:]
    
    selected_GMs = [old_global_models[ii] for ii in GM_intv]
    selected_CMs = [old_client_models[jj] for jj in CM_intv]
    
    
    """1. First, complete the model overlay from the initial model to the first round of global train"""
    """
    Since the inIT_model does not contain any information about the forgotten user at the start of the FL training, you just need to overlay the local Model of the other retained users, You can get the Global Model after the first round of global training.
    """
    epoch = 0
    unlearn_global_models = list()
    unlearn_global_models.append(copy.deepcopy(selected_GMs[0]))
    
    new_global_model = fedavg(selected_CMs[epoch])
    unlearn_global_models.append(copy.deepcopy(new_global_model))
    print("Federated Unlearning Global Epoch  = {}".format(epoch))
    
    """2. Then, the first round of global model as a starting point, the model is gradually corrected"""
    """
    In this step, the global Model obtained from the first round of global training was used as the new starting point for training, and a small amount of training was carried out with the data of the reserved user (a small amount means reducing the local epoch, i.e. Reduce the number of local training rounds for each user. The parameter forget_local_epoch_ratio is to control and reduce the number of local training rounds.) Gets the direction of iteration of the local Model parameter for each reserved user, starting with new_global_model.Note that this part of the user model is ref_client_models.
    
    Then we use the old_client_models and old_global_models saved from the unforgotten FL training, and the ref_client_models and new_global_Model that we get when we forget a user,To build the global model for the next round
    
    
    (ref_client_models - new_global_model) / ||ref_client_models - new_global_model||，Indicates the direction of model parameter iteration starting with a new global model that removes a user.Mark the direction as step_direction

    ||old_client_models - old_global_model||，Indicates the step size of the model parameter iteration starting with the old global model with a user removed.Step step_length
    
    So, the final direction of the new reference model is step_direction*step_length + new_global_model。
    """
    """
    Intuitive explanation of this part: Usually in IID data, after the data is sharded, the direction of model parameter iteration is roughly the same.The basic idea is to take full advantage of the client-model parameter data saved in standard FL training, and then, by correcting this part of the parameter, apply it to the iteration of the new global model that forgets a user.
    
    For unforgotten FL:oldGM_t--> oldCM0, oldCM1, oldCM2, oldCM3--> oldGM_t+1
    for unblearning FL：newGM_t-->newCM0, newCM1, newCM2, newCM3--> newGM_t+1
    oldGM_t and newGM_t essentially represents a different starting point for training. However, under the IID data, oldCM and newCM should converge in roughly the same direction.
    Therefore, we get newCM by using newcm-newgm_t as the starting point and training fewer rounds on user data, and then using (newcm-newgm_t)/|| newcm-newgm_t || as the current forgetting setting,
    Direction of model parameter iteration.Take || oldcm-oldgm_t || as the iteration step, and finally use || oldcm-oldgm_t ||*(newcm-newgm_t)/|| newcm-newgm_t |0 |1 for the iteration of the new model.
    FedEraser iterative formula: newGM_t+1 = newGM_t + ||oldCM - oldGM_t||*(newCM - newGM_t)/||newCM - newGM_t||
    
    """
    

    CONST_local_epoch = copy.deepcopy(FL_params.local_epoch)
    FL_params.local_epoch = np.ceil(FL_params.local_epoch*FL_params.forget_local_epoch_ratio)
    FL_params.local_epoch = np.int16(FL_params.local_epoch)

    CONST_global_epoch = copy.deepcopy(FL_params.global_epoch)
    FL_params.global_epoch = CM_intv.shape[0]
    
    
    print('Local Calibration Training epoch = {}'.format(FL_params.local_epoch))
    for epoch in range(FL_params.global_epoch):
        if(epoch == 0):
            continue
        print("Federated Unlearning Global Epoch  = {}".format(epoch))
        global_model = unlearn_global_models[epoch]

        new_client_models  = global_train_once(global_model, client_data_loaders, test_loader, FL_params)

        new_GM = unlearning_step_once(selected_CMs[epoch], new_client_models, selected_GMs[epoch+1], global_model)
        
        unlearn_global_models.append(new_GM)
    FL_params.local_epoch = CONST_local_epoch
    FL_params.global_epoch = CONST_global_epoch
    return unlearn_global_models
    
def unlearning_step_once(old_client_models, new_client_models, global_model_before_forget, global_model_after_forget):
    """
    

    Parameters
    ----------
    old_client_models : list of DNN models
        When there is no choice to forget (if_forget=False), use the normal continuous learning training to get each user's local model.The old_client_models do not contain models of users that are forgotten.
        Models that require forgotten users are not discarded in the Forget function
    ref_client_models : list of DNN models
        When choosing to forget (if_forget=True), train with the same Settings as before, except that the local epoch needs to be reduced, other parameters are set in the same way.
        Using the above training Settings, the new global model is taken as the starting point and the reference model is trained.The function of the reference model is to identify the direction of model parameter iteration starting from the new global model
        
    global_model_before_forget : The old global model
        DESCRIPTION.
    global_model_after_forget : The New global model
        DESCRIPTION.

    Returns
    -------
    return_global_model : After one iteration, the new global model under the forgetting setting

    """
    old_param_update = dict()#Model Params： oldCM - oldGM_t
    new_param_update = dict()#Model Params： newCM - newGM_t
    
    new_global_model_state = global_model_after_forget.state_dict()#newGM_t
    
    return_model_state = dict()#newGM_t + ||oldCM - oldGM_t||*(newCM - newGM_t)/||newCM - newGM_t||
    
    assert len(old_client_models) == len(new_client_models)
    
    for layer in global_model_before_forget.state_dict().keys():
        old_param_update[layer] = 0*global_model_before_forget.state_dict()[layer]
        new_param_update[layer] = 0*global_model_before_forget.state_dict()[layer]
        
        return_model_state[layer] = 0*global_model_before_forget.state_dict()[layer]
        
        for ii in range(len(new_client_models)):
            old_param_update[layer] += old_client_models[ii].state_dict()[layer]
            new_param_update[layer] += new_client_models[ii].state_dict()[layer]
        old_param_update[layer] /= (ii+1)#Model Params： oldCM
        new_param_update[layer] /= (ii+1)#Model Params： newCM
        
        old_param_update[layer] = old_param_update[layer] - global_model_before_forget.state_dict()[layer]#参数： oldCM - oldGM_t
        new_param_update[layer] = new_param_update[layer] - global_model_after_forget.state_dict()[layer]#参数： newCM - newGM_t
        
        step_length = torch.norm(old_param_update[layer])#||oldCM - oldGM_t||
        step_direction = new_param_update[layer]/torch.norm(new_param_update[layer])#(newCM - newGM_t)/||newCM - newGM_t||
        
        return_model_state[layer] = new_global_model_state[layer] + step_length*step_direction
    
    
    
    
    
    return_global_model = copy.deepcopy(global_model_after_forget)
    
    return_global_model.load_state_dict(return_model_state)
    
    return return_global_model


    # return forget_global_model
    
    
def unlearning_without_cali(old_global_models, old_client_models, FL_params):
    """
    

    Parameters
    ----------
    old_client_models : list of DNN models
        All user local update models are saved during the federated learning and training process that is not forgotten.
    FL_params : parameters
        All parameters in federated learning and federated forgetting learning

    Returns
    -------
    global_models : List of DNN models
        In each update round, the client model of the user who needs to be forgotten is removed, and the parameters of other users' client models are directly superimposing to form the new Global Model of each round

    """
    """
    The basic process is as follows：For unforgotten FL:oldGM_t--> oldCM0, oldCM1, oldCM2, oldCM3--> oldGM_t+1
                 For unlearning FL：newGM_t-->The parameters of oldCM and oldGM were directly leveraged to update global model--> newGM_t+1
    The update process is as follows：newGM_t+1 = (oldCM - oldGM_t) + newGM_t
    """
    if(FL_params.if_unlearning == False):
        raise ValueError('FL_params.if_unlearning should be set to True, if you want to unlearning with a certain user')
        
    if(not(FL_params.forget_client_idx in range(FL_params.N_client))):
        raise ValueError('FL_params.forget_client_idx is note assined correctly, forget_client_idx should in {}'.format(range(FL_params.N_client)))
    forget_client = FL_params.forget_client_idx
    
    
    for ii in range(FL_params.global_epoch):
        temp = old_client_models[ii*FL_params.N_client : ii*FL_params.N_client+FL_params.N_client]
        temp.pop(forget_client)
        old_client_models.append(temp)
    old_client_models = old_client_models[-FL_params.global_epoch:]
    
    uncali_global_models = list()
    uncali_global_models.append(copy.deepcopy(old_global_models[0]))
    epoch = 0
    uncali_global_model = fedavg(old_client_models[epoch])
    uncali_global_models.append(copy.deepcopy(uncali_global_model))
    print("Federated Unlearning without Clibration Global Epoch  = {}".format(epoch))
    
    """
    new_GM_t+1 = newGM_t + (oldCM_t - oldGM_t)
    
    For standard federated learning:oldGM_t --> oldCM_t --> oldGM_t+1
    For accumulatring:    newGM_t --> (oldCM_t - oldGM_t) --> oldGM_t+1
    For uncalibrated federated forgotten learning, the parameter update of the unforgotten user in standard federated learning is used to directly overlay the new global model to obtain the next round of new global model.
    """
    old_param_update = dict()#(oldCM_t - oldGM_t)
    return_model_state = dict()#newGM_t+1
    
    for epoch in range(FL_params.global_epoch):
        if(epoch == 0):
            continue
        print("Federated Unlearning Global Epoch  = {}".format(epoch))
        
        current_global_model = uncali_global_models[epoch]#newGM_t
        current_client_models = old_client_models[epoch]#oldCM_t
        old_global_model = old_global_models[epoch]#oldGM_t
        # global_model_before_forget = old_global_models[epoch]#old_GM_t
        
        
        for layer in current_global_model.state_dict().keys():
            #State variable initialization
            old_param_update[layer] = 0*current_global_model.state_dict()[layer]
            return_model_state[layer] = 0*current_global_model.state_dict()[layer]
            
            for ii in range(len(current_client_models)):
                old_param_update[layer] += current_client_models[ii].state_dict()[layer]
            old_param_update[layer] /= (ii+1)# oldCM_t
            
            old_param_update[layer] = old_param_update[layer] - old_global_model.state_dict()[layer]#参数： oldCM_t - oldGM_t

            return_model_state[layer] = current_global_model.state_dict()[layer] + old_param_update[layer]#newGM_t + (oldCM_t - oldGM_t)
            
        return_global_model = copy.deepcopy(old_global_models[0])
        return_global_model.load_state_dict(return_model_state)
            
        uncali_global_models.append(return_global_model)

    return uncali_global_models
    
    
    
    
    

    
    



























