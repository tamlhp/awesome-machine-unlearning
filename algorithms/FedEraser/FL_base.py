# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 22:15:13 2020

@author: jojo
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


def FL_Train(init_global_model, client_data_loaders, test_loader, FL_params):
    if(FL_params.if_retrain == True):
        raise ValueError('FL_params.if_retrain should be set to False, if you want to train, not retrain FL model')
    if(FL_params.if_unlearning == True):
        raise ValueError('FL_params.if_unlearning should be set to False, if you want to train, not unlearning FL model')
    
    # if(FL_params._save_all_models == False):
    #     # print("FL Training without Forgetting...")
    #     global_model = init_global_model
    #     for epoch in range(FL_params.global_epoch):
    #         client_models = global_train_once(global_model, client_data_loaders, test_loader, FL_params)
    #         #IMPORTANT：这里有一点要注意，就是global_train_once在训练过程中，是直接在input的client_models上进行训练，因此output的client_models与input的client_models是同一组模型，只不过input没有经过训练，而output经过了训练。
    # #   IMPORTANT：因此，为了实现Federated unlearning，我们需要在global train之前就将client——models中的模型进行保存。可以使用deepcopy，或者硬盘io方式。
    #         global_model = fedavg(client_models)
    #         print(30*'^')
    #         print("Global training epoch = {}".format(epoch))
    #         # test(global_model, test_loader)
    #         print(30*'v')
        
    #     return global_model
    # elif (FL_params._save_all_models == True):
        # print("FL Training with Forgetting...")
    all_global_models = list()
    all_client_models = list()
    global_model = init_global_model
    
    all_global_models.append(copy.deepcopy(global_model))
    
    for epoch in range(FL_params.global_epoch):
        client_models = global_train_once(global_model, client_data_loaders, test_loader, FL_params)
        #IMPORTANT：这里有一点要注意，就是global_train_once在训练过程中，是直接在input的client_models上进行训练，因此output的client_models与input的client_models是同一组模型，只不过input没有经过训练，而output经过了训练。
   #IMPORTANT：因此，为了实现Federated unlearning，我们需要在global train之前就将client——models中的模型进行保存。可以使用deepcopy，或者硬盘io方式。
    #IMPORTANT: It is IMPORTANT to note here that global_train_once is trained directly on the input client_models during training, so the output's client_models are the same set of models as the input's client_models, except that the input is untrained while the output is trained.
    #Therefore, in order to implement Federated Unlearning, we need to save the models in Client -- Models before global Train.You can use DeepCopy, or hard disk IO.
        all_client_models += client_models
        global_model = fedavg(client_models)
        # print(30*'^')
        print("Global Federated Learning epoch = {}".format(epoch))
        # test(global_model, test_loader)
        # print(30*'v')
        # print(len(all_client_models))
        all_global_models.append(copy.deepcopy(global_model))
        
    return all_global_models, all_client_models
        
        


def FL_Retrain(init_global_model, client_data_loaders, test_loader, FL_params):
    if(FL_params.if_retrain == False):
        raise ValueError('FL_params.if_retrain should be set to True, if you want to retrain FL model')
    if(FL_params.forget_client_idx not in range(FL_params.N_client)):
        raise ValueError('FL_params.forget_client_idx should be in [{}], if you want to use standard FL train with forget the certain client dataset.'.format(range(FL_params.N_client)))
    # forget_idx= FL_params.forget_idx
    print('\n')
    print(5*"#"+"  Federated Retraining Start  "+5*"#")
    # std_time = time.time()
    print("Federated Retrain with Forget Client NO.{}".format(FL_params.forget_client_idx))
    retrain_GMs = list()
    all_client_models = list()
    retrain_GMs.append(copy.deepcopy(init_global_model))
    global_model = init_global_model
    for epoch in range(FL_params.global_epoch):
        client_models = global_train_once(global_model, client_data_loaders, test_loader, FL_params)
        #IMPORTANT：这里有一点要注意，就是global_train_once在训练过程中，是直接在input的client_models上进行训练，因此output的client_models与input的client_models是同一组模型，只不过input没有经过训练，而output经过了训练。
        #IMPORTANT：这里有一点要注意，就是global_train_once在训练过程中，是直接在input的client_models上进行训练，因此output的client_models与input的client_models是同一组模型，只不过input没有经过训练，而output经过了训练。：It is important to note that global_train_once is trained directly on the input client_models during training, so the output's client_models are the same set of models as the input's client_models, except that the input is untrained while the output is trained.
#   IMPORTANT：因此，为了实现Federated unlearning，我们需要在global train之前就将client——models中的模型进行保存。可以使用deepcopy，或者硬盘io方式。
#IMPORTANT: Therefore, in order to implement Federated Unlearning, we need to save the models in Client -- Models before global Train.You can use DeepCopy, or hard disk IO.
        global_model = fedavg(client_models)
        # print(30*'^')
        print("Global Retraining epoch = {}".format(epoch))
        # test(global_model, test_loader)
        # print(30*'v')
        retrain_GMs.append(copy.deepcopy(global_model))
        
        all_client_models += client_models
    # end_time = time.time()
    print(5*"#"+"  Federated Retraining End  "+5*"#")
    return retrain_GMs
    
    
    

"""
Function：
For the global round of training, the data and optimizer of each global_ModelT is used. The global model of the previous round is the initial point and the training begins.
NOTE:The global model inputed is the global model for the previous round
    The output client_Models is the model that each user trained separately.
"""
#training sub function    
def global_train_once(global_model, client_data_loaders, test_loader, FL_params):
    #使用每个client的模型、优化器、数据，以client_models为训练初始模型，使用client用户本地的数据和优化器，更新得到upodate——client_models
    #Note：需要注意的一点是，global_train_once只是在全局上对模型的参数进行一次更新
    #Using the model, optimizer, and data of each client, training the initial model with client_models, updating the UPODate -- client_models using the client user's local data and optimizer
    #Note: It is important to Note that global_train_once is only a global update to the parameters of the model
    # update_client_models = list()
    device = torch.device("cuda" if FL_params.use_gpu*FL_params.cuda_state else "cpu")
    device_cpu = torch.device("cpu")
    
        
    client_models = []
    client_sgds = []
    for ii in range(FL_params.N_client):
        client_models.append(copy.deepcopy(global_model))
        client_sgds.append(optim.SGD(client_models[ii].parameters(), lr=FL_params.local_lr, momentum=0.9))

    
    
    for client_idx in range(FL_params.N_client):
        if(((FL_params.if_retrain) and (FL_params.forget_client_idx == client_idx)) or ((FL_params.if_unlearning) and (FL_params.forget_client_idx == client_idx))):
            
            continue
        # if((FL_params.if_unlearning) and (FL_params.forget_client_idx == client_idx)):
        #     continue
        # print(30*'-')
        # print("Now training Client No.{}  ".format(client_idx))
        model = client_models[client_idx]
        optimizer = client_sgds[client_idx]
        
        
        model.to(device)
        model.train()
        
        #local training
        for local_epoch in range(FL_params.local_epoch):
            for batch_idx, (data, target) in enumerate(client_data_loaders[client_idx]):
                data = data.to(device)
                target = target.to(device)
                
                optimizer.zero_grad()
                pred = model(data)
                criteria = nn.CrossEntropyLoss()
                loss = criteria(pred, target)
                loss.backward()
                optimizer.step()
                
            if(FL_params.train_with_test):
                print("Local Client No. {}, Local Epoch: {}".format(client_idx, local_epoch))
                test(model, test_loader)
        
        
        # if(FL_params.use_gpu*FL_params.cuda_state):
        model.to(device_cpu)
        client_models[client_idx] = model
        
    if(((FL_params.if_retrain) and (FL_params.forget_client_idx == client_idx))):
        #只有retrian 需要丢弃client 模型；如果不是在retrain的话，就不需要丢弃模型
        #Only retrian needs to discard the Client model;If it's not in Retrain, there's no need to discard the model
        client_models.pop(FL_params.forget_client_idx)
        return client_models
    elif((FL_params.if_unlearning) and (FL_params.forget_client_idx in range(FL_params.N_client))):
        client_models.pop(FL_params.forget_client_idx)
        return client_models
    else:
        return client_models
            

"""
Function：
Test the performance of the model on the test set
"""
def test(model, test_loader):
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            criteria = nn.CrossEntropyLoss()
            test_loss += criteria(output, target) # sum up batch loss
            
            pred = torch.argmax(output,axis=1)
            test_acc += accuracy_score(pred,target)
        
    test_loss /= len(test_loader.dataset)
    test_acc = test_acc/np.ceil(len(test_loader.dataset)/test_loader.batch_size)
    print('Test set: Average loss: {:.8f}'.format(test_loss))         
    print('Test set: Average acc:  {:.4f}'.format(test_acc))    
    return (test_loss, test_acc)
    
    
"""
Function：
FedAvg
"""    
def fedavg(local_models):
# def fedavg(local_models, local_model_weights=None):
    """
    Parameters
    ----------
    local_models : list of local models
        DESCRIPTION.In federated learning, with the global_model as the initial model, each user uses a collection of local models updated with their local data.
    local_model_weights : tensor or array
        DESCRIPTION. The weight of each local model is usually related to the accuracy rate and number of data of the local model.(Bypass)

    Returns
    -------
    update_global_model
        Updated global model using fedavg algorithm
    """
    # N = len(local_models)
    # new_global_model = copy.deepcopy(local_models[0])
    # print(len(local_models))
    global_model = copy.deepcopy(local_models[0])
    avg_state_dict = global_model.state_dict()
    
    local_state_dicts = list()
    for model in local_models:
        local_state_dicts.append(model.state_dict())
    
    
    for layer in avg_state_dict.keys():
        avg_state_dict[layer] *= 0 
        for client_idx in range(len(local_models)):
            avg_state_dict[layer] += local_state_dicts[client_idx][layer]
        avg_state_dict[layer] /= len(local_models)
    
    global_model.load_state_dict(avg_state_dict)
    return global_model 