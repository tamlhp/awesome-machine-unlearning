# FedEraser (Federated Unlearning)
## About The Project
FedEraser allows a federated client to quit the Federated Learning system and eliminate the influences of his or her data on the global model trained by the standard Federated Learning. 

## Presented Unlearning Methods
This code provides three Federated Unlearning methods:
- **Method1: FedEraser (Federated Unlearning, which is named FedEraser in our paper)**.The parameters of the client model saved by the not forgotten user in the standard FL training process were taken as the step size of the global model iteration, and then the new global model was taken as the starting point for the training, and a small amount of training was carried out, and the parameters of the new Client model were taken as the direction of the iteration of the new global model.Iterate over the new global model using the step \times direction.


- **Method2: Unlearning without Cali (Directly Accumulating)**.The local model of each round saved by the standard federated learning when not forgotten is directly used, the client model of the forgotten user is removed, and the client Models of other users are directly aggregated to obtain the new global model.

- **Method3: Retrain (Federated Retraining).**Retraining without user data that needs to be forgotten.

Besides, this code also provides the function of membership inference attacks, to evaluate whether the unlearned client's data has been unlearned by the model. 


The main function is contained in Fed_Unlearn_main.py. 


## Getting Started
### Prerequisites
**Gradeint-Leaks** requires the following packages: 
- Python 3.8.3
- Pytorch 1.6
- Sklearn 0.23.1
- Numpy 1.18
- Scipy 1.5


### File Structure 
```
Gradient-Leaks
├── datasets
│   ├── Adult
│   ├──  Bank
│   ├──  Purchase
│   └── MNIST
├── data_preprocessing.py
├── Fed_Unlearn_base.py
├── Fed_Unlearn_main.py
├── FL_base.py
├── membership_inference.py
└── model_initiation.py
```
There are several parts of the code:
- datasets folder: This folder contains the training and testing data for the target model.  In order to reduce the memory space, we just list the  links to theset dataset here. 
   -- Adult: https://archive.ics.uci.edu/ml/datasets/Adult
   -- Bank: https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
   -- Purchase: https://github.com/privacytrustlab/datasets/blob/master/dataset_purchase.tgz
   -- MNIST: http://yann.lecun.com/exdb/mnist/
- data_preprocessing.py: This file contains the preprocessing of the raw data in datasets folder.
- Fed_Unlearn_base.py: This file contains the base function of FedEraser, which corresponds to **Section III** in our paper.
- ***Fed_Unlearn_main.py: The main function of Gradient-Leaks.***
- FL_base.py: This file contains the function of Federated Learning, such as FedAvg, Local-Training. 
- membership_inference.py: This file contains the training process of the membership feature extraction model and the  attack model. 
- model_initiation.py: This file contains the structure of the global model corresponding to each dataset that we used in our experiment.  

## Parameter Setting of FedEraser
The attack settings of Gradient-Leaks are determined in the parameter **FL_params** in **Fed_Unlearn_main.py**. 
- ***Federated Learning Model Training Settings***
-- FL_params.N_total_client: the number of federated clients 
-- FL_params.N_client: 
-- FL_params.data_name: select the dataset 
-- FL_params.global_epoch: the number of global training  epoch in federated learning 
-- FL_params.local_epoch: the number of client local training   epoch in federated learning 
-- FL_params.local_batch_size: the local batch size of the client's local training 
-- FL_params.local_lr: the local learning rate of the client's local training 
-- FL_params.test_batch_size: the testing  batch size of the client's local training 
-- FL_params.seed: random seed 
-- FL_params.save_all_model: If=True, saving all the updates and intermediate model. 
-- FL_params.cuda_state: check whether gpu is available (torch.cuda.is_available())
-- FL_params.use_gpu: controlling whether to use gpu 
-- FL_params.train_with_test: controlling whether testings are performed at the end of each global round of training


- ***Federated Unlearning Settings***
-- FL_params.unlearn_interval: Used to control how many rounds the model parameters are saved. $1$ represents the parameter saved once per round. (corresponding to N_itv in our paper)
	-- FL_params.forget_local_epoch_ratio: When a user is selected to be forgotten, other users need to train several rounds of on-line training in their respective data sets to obtain the general direction of model convergence in order to provide the general direction of model convergence. $forget_local_epoch_ratio \times local_epoch is$ the number of rounds of local training when we need to get the convergence direction of each local model
-- FL_params.forget_client_idx = 2 #If want to forget, change None to the client index                 
-- FL_params.if_retrain: If set to True, the global model is retrained using the FL-Retrain function, and data corresponding to the user for the forget_client_IDx number is discarded. If this parameter is set to False, only the global model after the final training is completed is output
-- FL_params.if_unlearning: If set to False, the global_train_once function will not skip users that need to be forgotten;If set to True, global_train_once skips the forgotten user during training



## Execute Gradient-Leaks
*** Run Fed_Unlearn_main.py.  ***




