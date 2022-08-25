import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_sparse

##########################################################################
# close form solution utils
def remove_data(X_prime, Y_prime, XtX_inv, W):
    # (X_prime, Y_prime): data to remove;
    # W_orginal: params before remove;
    
    num_data = X_prime.shape[0]
    
    A = XtX_inv@X_prime.T
    B = torch.linalg.inv(torch.eye(num_data).to(X_prime) - X_prime@XtX_inv@X_prime.T)
    C = Y_prime - X_prime@W
    D = X_prime@XtX_inv
    
    return XtX_inv + A@B@D, W - A@B@C

def add_data(X_prime, Y_prime, XtX_inv, W):
    # (X_prime, Y_prime): data to add;
    # W_orginal: params before add data;
    
    num_data = X_prime.shape[0]
    
    A = XtX_inv@X_prime.T
    B = torch.linalg.inv(torch.eye(num_data).to(X_prime) + X_prime@XtX_inv@X_prime.T)
    C = Y_prime - X_prime@W
    D = X_prime@XtX_inv
    
    return XtX_inv - A@B@D, W + A@B@C

def find_w(X, Y, lam=0):    
    try:
        Xtx_inv = torch.linalg.inv(X.T@X + lam*torch.eye(X.size(1)))
        Xty = X.T@Y
        W = Xtx_inv@Xty
    except:
        try:
            print('Feat matrix is not inversible, add random noise')
            X = X + torch.randn_like(X)*1e-5
            Xtx_inv = torch.linalg.inv(X.T@X + lam*torch.eye(X.size(1)))
            Xty = X.T@Y
            W = Xtx_inv@Xty
        except:
            print('Feat matrix is not inversible, use psudo inverse')
            Xtx_inv = torch.linalg.pinv(X.T@X + lam*torch.eye(X.size(1)))
            Xty = X.T@Y
            W = Xtx_inv@Xty            
    return Xtx_inv, W

def predict(X, W):
    return X@W
