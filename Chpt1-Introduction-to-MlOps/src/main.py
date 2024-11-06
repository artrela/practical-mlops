'''
General functions to provide testing for workflow learning
'''

import torch

def scale_tensor(x: torch.Tensor, a: float)->torch.Tensor:
    '''scale tensor by a factor'''
    return x * a

def transpose(x: torch.Tensor)->torch.Tensor:
    '''as function states'''
    return x.T

def relu(x: torch.Tensor)->torch.Tensor:
    '''apply torch relu activation'''
    return torch.relu(x)
