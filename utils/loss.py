import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def true_metric_loss(true, no_of_classes, scale=1):
    batch_size = true.size(0)
    true = true.view(batch_size,1)
    true_labels = torch.cuda.LongTensor(true).repeat(1, no_of_classes).float()
    class_labels = torch.arange(no_of_classes).float().cuda()
    phi = (scale * torch.abs(class_labels - true_labels)).cuda()
    y = nn.Softmax(dim=1)(-phi)
    return y

def loss_function(output, labels, loss_type, expt_type, scale):
    if loss_type == 'oe':
        targets = true_metric_loss(labels, expt_type, scale)
        return torch.sum(- targets * F.log_softmax(output, -1), -1).mean()

    elif loss_type == 'cross':
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(output, labels) 
        return loss
    
    elif loss_type == 'reg':
        loss_fct = nn.MSELoss()
        loss = loss_fct(output.squeeze(), labels.squeeze())    
        return loss
    
    elif loss_type == 'l1':
        loss_fct = nn.L1Loss()
        loss = loss_fct(output.squeeze(), labels.squeeze())    
        return loss

