import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def get_loss_fn(loss_fn_name):
    return globals()[loss_fn_name]


class Focal_Loss(torch.nn.Module):
    def __init__(self, alpha=.25, gamma=2, logits=False, reduce=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.CrossEntropyLoss=torch.nn.CrossEntropyLoss(reduction='none')

    def __call__(self, inputs, targets, *kwargs):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:

            BCE_loss = self.CrossEntropyLoss(inputs, targets)

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        
        if self.reduce:
            return F_loss.sum() / (len(targets) - (targets == -100).sum())
        else:
            return F_loss
        
        
class CrossEntropy_Loss():
    
    def __init__(self):
        self.CrossEntropyLoss=torch.nn.CrossEntropyLoss(reduction='none')
            
    def __call__(self, preds, targets, reduce=True):
        
        loss = self.CrossEntropyLoss(preds, targets)
        
        if reduce:
            return loss.sum() / (len(targets) - (targets == -100).sum())
        
        else:
            return loss
    
    
