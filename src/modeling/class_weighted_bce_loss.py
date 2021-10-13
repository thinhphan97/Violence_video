import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, class_weights, reduction='none'):
        super(WeightedBCEWithLogitsLoss, self).__init__()
        self.class_weights = class_weights
        self.reduction = reduction

    def forward(self, input, target):
        self.class_weights = self.class_weights.to(input.device)
        if input.dtype == torch.float16:
            self.class_weights = self.class_weights.half()
        class_losses = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        class_losses = class_losses.mean(0) * self.class_weights
        if self.reduction == 'none':
            return class_losses
        elif self.reduction == 'mean':  
            return class_losses.sum() / self.class_weights.sum()
