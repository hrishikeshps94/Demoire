import torch
import torch.nn as nn

class KTLoss(nn.Module):
    def __init__(self,alpha=1) -> None:
        super(KTLoss,self).__init__()
        self.l1 = nn.L1Loss()
        self.alpha = alpha
    def forward(self,gt_feat,inp_feat):
        loss = self.alpha*self.l1(gt_feat,inp_feat)
        return loss.mean()