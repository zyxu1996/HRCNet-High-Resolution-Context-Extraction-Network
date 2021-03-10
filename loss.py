import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable


class Edge_loss(nn.Module):

    def __init__(self, ignore_index=255):
        super(Edge_loss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, pred, label):
        # h, w = label.size(1), label.size(2)
        pos_num = torch.sum(label == 1, dtype=torch.float)
        neg_num = torch.sum(label == 0, dtype=torch.float)

        weight_pos = neg_num / (pos_num + neg_num)
        weight_neg = pos_num / (pos_num + neg_num)
        weights = torch.tensor([weight_neg, weight_pos])
        edge_loss = F.cross_entropy(pred, label,
                                weights.cuda(), ignore_index=self.ignore_index)

        return edge_loss
        
        
class SE_loss(nn.Module):

    def __init__(self, weight=None):
        super(SE_loss, self).__init__()
        self.bceloss = nn.BCELoss(weight)

    def forward(self, se_pred, target):
        se_target = self._get_batch_label_vector(target, nclass=6).type_as(se_pred)
        loss = self.bceloss(se_pred, se_target)
        return loss

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(),
                               bins=nclass, min=0,
                               max=nclass-1)
            vect = hist>0
            tvect[i] = vect
        return tvect