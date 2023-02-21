import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import util


# 根据边元特性、节点元特性、自适应产生邻接矩阵
class AdjMeta(nn.Module):
    def __init__(self, args, N, T):
        """
        args: 输入参数
        """
        super(AdjMeta, self).__init__()
        self.N = N
        self.T = T
        self.D = args.dmodel

        self.AdpAdjEm1 = nn.Parameter(torch.randn(N, args.M).to(args.device), requires_grad=True).to(args.device)
        self.AdpAdjEm2 = nn.Parameter(torch.randn(args.M, N).to(args.device), requires_grad=True).to(args.device)

        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self):

        AdpAdj = F.softmax(F.relu(torch.mm(self.AdpAdjEm1,self.AdpAdjEm2)),dim=1) # NN

        return AdpAdj
