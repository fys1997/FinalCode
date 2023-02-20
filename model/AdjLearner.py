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

        self.EmcAdjCnn = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(1, 1))

        self.NmcAdjEmLinear1 = nn.Linear(in_features=self.D, out_features=args.M)
        self.NmcAdjEmLinear2 = nn.Linear(in_features=self.D, out_features=args.M)

        self.AdpAdjEm1 = nn.Parameter(torch.randn(N, args.M).to(args.device), requires_grad=True).to(args.device)
        self.AdpAdjEm2 = nn.Parameter(torch.randn(args.M, N).to(args.device), requires_grad=True).to(args.device)

        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, EMC, NMC):
        """
        EMC: 边元特性 [N*N*32]
        NMC: 节点元特性[N*D]

        return: Adj 邻接矩阵 [N*N]
        """
        EMC = EMC.unsqueeze(0).permute(0,3,1,2).contiguous() # 1*32*N*N
        EmcAdj = self.EmcAdjCnn(EMC).squeeze(0).squeeze(0) # N*N
        EmcAdj = F.softmax(F.relu(EmcAdj),dim=1) # N*N

        NmcAdjEm1 = self.dropout(self.NmcAdjEmLinear1(NMC)) # N*M
        NmcAdjEm2 = self.dropout(self.NmcAdjEmLinear2(NMC)) # N*M
        NmcAdjEm2 = NmcAdjEm2.permute(1,0).contiguous() # M*N
        NmcAdj = F.softmax(F.relu(torch.mm(NmcAdjEm1,NmcAdjEm2)),dim=1)

        AdpAdj = F.softmax(F.relu(torch.mm(self.AdpAdjEm1,self.AdpAdjEm2)),dim=1) # NN

        Adj = EmcAdj*torch.sigmoid(NmcAdj+AdpAdj)+NmcAdj*torch.sigmoid(EmcAdj+AdpAdj)+AdpAdj*torch.sigmoid(EmcAdj+AdpAdj) # N*N
        return Adj
