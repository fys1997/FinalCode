import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class GCNMeta(nn.Module):
    def __init__(self,args,N,T):
        """
        args: 输入参数
        """
        super(GCNMeta, self).__init__()
        self.N = N
        self.T = T
        self.spatialEmbed = np.loadtxt(args.distance_file, skiprows=1)
        self.spatialEmbed = self.spatialEmbed[self.spatialEmbed[..., 0].argsort()]
        self.spatialEmbed = torch.from_numpy(self.spatialEmbed[..., 1:]).float().to(args.device)  # 对应文件的space embed [N*64]

        self.dmodel = args.dmodel

        self.spatialEmbedLinear = nn.Linear(in_features=64,out_features=N)

        self.trainMatrix1 = nn.Parameter(torch.randn(N,args.M).to(args.device),requires_grad=True).to(args.device)
        self.trainMatrix2 = nn.Parameter(torch.randn(args.M,N).to(args.device), requires_grad=True).to(args.device)

        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, t):
        """
        t: batch*T*dmodel

        return: matrix :根据时间、距离学习的邻接矩阵，batch*T*N*N
        """
        matrix = self.dropout(self.spatialEmbedLinear(self.spatialEmbed)) # N*N
        adaptiveMatrix = torch.mm(self.trainMatrix1,self.trainMatrix2) # N*N
        return F.softmax(F.relu(matrix+adaptiveMatrix),dim=1)


