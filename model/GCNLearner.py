import torch
import torch.nn as nn
import numpy as np


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

        self.spatialEmbedLinear = nn.Linear(in_features=64,out_features=self.dmodel)
        self.metaLinear1 = nn.Linear(in_features=self.dmodel, out_features=N)
        self.metaCNN = nn.Conv2d(in_channels=T+N,out_channels=N,kernel_size=(1,1))
        self.metaLinear2 = nn.Linear(in_features=1, out_features=T)

    def forward(self, t):
        """
        t: batch*T*dmodel

        return: matrix :根据时间、距离学习的邻接矩阵，batch*T*N*N
        """
        spatialEmbed = self.spatialEmbedLinear(self.spatialEmbed) # N*dmodel
        batch = t.size(0)
        spatialEmbed = spatialEmbed.unsqueeze(0) # 1*N*dmodel
        spatialEmbed = spatialEmbed.repeat(batch,1,1) # batch*N*dmodel
        metaInfo = torch.cat((t,spatialEmbed), 1) # batch*(T+N)*dmodel
        metaInfo = self.metaLinear1(metaInfo).unsqueeze(3) # batch*(T+N)*N*1
        metaInfo = self.metaCNN(metaInfo) # batch*N*N*1
        matrix = self.metaLinear2(metaInfo).permute(0,3,1,2).contiguous() # batch*T*N*N
        return matrix


