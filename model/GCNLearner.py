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
        self.metaLinear1 = nn.Linear(in_features=self.dmodel, out_features=N*N)
        self.metaLinear2 = nn.Linear(in_features=T+N, out_features=T)

    def forward(self, t):
        """
        tX: batch*T*dmodel

        return: matrix :根据时间、距离学习的邻接矩阵，batch*T*N*N
        """
        spatialEmbed = self.spatialEmbedLinear(self.spatialEmbed) # N*dmodel
        batch = t.size(0)
        spatialEmbed = spatialEmbed.unsqueeze(0) # 1*N*dmodel
        spatialEmbed = spatialEmbed.repeat(batch,1,1) # batch*N*dmodel
        metaInfo = torch.cat((t,spatialEmbed), 1) # batch*(T+N)*dmodel
        metaInfo = self.metaLinear1(metaInfo) # batch*(T+N)*(N*N)
        metaInfo = self.metaLinear2(metaInfo.permute(0,2,1).contiguous()).permute(0,2,1).contiguous() # batch*T*(N*N)
        matrix = metaInfo.view(batch,self.T,self.N,self.N) # batch*T*N*N
        return matrix


