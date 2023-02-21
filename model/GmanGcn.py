import torch
import torch.nn as nn
import model.GCN as GCN

import torch.nn.functional as F
from math import sqrt
import numpy as np
from model.AdjLearner import AdjMeta


class GcnAttentionCell(nn.Module):
    def __init__(self, N, Tin, args):
        """
        """
        super(GcnAttentionCell, self).__init__()
        self.temporalAttention=TemMulHeadAtte(N=N, args=args,T=Tin)

        self.device=args.device
        # 设置gate门
        self.gate=nn.Linear(in_features=2*args.dmodel,out_features=args.dmodel)
        # 设置图卷积层捕获空间特征
        self.Gcn=GCN.GCN(Tin=Tin, N=N, args=args)
        self.dropout = nn.Dropout(p=args.dropout)
        self.batchNormGate = nn.BatchNorm2d(num_features=args.dmodel)

    def forward(self,hidden, matrix):
        """

        :param hidden: 此次输入的hidden:batch*N*Tin*dmodel
        :param matrix: 动态得到的矩阵 batch*T*N*N
        :return:
        """
        # GCN捕获空间依赖
        gcnOutput=self.Gcn(hidden.permute(0,3,1,2).contiguous(),matrix) # batch*dmodel*N*T
        gcnOutput=gcnOutput.permute(0,2,3,1).contiguous() # batch*N*T*dmodel

        # 捕获时间依赖

        key=hidden # batch*N*Tin*dmodel

        query=hidden # batch*N*Tin*dmodel

        value=hidden # batch*N*Tin*dmodel

        # 做attention
        atten_mask=GcnAttentionCell.generate_square_subsequent_mask(B=query.size(0), N=query.size(1), T=query.size(2)).to(self.device) # batch*N*1*Tq*Ts
        value,atten=self.temporalAttention.forward(query=query, key=key, value=value, atten_mask=atten_mask) # batch*N*T*dmodel

        # 做gate
        gateInput=torch.cat([gcnOutput,value],dim=3) # batch*N*Tin*2dmodel
        gateInput=self.dropout(self.gate(gateInput)) # batch*N*Tin*dmodel
        gateInput=self.batchNormGate(gateInput.permute(0,3,1,2).contiguous()) # batch*dmodel*N*Tin
        z=torch.sigmoid(gateInput.permute(0,2,3,1).contiguous()) # batch*N*Tin*dmodel
        finalHidden=z*gcnOutput+(1-z)*value # batch*N*Tin*dmodel

        return finalHidden # batch*N*Tin*dmodel

    @staticmethod
    def generate_square_subsequent_mask(B,N,T) -> torch.Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        mask_shape=[B,N,1,T,T]
        with torch.no_grad():
            return torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1)


class GcnEncoder(nn.Module):
    def __init__(self,N,Tin,args):
        super(GcnEncoder, self).__init__()
        self.encoderBlock=nn.ModuleList()
        self.AdjLearner = AdjMeta(args=args, N=N, T=Tin)
        for i in range(args.encoderBlocks):
            self.encoderBlock.append(GcnAttentionCell(N=N, Tin=Tin, args=args))

        self.device=args.device
        self.encoderBlocks=args.encoderBlocks

        self.xFull=nn.Linear(in_features=2,out_features=args.dmodel)
        self.N=N

    def forward(self,x):
        """

        :param x: 流量数据:[batch*N*Tin*2]
        :return:
        """
        x=self.xFull(x) # batch*N*Tin*dmodel

        hidden=x # batch*N*Tin*dmodel
        skip=0
        matrix = self.AdjLearner() # N*N
        for i in range(self.encoderBlocks):
            hidden=self.encoderBlock[i].forward(hidden=hidden, matrix=matrix)
            skip = skip + hidden

        return skip+x


class GcnDecoder(nn.Module):
    def __init__(self,N,Tout,Tin,args):
        super(GcnDecoder, self).__init__()
        self.N=N
        self.Tin=Tin
        self.Tout=Tout
        self.TinToutCNN = nn.Conv2d(in_channels=Tin, out_channels=Tout, kernel_size=(1,1))
        self.predictLinear = nn.Linear(in_features=args.dmodel, out_features=2)

        self.decoderBlock = nn.ModuleList()
        self.decoderBlocks = args.decoderBlocks
        for i in range(args.decoderBlocks):
            self.decoderBlock.append(GcnAttentionCell(N=N, Tin=Tout, args=args))

        self.AdjLearner = AdjMeta(args=args, N=N, T=Tout)
        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self,x):
        """

        :param x: # batch*N*Tin*dmodel
        :return:
        """
        x = x.permute(0,2,1,3).contiguous()
        x = self.TinToutCNN(x) # batch*Tout*N*D
        x = x.permute(0,2,1,3).contiguous() # batch*N*Tout*D

        hidden = x  # batch*N*Tout*dmodel
        skip = 0
        matrix = self.AdjLearner() # N*N

        for i in range(self.decoderBlocks):
            hidden = self.decoderBlock[i].forward(hidden=hidden, matrix=matrix)
            skip = skip+hidden
        x=self.dropout(self.predictLinear(x+skip)) # batch*N*Tout*2

        return x # batch*N*Tout*2


class TemMulHeadAtte(nn.Module):
    def __init__(self,N,args,T):
        """
        """
        super(TemMulHeadAtte, self).__init__()
        self.dmodel=args.dmodel
        self.num_heads=args.head
        self.dropout=nn.Dropout(p=args.dropout)
        self.device=args.device
        self.M=args.M
        self.N=N

        d_keys=self.dmodel//self.num_heads
        d_values=self.dmodel//self.num_heads
        self.QueryLinear = nn.Linear(in_features=self.dmodel, out_features=d_keys*self.num_heads)
        self.KeyLinear = nn.Linear(in_features=self.dmodel, out_features=d_keys*self.num_heads)
        self.ValueLinear = nn.Linear(in_features=self.dmodel, out_features=d_values*self.num_heads)
        self.OutLinear = nn.Linear(in_features=d_keys*self.num_heads, out_features=self.dmodel)

    def forward(self, query, key, value, atten_mask):
        """

        :param query: [batch*N*T*dmodel]
        :param key: [batch*N*T*dmodel]
        :param value: [batch*N*T*dmodel]
        :param atten_mask: [batch*N*1*Tq*Ts]
        :return: [batch*N*T*dmodel]
        """
        B,N,T,E=query.shape
        H=self.num_heads

        query = self.QueryLinear(query) # batch*N*T*(key*dmodel)
        query = query.view(B,N,T,H,-1) # batch*N*T*heads*dkeys

        key = self.KeyLinear(key)
        key = key.view(B,N,T,H,-1) # batch*N*T*heads*dkeys

        value = self.ValueLinear(value)
        value = value.view(B,N,T,H,-1) # batch*N*T*heads*dkeys

        scale=1./sqrt(query.size(4))

        scores = torch.einsum("bnthe,bnshe->bnhts",(query,key)) # batch*N*head*Tq*Ts
        if atten_mask is not None:
            scores.masked_fill_(atten_mask,-np.inf) # batch*N*head*Tq*Ts
        scores=torch.softmax(scale*scores,dim=-1)

        value=torch.einsum("bnhts,bnshd->bnthd",(scores,value)) # batch*N*T*heads*d_values
        value=value.contiguous()
        value=value.view(B,N,T,-1) # batch*N*T*(heads*d_values)
        value = self.OutLinear(value) # batch*N*T*D

        # 返回最后的向量和得到的attention分数
        return value,scores


class GcnAtteNet(nn.Module):
    def __init__(self, N, Tin, Tout, args):
        super(GcnAtteNet, self).__init__()
        self.N = N
        self.GcnEncoder=GcnEncoder(N=N,Tin=Tin,args=args)
        self.GcnDecoder=GcnDecoder(N=N,Tout=Tout,Tin=Tin,args=args)

    def forward(self,X):
        output = self.GcnEncoder(X)  # batch*N*Tin*dmodel
        result = self.GcnDecoder(output)  # batch*N*Tout*2
        return result






