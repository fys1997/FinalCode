import torch
import torch.nn as nn
import model.GCN as GCN

import torch.nn.functional as F
from math import sqrt
import numpy as np
from model.AttentionLearner import KQVLinear
from model.AdjLearner import AdjMeta
from model.NCL import NCL
from model.NodeNormalWLearner import NodeNormalWLearner


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

    def forward(self,hidden, matrix, EMC, NMC):
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
        value,atten=self.temporalAttention.forward(query=query, key=key, value=value, atten_mask=atten_mask, NMC=NMC) # batch*N*T*dmodel

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

    def forward(self,x, NMC, EMC):
        """

        :param x: 流量数据:[batch*N*Tin*2]
        :return:
        """
        x=self.xFull(x) # batch*N*Tin*dmodel

        hidden=x # batch*N*Tin*dmodel
        skip=0
        matrix = self.AdjLearner(EMC, NMC) # N*N
        for i in range(self.encoderBlocks):
            hidden=self.encoderBlock[i].forward(hidden=hidden, matrix=matrix, EMC=EMC, NMC=NMC)
            skip = skip + hidden

        return skip+x


class GcnDecoder(nn.Module):
    def __init__(self,N,Tout,Tin,args):
        super(GcnDecoder, self).__init__()
        self.N=N
        self.Tin=Tin
        self.Tout=Tout
        self.TinToutWLearner = NodeNormalWLearner(args=args, N=N, in_features=Tin, out_features=Tout)
        self.predictLinear = nn.Linear(in_features=args.dmodel, out_features=2)

        self.decoderBlock = nn.ModuleList()
        self.decoderBlocks = args.decoderBlocks
        for i in range(args.decoderBlocks):
            self.decoderBlock.append(GcnAttentionCell(N=N, Tin=Tout, args=args))

        self.AdjLearner = AdjMeta(args=args, N=N, T=Tout)
        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self,x,NMC,EMC):
        """

        :param x: # batch*N*Tin*dmodel
        :return:
        """
        TinToutW = self.TinToutWLearner(NMC) # N*TinTout
        x = torch.einsum("bntd,nto->bnod",(x,TinToutW)) # batch*N*Tout*D

        hidden = x  # batch*N*Tout*dmodel
        skip = 0
        matrix = self.AdjLearner(EMC=EMC, NMC=NMC) # N*N

        for i in range(self.decoderBlocks):
            hidden = self.decoderBlock[i].forward(hidden=hidden, matrix=matrix, EMC=EMC, NMC=NMC)
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

        self.AttentionLearner = KQVLinear(args=args, N=N, T=T)

        self.OutCnn = nn.Conv2d(in_channels=d_values*self.num_heads,out_channels=self.dmodel,kernel_size=(1,1))
        self.batchNormOut = nn.BatchNorm2d(num_features=self.dmodel)
        self.LeakeyRelu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, query, key, value, atten_mask, NMC):
        """

        :param query: [batch*N*T*dmodel]
        :param key: [batch*N*T*dmodel]
        :param value: [batch*N*T*dmodel]
        :param atten_mask: [batch*N*1*Tq*Ts]
        :return: [batch*N*T*dmodel]
        """
        B,N,T,E=query.shape
        H=self.num_heads

        Wk,bk,Wq,bq,Wv,bv = self.AttentionLearner(NMC) # (N*dmodel*(dkeys*num_heads)), (N*(dkeys*num_heads))

        query = torch.einsum("bnti,nik->bntk",(query,Wq)) # batch*N*T*(dk*heads)
        query = query+bq.unsqueeze(1).unsqueeze(0) # batch*N*T*(dk*heads)
        query = query.view(B,N,T,H,-1) # batch*N*T*heads*dkeys

        key = torch.einsum("bnti,nik->bntk",(key,Wk)) # batch*N*T*(dk*heads)
        key = key+bk.unsqueeze(1).unsqueeze(0) # batch*N*T*(dk*heads)
        key = key.view(B,N,T,H,-1) # batch*N*T*heads*dkeys

        value = torch.einsum("bnti,nik->bntk", (value, Wv))  # batch*N*T*(dk*heads)
        value = value+bv.unsqueeze(1).unsqueeze(0) # batch*N*T*(dk*heads)
        value = value.view(B,N,T,H,-1) # batch*N*T*heads*dkeys

        scale=1./sqrt(query.size(4))

        scores = self.LeakeyRelu(torch.einsum("bnthe,bnshe->bnhts",(query,key))) # batch*N*head*Tq*Ts
        if atten_mask is not None:
            scores.masked_fill_(atten_mask,-np.inf) # batch*N*head*Tq*Ts
        scores=torch.softmax(scale*scores,dim=-1)

        value=torch.einsum("bnhts,bnshd->bnthd",(scores,value)) # batch*N*T*heads*d_values
        value=value.contiguous()
        value=value.view(B,N,T,-1) # batch*N*T*(heads*d_values)
        value = self.OutCnn(value.permute(0,3,1,2).contiguous()) # batch*D*N*T

        value = self.batchNormOut(value) # batch*dmodel*N*T
        value = torch.sigmoid(value).permute(0,2,3,1).contiguous() # batch*N*T*dmodel

        # 返回最后的向量和得到的attention分数
        return value,scores


class GcnAtteNet(nn.Module):
    def __init__(self, N, Tin, Tout, args, NC, EC):
        super(GcnAtteNet, self).__init__()
        self.N = N
        self.GcnEncoder=GcnEncoder(N=N,Tin=Tin,args=args)
        self.GcnDecoder=GcnDecoder(N=N,Tout=Tout,Tin=Tin,args=args)
        self.NC = NC
        self.EC = EC
        self.NCL = NCL(args,N)

    def forward(self,X):
        NMC = self.NCL(self.NC) # N*D
        EMC = self.EC # N*N*32
        output = self.GcnEncoder(X, NMC=NMC, EMC=EMC)  # batch*N*Tin*dmodel
        result = self.GcnDecoder(output, NMC=NMC, EMC=EMC)  # batch*N*Tout*2
        return result






