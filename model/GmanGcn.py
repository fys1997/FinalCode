import torch
import torch.nn as nn
import model.GCN as GCN

import torch.nn.functional as F
from math import sqrt
import numpy as np
from model.AttentionLearner import AttentionMeta
from model.GCNLearner import GCNMeta


class GcnEncoderCell(nn.Module):
    def __init__(self, N, Tin, args):
        """
        """
        super(GcnEncoderCell, self).__init__()
        self.temporalAttention=TemMulHeadAtte(N=N, args=args,T=Tin)

        self.device=args.device
        # 设置gate门
        self.gate=nn.Linear(in_features=2*args.dmodel,out_features=args.dmodel)
        # 设置图卷积层捕获空间特征
        self.Gcn=GCN.GCN(Tin=Tin, N=N, args=args)

    def forward(self,hidden,tXin,matrix):
        """

        :param hidden: 此次输入的hidden:batch*N*Tin*dmodel
        :param tXin: 加了timeEmbedding的x值：tXin:[batch*N*Tin*dmodel]
        :param matrix: 动态得到的矩阵 batch*T*N*N
        :return:
        """
        # GCN捕获空间依赖
        gcnOutput=self.Gcn(hidden.permute(0,3,1,2).contiguous(),matrix) # batch*dmodel*N*T
        gcnOutput=gcnOutput.permute(0,2,3,1).contiguous() # batch*N*T*dmodel

        # 捕获时间依赖

        key=torch.cat([hidden,tXin],dim=3) # batch*N*Tin*2dmodel

        query=torch.cat([hidden,tXin],dim=3) # batch*N*Tin*2dmodel

        value=torch.cat([hidden,tXin],dim=3) # batch*N*Tin*2dmodel

        # 做attention
        atten_mask=GcnEncoderCell.generate_square_subsequent_mask(B=query.size(0),N=query.size(1),T=query.size(2)).to(self.device) # batch*N*1*Tq*Ts
        value,atten=self.temporalAttention.forward(query=query,key=key,value=value,atten_mask=atten_mask,tX=tXin[:,0,:,:]) # batch*N*T*dmodel

        # 做gate
        gateInput=torch.cat([gcnOutput,value],dim=3) # batch*N*Tin*2dmodel
        gateInput=self.gate(gateInput) # batch*N*Tin*dmodel
        gateInput=gateInput.permute(0,3,1,2).contiguous() # batch*dmodel*N*Tin
        z=torch.sigmoid(gateInput.permute(0,2,3,1).contiguous()) # batch*N*Tin*dmodel
        finalHidden=z*gcnOutput+(1-z)*value # batch*N*Tin*dmodel

        return finalHidden+hidden # batch*N*Tin*dmodel

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
        self.GCNMeta = GCNMeta(args=args,N=N,T=Tin)
        for i in range(args.encoderBlocks):
            self.encoderBlock.append(GcnEncoderCell(N=N,Tin=Tin, args=args))

        self.device=args.device
        self.encoderBlocks=args.encoderBlocks

        self.xFull=nn.Linear(in_features=1,out_features=args.dmodel)
        self.N=N

    def forward(self,x,tx):
        """

        :param x: 流量数据:[batch*N*Tin*1]
        :param tx: 时间数据:[batch*N*Tin*dmodel]
        :return:
        """
        x=self.xFull(x) # batch*N*Tin*dmodel

        hidden=x # batch*N*Tin*dmodel
        skip=0
        matrix = self.GCNMeta(t=tx[:,0,:,:]) # batch*T*N*N
        for i in range(self.encoderBlocks):
            hidden=self.encoderBlock[i].forward(hidden=hidden,tXin=tx, matrix=matrix)
            skip = skip + hidden

        return skip+x


class GcnDecoder(nn.Module):
    def __init__(self,N,Tout,Tin,args):
        super(GcnDecoder, self).__init__()
        self.N=N
        self.Tin=Tin
        self.Tout=Tout
        self.dmodelCNN=nn.Conv2d(in_channels=args.dmodel,out_channels=1,kernel_size=(1,1))
        self.TinToutTrainMatrix1=nn.Parameter(torch.randn(N,args.M).to(args.device),requires_grad=True).to(args.device) # N*M
        self.TinToutTrainMatrix2=nn.Parameter(torch.randn(args.M,Tin*Tout).to(args.device),requires_grad=True).to(args.device) # M*(Tin*Tout)
        self.GcnDecoderCell=GcnEncoderCell(N=N,Tin=Tin, args=args)
        self.GCNMeta = GCNMeta(args=args,N=N,T=Tout)

    def forward(self,x,ty):
        """

        :param x: # batch*N*Tin*dmodel
        :param ty: batch*N*Tout*dmodel
        :return:
        """
        x=x.permute(0,3,1,2).contiguous() # batch*dmodel*N*Tin
        TinToutTrainMatrix = torch.reshape(torch.mm(self.TinToutTrainMatrix1,self.TinToutTrainMatrix2),(self.N,self.Tin,self.Tout)) # N*Tin*Tout
        x = torch.einsum("bdni,nit->bdnt",(x,TinToutTrainMatrix)) # batch*dmodel*N*Tout
        x=x.permute(0,2,3,1).contiguous() # batch*N*Tout*dmodel
        matrix = self.GCNMeta(t=ty[:,0,:,:]) # batch*Tout*N*N
        x=self.GcnDecoderCell.forward(hidden=x,tXin=ty,matrix=matrix) # batch*N*Tout*dmodel
        x=self.dmodelCNN(x.permute(0,3,1,2).contiguous()).squeeze(dim=1) # batch*N*Tout

        return x # batch*N*Tout


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

        d_keys=2*self.dmodel//self.num_heads
        d_values=2*self.dmodel//self.num_heads

        self.AttentionMeta = AttentionMeta(args=args,N=N,T=T)
        self.leakeyRelu = nn.LeakyReLU(0.1)

        # self.query_projection=nn.Linear(in_features=2*self.dmodel,out_features=d_keys*self.num_heads)
        # self.key_projection=nn.Linear(in_features=2*self.dmodel,out_features=d_keys*self.num_heads)
        # self.value_projection=nn.Linear(in_features=2*self.dmodel,out_features=d_values*self.num_heads)
        self.out_projection=nn.Linear(in_features=d_values*self.num_heads,out_features=self.dmodel)

    def forward(self,query,key,value,atten_mask,tX):
        """

        :param query: [batch*N*T*dmodel]
        :param key: [batch*N*T*dmodel]
        :param value: [batch*N*T*dmodel]
        :param atten_mask: [batch*N*1*Tq*Ts]
        :param tX: [batch*T*dmodel]
        :return: [batch*N*T*dmodel]
        """
        B,N,T,E=query.shape
        H=self.num_heads

        K,Q,V = self.AttentionMeta(tX) # batch*N*2dmodel*(d_keys*num_heads)

        query = self.leakeyRelu(torch.einsum("bnti,bnik->bntk",(query,Q)).view(B,N,T,H,-1)) # batch*N*T*heads*d_keys
        key = self.leakeyRelu(torch.einsum("bnti,bnik->bntk",(key,K)).view(B,N,T,H,-1)) # batch*N*T*heads*d_keys
        value = self.leakeyRelu(torch.einsum("bnti,bnik->bntk",(value,V)).view(B,N,T,H,-1)) # batch*N*T*heads*d_values
        # query=F.relu(self.query_projection(query).view(B,N,T,H,-1)) # batch*N*T*heads*d_keys
        # key=F.relu(self.key_projection(key).view(B,N,T,H,-1)) # batch*N*T*heads*d_keys
        # value=F.relu(self.value_projection(value).view(B,N,T,H,-1)) # batch*N*T*heads*d_values

        scale=1./sqrt(query.size(4))

        scores=torch.einsum("bnthe,bnshe->bnhts",(query,key)) # batch*N*head*Tq*Ts
        if atten_mask is not None:
            scores.masked_fill_(atten_mask,-np.inf) # batch*N*head*Tq*Ts
        scores=self.dropout(torch.softmax(scale*scores,dim=-1))

        value=torch.einsum("bnhts,bnshd->bnthd",(scores,value)) # batch*N*T*heads*d_values
        value=value.contiguous()
        value=value.view(B,N,T,-1) # batch*N*T*dmodel
        value=self.leakeyRelu(self.out_projection(value))

        # 返回最后的向量和得到的attention分数
        return value,scores


class GcnAtteNet(nn.Module):
    def __init__(self, N, Tin, Tout, args):
        super(GcnAtteNet, self).__init__()
        self.N = N
        self.GcnEncoder=GcnEncoder(N=N,Tin=Tin,args=args)
        self.GcnDecoder=GcnDecoder(N=N,Tout=Tout,Tin=Tin,args=args)

    def forward(self,vx,tx,ty):
        tx = tx.unsqueeze(dim=1)  # batch*1*Tin*dmodel
        tx = tx.repeat(1, self.N, 1, 1)  # batch*N*Tin*dmodel
        ty = ty.unsqueeze(dim=1)
        ty = ty.repeat(1, self.N, 1, 1)  # batch*N*Tout*dmodel
        output = self.GcnEncoder(vx.unsqueeze(dim=3), tx)  # batch*N*Tin*dmodel
        result = self.GcnDecoder(output, ty)  # batch*N*Tout
        return result






