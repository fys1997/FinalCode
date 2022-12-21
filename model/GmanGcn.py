import torch
import torch.nn as nn
import model.GCN as GCN
import model.timeEmbedding as TE
import torch.nn.functional as F
from math import sqrt
import numpy as np


class GcnEncoderCell(nn.Module):
    def __init__(self,N,hops,device,tradGcn,dropout,dmodel,num_heads,Tin,M):
        """

        :param num_embedding: 有多少组时间，此处288
        :param embedding_dim:
        :param Tin:
        :param hops: gcn 跳数
        :param device:
        :param tradGcn:
        :param dropout:
        :param dmodel:
        :param num_heads:
        :param Tin: 输入时间的长度
        :param M: 跟GCN矩阵特性增强有关的做矩阵分解的维度值M
        """
        super(GcnEncoderCell, self).__init__()
        self.temporalAttention=TemMulHeadAtte(dmodel=dmodel,num_heads=num_heads,dropout=dropout,device=device,M=M,N=N)

        self.device=device
        # 设置gate门
        self.gate=nn.Linear(in_features=2*dmodel,out_features=dmodel)
        # 设置图卷积层捕获空间特征
        self.Gcn=GCN.GCN(Tin=Tin, N=N, device=device, tradGcn=tradGcn,
                         dropout=dropout, hops=hops, dmodel=dmodel, M=M)

    def forward(self,hidden,tXin):
        """

        :param x: 只含流量值的embed batch*N*Tin*dmodel
        :param hidden: 此次输入的hidden:batch*N*Tin*dmodel
        :param tXin: 加了timeEmbedding的x值：tXin:[batch*N*Tin*dmodel]
        :return:
        """
        # GCN捕获空间依赖
        gcnOutput=self.Gcn(hidden.permute(0,3,1,2).contiguous(),tXin) # batch*dmodel*N*T
        gcnOutput=gcnOutput.permute(0,2,3,1).contiguous() # batch*N*T*dmodel

        # 捕获时间依赖

        key=torch.cat([hidden,tXin],dim=3) # batch*N*Tin*2dmodel

        query=torch.cat([hidden,tXin],dim=3) # batch*N*Tin*2dmodel

        value=torch.cat([hidden,tXin],dim=3) # batch*N*Tin*dmodel

        # 做attention
        atten_mask=GcnEncoderCell.generate_square_subsequent_mask(B=query.size(0),N=query.size(1),T=query.size(2)).to(self.device) # batch*N*1*Tq*Ts
        value,atten=self.temporalAttention.forward(query=query,key=key,value=value,atten_mask=atten_mask) # batch*N*T*dmodel

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
    def __init__(self,num_embedding,N,hops,device,tradGcn,
                 dropout,dmodel,num_heads,Tin,encoderBlocks,M):
        super(GcnEncoder, self).__init__()
        self.encoderBlock=nn.ModuleList()
        for i in range(encoderBlocks):
            self.encoderBlock.append(GcnEncoderCell(N=N, hops=hops, device=device, tradGcn=tradGcn,
                                                    dropout=dropout, dmodel=dmodel, num_heads=num_heads, Tin=Tin,
                                                    M=M))
        self.timeEmbed=TE.timeEmbedding(num_embedding=num_embedding,embedding_dim=dmodel,dropout=dropout)
        self.device=device
        self.encoderBlocks=encoderBlocks

        self.xFull=nn.Linear(in_features=1,out_features=dmodel)

    def forward(self,x,tx,ty):
        """

        :param x: 流量数据:[batch*N*Tin*1]
        :param tx: 时间数据:[batch*N*Tin]
        :param spaceEmbed: 空间特征:[N*dmodel]
        :param ty: [batch*N*Tout]
        :return:
        """
        x=self.xFull(x) # batch*N*Tin*dmodel
        tx=self.timeEmbed(tx) # batch*N*Tin*dmodel
        ty=self.timeEmbed(ty) # batch*N*Tout*dmodel
        hidden=x # batch*N*Tin*dmodel
        skip=0
        for i in range(self.encoderBlocks):
            hidden=self.encoderBlock[i].forward(hidden=hidden,tXin=tx) # Tin*batch*N
            skip = skip + hidden

        return skip+x,ty


class GcnDecoder(nn.Module):
    def __init__(self,N,dmodel,Tout,Tin,num_heads,dropout,device,hops,tradGcn,M):
        super(GcnDecoder, self).__init__()
        self.N=N
        self.Tin=Tin
        self.Tout=Tout
        self.dmodelCNN=nn.Conv2d(in_channels=dmodel,out_channels=1,kernel_size=(1,1))
        self.TinToutTrainMatrix1=nn.Parameter(torch.randn(N,M).to(device),requires_grad=True).to(device) # N*M
        self.TinToutTrainMatrix2=nn.Parameter(torch.randn(M,Tin*Tout).to(device),requires_grad=True).to(device) # M*(Tin*Tout)
        self.GcnDecoderCell=GcnEncoderCell(N=N,hops=hops,device=device,
                                           tradGcn=tradGcn,dropout=dropout,dmodel=dmodel,num_heads=num_heads,Tin=Tout,
                                           M=M)

    def forward(self,x,ty):
        """

        :param x: # batch*N*Tin*dmodel
        :param ty: batch*N*Tout*dmodel
        :return:
        """
        ty=ty.permute(0,2,1,3).contiguous() # batch*Tout*N*dmodel
        x=x.permute(0,1,3,2).contiguous() # batch*N*dmodel*Tin
        TinToutTrainMatrix = torch.reshape(torch.mm(self.TinToutTrainMatrix1,self.TinToutTrainMatrix2),(self.N,self.Tin,self.Tout)) # N*Tin*Tout
        x = torch.einsum("bndi,nit->bndt",(x,TinToutTrainMatrix)) # batch*N*dmodel*Tout
        x=x.permute(0,1,3,2).contiguous() # batch*N*Tout*dmodel
        x=self.GcnDecoderCell.forward(hidden=x,tXin=ty.permute(0,2,1,3).contiguous()) # batch*N*Tout*dmodel
        x=self.dmodelCNN(x.permute(0,3,1,2).contiguous()).squeeze(dim=1) # batch*N*Tout

        return x # batch*N*Tout


class TemMulHeadAtte(nn.Module):
    def __init__(self,dmodel,num_heads,dropout,device,M,N):
        """

        :param dmodel: embeddings之后的每个V每个T时刻的size
        :param num_heads: 多头注意力机制的head count
        """
        super(TemMulHeadAtte, self).__init__()
        self.dmodel=dmodel
        self.num_heads=num_heads
        self.dropout=nn.Dropout(p=dropout)
        self.device=device
        self.M=M
        self.N=N

        d_keys=2*dmodel//num_heads
        d_values=2*dmodel//num_heads

        self.query_projection1=nn.Parameter(torch.randn(N*dmodel,M),requires_grad=True).to(device)
        self.query_projection2=nn.Parameter(torch.randn(M,2*d_keys*num_heads),requires_grad=True).to(device)

        self.key_projection1=nn.Parameter(torch.randn(N*dmodel,M),requires_grad=True).to(device)
        self.key_projection2=nn.Parameter(torch.randn(M,2*d_keys*num_heads),requires_grad=True).to(device)

        self.value_projection1=nn.Parameter(torch.randn(N*dmodel,M),requires_grad=True).to(device)
        self.value_projection2=nn.Parameter(torch.randn(M,2*d_values*num_heads),requires_grad=True).to(device)

        # self.query_projection=nn.Linear(in_features=2*dmodel,out_features=d_keys*num_heads)
        # self.key_projection=nn.Linear(in_features=2*dmodel,out_features=d_keys*num_heads)
        # self.value_projection=nn.Linear(in_features=2*dmodel,out_features=d_values*num_heads)
        self.out_projection=nn.Linear(in_features=d_values*num_heads,out_features=dmodel)

    def forward(self,query,key,value,atten_mask):
        """

        :param query: [batch*N*T*dmodel]
        :param key: [batch*N*T*dmodel]
        :param value: [batch*N*T*dmodel]
        :param atten_mask: [batch*N*1*Tq*Ts]
        :return: [batch*N*T*dmodel]
        """
        B,N,T,E=query.shape
        H=self.num_heads

        query_projection=torch.mm(self.query_projection1,self.query_projection2) # N*(dmodel*2*d_keys*num_heads)
        query_projection=torch.reshape(query_projection,(self.N,2*self.dmodel,-1)) # N *(2dmodel)*(d_keys*num_heads)
        key_projection=torch.mm(self.key_projection1,self.key_projection2) # N*(dmodel*2*d_keys*num_heads)
        key_projection=torch.reshape(key_projection,(self.N,2*self.dmodel,-1)) # N*(2dmodel)*(d_keys*num_heads)
        value_projection=torch.mm(self.value_projection1,self.value_projection2) # N*(dmodel*2*d_values*num_heads)
        value_projection=torch.reshape(value_projection,(self.N,2*self.dmodel,-1)) # N*(2dmodel)*(d_values*num_heads)

        query=F.relu(torch.einsum("bnti,nik->bntk",(query,query_projection)).view(B,N,T,H,-1)) # batch*N*T*heads*d_keys
        key=F.relu(torch.einsum("bnti,nik->bntk",(key,key_projection)).view(B,N,T,H,-1)) # batch*N*T*heads*d_keys
        value=F.relu(torch.einsum("bnti,nik->bntk",(value,value_projection)).view(B,N,T,H,-1)) # batch*N*T*heads*d_values

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
        value=F.relu(self.out_projection(value))

        # 返回最后的向量和得到的attention分数
        return value,scores


class GcnAtteNet(nn.Module):
    def __init__(self,num_embedding,N,hops,device,tradGcn,
                 dropout,dmodel,num_heads,Tin,Tout,encoderBlocks,M):
        super(GcnAtteNet, self).__init__()
        self.GcnEncoder=GcnEncoder(num_embedding=num_embedding,N=N,hops=hops,
                                   device=device,tradGcn=tradGcn,dropout=dropout,dmodel=dmodel,num_heads=num_heads,
                                   Tin=Tin,encoderBlocks=encoderBlocks, M=M)
        self.GcnDecoder=GcnDecoder(N=N,dmodel=dmodel,Tout=Tout,Tin=Tin,num_heads=num_heads,dropout=dropout,device=device,
                                   hops=hops,tradGcn=tradGcn, M=M)

    def forward(self,vx,tx,ty):
        output, ty = self.GcnEncoder(vx.unsqueeze(dim=3), tx, ty)  # batch*N*Tin*dmodel
        result = self.GcnDecoder(output, ty)  # batch*N*Tout
        return result






