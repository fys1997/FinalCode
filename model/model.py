# 此处定义自己的模型
import torch.nn as nn
import torch
import model.GmanGcn as GG
import numpy as np


class mixNet(nn.Module):
    def __init__(self, args, device, T, N, outputT):
        '''

        :param args: 一些设置的参数
        :param data: 训练的数据
        '''
        super(mixNet, self).__init__()

        self.outputT = outputT  # output sequence length
        self.device = device
        # 此处m指为了对GCN中特性增强矩阵W做矩阵分解的维度
        self.m = args.M
        # 节点数
        self.num_nodes = N

        # hops的值
        self.hops = args.hops

        self.dropout = nn.Dropout(p=args.dropout)

        self.GcnAtteNet=GG.GcnAtteNet(num_embedding=args.num_embedding, N=N, hops=args.hops, device=device, tradGcn=args.tradGcn,
                                      dropout=args.dropout, dmodel=args.dmodel, num_heads=args.head,
                                      Tin=T, Tout=outputT, encoderBlocks=args.encoderBlocks,M=args.M)

    def forward(self, X, Y, teacher_forcing_ratio):
        """

        :param X: 输入数据，X:batch*node*T*2
        :param Y: 真实值，Y:batch*node*outputT*2
        :return: 输出数据: Y:batch*node*T
        """
        vx=X[...,0] # batch*node*Tin 表示X车流量
        tx=X[...,1] # batch*node*Tin 表示输入X的时间index
        ty=Y[...,1] # batch*node*Tout 表示Y的时间index
        # 把sp
        # 开始encoder
        result=self.GcnAtteNet(vx,tx,ty)
        # result=torch.cat([result,vx[...,-self.arSize:]],dim=2)
        # result=self.predict(result)

        return result
