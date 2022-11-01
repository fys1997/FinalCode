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
        self.N = N
        self.T = T
        self.outputT = outputT  # output sequence length
        self.device = device
        # 此处m指预训练矩阵的维度数
        self.m = args.M
        # 节点数
        self.num_nodes = N
        self.trainMatrix1 = nn.Parameter(torch.randn(self.num_nodes, self.m).to(device), requires_grad=True).to(device)
        self.trainMatrix2 = nn.Parameter(torch.randn(self.m, self.num_nodes).to(device), requires_grad=True).to(device)
        # hops的值
        self.hops = args.hops
        # AR模块窗口的大小
        self.arSize = args.arSize
        self.dropout = nn.Dropout(p=args.dropout)
        # 定义GCNEncoder
        # self.GcnEncoder=GG.GcnEncoder(num_embedding=args.num_embedding,N=N,trainMatrix1=self.trainMatrix1,
        #                               trainMatrix2=self.trainMatrix2,hops=self.hops,device=device,tradGcn=args.tradGcn,
        #                               dropout=args.dropout,dmodel=args.dmodel,num_heads=args.head,Tin=T,encoderBlocks=args.encoderBlocks)
        # self.GcnDecoder=GG.GcnDecoder(N=N,dmodel=args.dmodel,Tout=outputT,Tin=T,num_heads=args.head,dropout=args.dropout,device=device,
        #                               trainMatrix1=self.trainMatrix1,trainMatrix2=self.trainMatrix2,hops=self.hops,tradGcn=args.tradGcn)
        self.GcnAtteNet=GG.GcnAtteNet(num_embedding=args.num_embedding,N=N,trainMatrix1=self.trainMatrix1,trainMatrix2=self.trainMatrix2,
                                      hops=self.hops,device=device,tradGcn=args.tradGcn,dropout=args.dropout,dmodel=args.dmodel,
                                      num_heads=args.head,Tin=T,Tout=outputT,encoderBlocks=args.encoderBlocks)
        # predict layer
        # self.predict=nn.Linear(in_features=outputT+self.arSize,out_features=outputT)
        # read spatial embedding
        self.spatialEmbed=np.loadtxt(args.spatialEmbedding,skiprows=1)
        self.spatialEmbed=self.spatialEmbed[self.spatialEmbed[...,0].argsort()]
        self.spatialEmbed=torch.from_numpy(self.spatialEmbed[...,1:]).float() # 对应文件的space embed [N*dmodel]

    def forward(self, X, Y, teacher_forcing_ratio):
        """

        :param X: 输入数据，X:batch*node*T*2
        :param Y: 真实值，Y:batch*node*outputT*2
        :return: 输出数据: Y:batch*node*T
        """
        vx=X[...,0] # batch*node*Tin 表示X车流量
        tx=X[...,1] # batch*node*Tin 表示输入X的时间index
        ty=Y[...,1] # batch*node*Tout 表示Y的时间index
        # 把spa
        # 开始encoder
        spatialEmbed=self.spatialEmbed.cuda()
        result=self.GcnAtteNet(vx,tx,ty,spatialEmbed)
        # result=torch.cat([result,vx[...,-self.arSize:]],dim=2)
        # result=self.predict(result)

        return result
