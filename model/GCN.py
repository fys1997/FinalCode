import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self,Tin,  device, hops,N,
                dmodel, M, tradGcn=False, dropout=0.1):
        super().__init__()
        self.T=Tin # 输入时间维度
        self.N=N # 邻接矩阵边数 N*N
        self.dmodel=dmodel

        self.device = device
        self.hops = hops
        self.tradGcn = tradGcn
        self.dropout = nn.Dropout(p=dropout)

        # 设置自适应邻接矩阵
        self.trainMatrix1=nn.Parameter(torch.randn(N,M).to(device),requires_grad=True).to(device)
        self.trainMatrix2=nn.Parameter(torch.randn(M,N).to(device),requires_grad=True).to(device)

        # 设置GCN增强的矩阵分解的维度
        self.trainW1List = list()
        self.trainW2List = list()
        for i in range(hops):
            self.trainW1List.append(nn.Parameter(torch.randn(N,M).to(device),requires_grad=True).to(device))
            self.trainW2List.append(nn.Parameter(torch.randn(M,Tin*Tin).to(device),requires_grad=True).to(device))

        # 运用传统图卷积
        self.tradGcn = tradGcn

        if tradGcn:
            self.tradGcnW = nn.ModuleList()
            for i in range(self.hops):
                self.tradGcnW.append(nn.Linear(self.T,self.T))
        else:
            self.gcnLinear = nn.Linear(Tin * (self.hops + 1), Tin)

    def forward(self,X,timeEmbedding):
        """

        :param X: batch*dmodel*node*T
        :param timeEmbedding: batch*N*Tin*dmodel
        :return: Hout:[batch*dmodel*node*T]
        """
        # 开始动态学习生成拉普拉斯矩阵 A=Relu(alpha*sin(timeEmbedding)+beta)
        timeEmbedding=timeEmbedding.permute(0,2,1,3).contiguous() # batch*Tin*N*dmodel
        A = F.relu(torch.mm(self.trainMatrix1,self.trainMatrix2)) # N*N
        A = F.softmax(A,dim=1) # N*N

        H = list()
        H.append(X)
        Hbefore = X  # X batch*dmodel*node*T
        # 开始图卷积部分
        if self.tradGcn == False:
            for k in range(self.hops):
                # 开始生成对应的GCN矩阵增强W矩阵
                W = torch.mm(self.trainW1List[k],self.trainW2List[k]) # N*(T*T)
                W = torch.reshape(W, (self.N,self.T,self.T)) #N*T*T
                # 完成AX
                Hnow=torch.einsum("nk,bdkt->bdnt", (A, Hbefore)) # batch*dmodel*N*T
                # 完成XW
                Hnow=torch.einsum("bdni,nit->bdnt",(Hnow,W)) # batch*dmodel*N*T
                Hnow=torch.sigmoid(X+Hnow)*torch.tanh(X+Hnow)
                H.append(Hnow)
                Hbefore = Hnow
            H = torch.cat(H, dim=3)  # batch*dmodel*N*(T*(hops+1))
            Hout = self.gcnLinear(H)  # batch*dmodel*N*T
            Hout = self.dropout(Hout) # batch*dmodel*N*T
        else:
            Hout=Hbefore
            for k in range(self.hops):
                Hout = torch.einsum("nk,bdkt->bdnt", (A, Hout))  # batch*N*T A*H
                Hout = self.tradGcnW[k](Hout)  # batch*dmodel*N*T A*H*W
                Hout = F.relu(Hout)  # relu(A*H*w)
            Hout = self.dropout(Hout)  # batch*dmodel*N*T
        return Hout
