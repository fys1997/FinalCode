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

        # 设置通过时间学习图矩阵的参数
        self.time2AdjLinear=nn.Linear(in_features=dmodel,out_features=N) # 学习邻接矩阵 batch*Tin*N*dmodel-> batch*Tin*N*N

        # 设置GCN增强的矩阵分解的维度
        self.trainW1=nn.Parameter(torch.randn(N,M).to(device),requires_grad=True).to(device) # N*M
        self.trainW2=nn.Parameter(torch.randn(M,dmodel*dmodel),requires_grad=True).to(device) # M*(dmodel*dmodel)

        # 设置动态邻接矩阵自学习层
        self.alpha1AdjLearning = nn.Parameter(torch.randn(N,M).to(device),requires_grad=True).to(device) # N*M
        self.alpha2AdjLearning = nn.Parameter(torch.randn(M,N).to(device),requires_grad=True).to(device) # M*N
        self.beta1AdjLearning = nn.Parameter(torch.randn(N,M).to(device),requires_grad=True).to(device) # N*M
        self.beta2AdjLearning = nn.Parameter(torch.randn(M,N).to(device),requires_grad=True).to(device) # M*N

        # 运用传统图卷积
        self.tradGcn = tradGcn
        if tradGcn:
            self.tradGcnW = nn.ModuleList()
            for i in range(self.hops):
                self.tradGcnW.append(nn.Linear(self.T, self.T))
        else:
            self.gcnLinear = nn.Linear(dmodel * (self.hops + 1), dmodel)

    def forward(self,X,timeEmbedding):
        """

        :param X: batch*dmodel*node*T
        :param timeEmbedding: batch*N*Tin*dmodel
        :return: Hout:[batch*dmodel*node*T]
        """
        # 开始动态学习生成拉普拉斯矩阵 A=Relu(alpha*sin(timeEmbedding)+beta)
        timeEmbedding=timeEmbedding.permute(0,2,1,3).contiguous() # batch*Tin*N*dmodel
        A = torch.sin(self.time2AdjLinear(timeEmbedding)) # batch*Tin*N*N 得到对应的邻接矩阵转换
        A = torch.einsum("nk,btnk->btnk",(torch.mm(self.alpha1AdjLearning,self.alpha2AdjLearning),A)) # batch*Tin*N*N
        A = A+torch.mm(self.beta1AdjLearning,self.beta2AdjLearning) # batch*Tin*N*N
        A = F.relu(A)

        H = list()
        X = X.permute(0,3,2,1).contiguous() # batch*Tin*node*dmodel
        H.append(X)
        Hbefore = X  # X batch*Tin*node*dmodel
        # 开始图卷积部分
        if self.tradGcn == False:
            # 开始生成对应的GCN矩阵增强W矩阵
            W = torch.mm(self.trainW1, self.trainW2)  # N*(dmodel*dmodel)
            W = torch.reshape(W, (self.N, self.dmodel, self.dmodel))  # N*dmodel*dmodel
            for k in range(self.hops):
                # 完成AX
                Hnow=torch.einsum("btnk,btkd->btnd", (A, Hbefore)) # batch*Tin*node*dmodel
                # 完成XW
                Hnow=Hnow.permute(0,2,1,3).contiguous() # bacth*N*Tin*dmodel
                Hnow=torch.einsum("bntk,nkd->bntd",(Hnow,W)) # batch*N*Tin*dmodel
                Hnow=Hnow.permute(0,2,1,3).contiguous() # batch*Tin*N*dmodel
                Hnow=torch.sigmoid(X+Hnow)*torch.tanh(X+Hnow)
                H.append(Hnow)
                Hbefore = Hnow
            H = torch.cat(H, dim=3)  # batch*Tin*N*(dmodel*(hops+1))
            Hout = self.gcnLinear(H)  # batch*Tin*N*dmodel
            Hout = self.dropout(Hout) # batch*Tin*N*dmodel
            Hout = Hout.permute(0,3,2,1).contiguous() # batch*dmodel*N*Tin
        else:
            Hout = Hbefore
            for k in range(self.hops):
                Hout = torch.einsum("nk,bdkt->bdnt", (A, Hout))  # batch*N*T A*H
                Hout = self.tradGcnW[k](Hout)  # batch*dmodel*N*T A*H*W
                Hout = F.relu(Hout)  # relu(A*H*w)
            Hout = self.dropout(Hout)  # batch*dmodel*N*T
        return Hout
