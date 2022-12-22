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
        self.trainW1List = nn.ModuleList()
        self.trainW2List = nn.ModuleList()
        for i in range(hops):
            self.trainW1List.append(nn.Parameter(torch.randn(N,M).to(device),requires_grad=True).to(device))
            self.trainW2List.append(nn.Parameter(torch.randn(M,dmodel*dmodel).to(device),requires_grad=True).to(device))

        # 运用传统图卷积
        self.tradGcn = tradGcn
        self.gcnLinear = nn.Linear(dmodel * (self.hops + 1), dmodel)

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
        X = X.permute(0,3,2,1).contiguous() # batch*Tin*node*dmodel
        H.append(X)
        Hbefore = X  # X batch*Tin*node*dmodel
        # 开始图卷积部分
        for k in range(self.hops):
            # 开始生成对应的GCN矩阵增强W矩阵
            W = torch.mm(self.trainW1List[k],self.trainW2List[k]) # N*(dmodel*dmodel)
            W = torch.reshape(W, (self.N,self.dmodel,self.dmodel)) #N*dmodel*dmodel
            # 完成AX
            Hnow=torch.einsum("nk,btkd->btnd", (A, Hbefore)) # batch*Tin*node*dmodel
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
        return Hout
