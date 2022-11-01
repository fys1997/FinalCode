import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self,T, trainMatrix1, trainMatrix2, device, hops,dmodel, tradGcn=False, dropout=0.1):
        super().__init__()
        self.T=T
        self.trainMatrix1 = trainMatrix1
        self.trainMatrix2 = trainMatrix2
        self.device = device
        self.hops = hops
        self.tradGcn = tradGcn
        self.dropout = nn.Dropout(p=dropout)
        # # 门
        # self.gate=nn.Linear(in_features=2*T,out_features=T)
        # # 设置batchnorm层
        # self.bn=nn.ModuleList()
        # for i in range(hops):
        #     self.bn.append(nn.BatchNorm2d(num_features=dmodel))
        # 运用传统图卷积
        self.tradGcn = tradGcn
        if tradGcn:
            self.tradGcnW = nn.ModuleList()
            for i in range(self.hops):
                self.tradGcnW.append(nn.Linear(self.T, self.T))
        else:
            self.gcnLinear = nn.Linear(self.T * (self.hops + 1), self.T)
        self.pLow=nn.Parameter(torch.rand(1).to(device)).to(device)
        self.aLow=nn.Parameter(torch.rand(1).to(device)).to(device)
        self.pMid = nn.Parameter(torch.rand(1).to(device)).to(device)
        self.aMid = nn.Parameter(torch.rand(1).to(device)).to(device)
        self.pHigh = nn.Parameter(torch.rand(1).to(device)).to(device)
        self.aHigh = nn.Parameter(torch.rand(1).to(device)).to(device)

    def forward(self,X):
        """

        :param X: batch*dmodel*node*T
        :return: Hout:[batch*dmodel*node*T]
        """
        adjMat = F.relu(torch.mm(self.trainMatrix1, self.trainMatrix2))
        adjMat = F.softmax(adjMat, dim=1)
        A=adjMat
        I=torch.eye(A.shape[0]).cuda()

        H = list()
        H.append(X)
        Hbefore = X  # X batches*dmodel*node*T
        # 开始图卷积部分
        if self.tradGcn == False:
            for k in range(self.hops):
                # low filter nk,bdkt->bdnt
                HLow=torch.einsum("nk,bdkt->bdnt", (self.pLow*(self.aLow*A+(1-self.aLow)*I), Hbefore))
                HMid=torch.einsum("nk,bdkt->bdnt", (self.pMid*(torch.pow(A,2)-self.aMid*I), Hbefore))
                HHigh=torch.einsum("nk,bdkt->bdnt", (self.pHigh*(-self.aHigh*A+(1-self.aHigh)*I), Hbefore))
                Hnow=torch.sigmoid(HLow*torch.sigmoid(HHigh+HMid)+HMid*torch.sigmoid(HLow+HHigh)+HHigh*torch.sigmoid(HLow+HMid))
                # Hnow=torch.einsum("nk,bdkt->bdnt", (A, Hbefore)) # batch*dmodel*node*T
                # gateInput=torch.cat([X,Hnow],dim=3) # batch*dmodel*node*2T
                # z=torch.sigmoid(self.bn[k](self.gate(gateInput))) # batch*dmodel*node*T
                # Hnow=z*Hnow+(1-z)*X # batch*dmodel*node*T
                Hnow=torch.sigmoid(X+Hnow)*torch.tanh(X+Hnow)
                H.append(Hnow)
                Hbefore = Hnow
            H = torch.cat(H, dim=3)  # batch*dmodel*N*(T*(hops+1))
            Hout = self.gcnLinear(H)  # batch*dmodel*N*T
            Hout = self.dropout(Hout) # batch*dmodel*N*T
        else:
            Hout = Hbefore
            for k in range(self.hops):
                Hout = torch.einsum("nk,bdkt->bdnt", (A, Hout))  # batch*N*T A*H
                Hout = self.tradGcnW[k](Hout)  # batch*dmodel*N*T A*H*W
                Hout = F.relu(Hout)  # relu(A*H*w)
            Hout = self.dropout(Hout)  # batch*dmodel*N*T
        return Hout
