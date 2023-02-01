import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, Tin, N, args):
        super().__init__()
        self.T = Tin  # 输入时间维度
        self.N = N  # 邻接矩阵边数 N*N
        self.dmodel = args.dmodel

        self.device = args.device
        self.hops = args.hops
        self.tradGcn = args.tradGcn
        self.dropout = nn.Dropout(p=args.dropout)

        # # 设置GCN增强的矩阵分解的维度 修改为nn.ParameterList
        # self.trainW1List = nn.ParameterList([nn.Parameter(torch.randn(N,args.M).to(args.device),requires_grad=True).to(args.device) for i in range(args.hops)]).to(args.device)
        # self.trainW2List = nn.ParameterList([nn.Parameter(torch.randn(args.M,Tin*Tin).to(args.device),requires_grad=True).to(args.device) for i in range(args.hops)]).to(args.device)
        # 运用传统图卷积
        self.tradGcn = args.tradGcn

        if self.tradGcn:
            self.tradGcnW = nn.ModuleList()
            for i in range(self.hops):
                self.tradGcnW.append(nn.Linear(self.T, self.T))
        else:
            self.gcnLinear = nn.Linear(Tin * (self.hops + 1), Tin)

    def forward(self, X, matrix):
        """

        :param X: batch*dmodel*node*T
        :param matrix: N*N
        :return: Hout:[batch*dmodel*node*T]
        """

        H = list()
        H.append(X)
        Hbefore = X  # X batch*dmodel*node*T
        # 开始图卷积部分
        if not self.tradGcn:
            for k in range(self.hops):
                # 开始生成对应的GCN矩阵增强W矩阵
                # W = torch.mm(self.trainW1List[k], self.trainW2List[k])  # N*(T*T)
                # W = torch.reshape(W, (self.N, self.T, self.T))  # N*T*T
                # 完成AX
                Hnow = torch.einsum("nk,bdkt->bdnt", (matrix, Hbefore))  # batch*dmodel*N*T
                # 完成XW
                # Hnow = torch.einsum("bdni,nit->bdnt", (Hnow, W))  # batch*dmodel*N*T
                Hnow = torch.sigmoid(X + Hnow) * torch.tanh(X + Hnow)
                H.append(Hnow)
                Hbefore = Hnow
            H = torch.cat(H, dim=3)  # batch*dmodel*N*(T*(hops+1))
            Hout = self.gcnLinear(H)  # batch*dmodel*N*T
            Hout = self.dropout(Hout)  # batch*dmodel*N*T
        else:
            Hout = Hbefore
            for k in range(self.hops):
                Hout = torch.einsum("nk,bdkt->bdnt", (matrix, Hout))  # batch*N*T A*H
                Hout = self.tradGcnW[k](Hout)  # batch*dmodel*N*T A*H*W
                Hout = F.relu(Hout)  # relu(A*H*w)
            Hout = self.dropout(Hout)  # batch*dmodel*N*T
        return Hout
