import torch
import torch.nn as nn
import pandas as pd
import numpy as np


class KQVLinear(nn.Module):
    def __init__(self, args, N, T):
        """
        args: 输入参数
        N: sensor数量
        """
        super(KQVLinear, self).__init__()
        self.N = N
        self.T = T
        self.D = args.dmodel
        self.num_heads = args.head
        self.d_keys = self.D//self.num_heads

        self.WkLinear = nn.Linear(in_features=self.D, out_features=self.D*self.d_keys*self.num_heads)
        self.bkLinear = nn.Linear(in_features=self.D, out_features=self.d_keys*self.num_heads)

        self.WqLinear = nn.Linear(in_features=self.D, out_features=self.D*self.d_keys*self.num_heads)
        self.bqLinear = nn.Linear(in_features=self.D, out_features=self.d_keys*self.num_heads)

        self.WvLinear = nn.Linear(in_features=self.D, out_features=self.D*self.d_keys*self.num_heads)
        self.bvLinear = nn.Linear(in_features=self.D, out_features=self.d_keys*self.num_heads)

        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, NMC):
        """
        NMC : 节点元特性 [N*D]
        return：
        Wk (N*D*(dkeys*num_heads))
        bk (N*(dkeys*num_heads))

        Wq (N*D*(dkeys*num_heads))
        bq (N*(dkeys*num_heads))

        Wv (N*D*(dkeys*num_heads))
        bv (N*(dkeys*num_heads))
        """
        Wk = self.dropout(self.WkLinear(NMC))
        Wk = torch.reshape(Wk,(self.N,self.D,-1)) # N*D*(dkeys*num_heads)
        bk = self.dropout(self.bkLinear(NMC)) # N*(dkeys*num_heads)

        Wq = self.dropout(self.WqLinear(NMC))
        Wq = torch.reshape(Wq,(self.N,self.D,-1)) # N*D*(dkeys*num_heads)
        bq = self.dropout(self.bqLinear(NMC)) # N*(dkeys*num_heads)

        Wv = self.dropout(self.WvLinear(NMC))
        Wv = torch.reshape(Wv,(self.N,self.D,-1)) # N*D*(dkeys*num_heads)
        bv = self.dropout(self.bvLinear(NMC)) # N*(dkeys*num_heads)
        return Wk,bk,Wq,bq,Wv,bv
