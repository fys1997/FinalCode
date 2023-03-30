import torch.nn as nn


class DLinearLearner(nn.Module):
    def __init__(self, args, N, T):
        super(DLinearLearner, self).__init__()
        self.N = N
        self.T = T
        self.D = args.dmodel

        self.WSeasonal = nn.Linear(in_features=self.D, out_features=T*T)
        self.WTrend = nn.Linear(in_features=self.D, out_features=T*T)

        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, NMC):
        """
        NMC [N*D]
        """
        WSeasonal = self.dropout(self.WSeasonal(NMC)) # n*(T*T)
        WSeasonal = WSeasonal.view(self.N, self.T, self.T)

        WTrend = self.dropout(self.WTrend(NMC)) # N*(T*T)
        WTrend = WTrend.view(self.N, self.T,self.T)

        return WSeasonal, WTrend
