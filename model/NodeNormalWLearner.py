import torch.nn as nn


class NodeNormalWLearner(nn.Module):
    def __init__(self, args, N, in_features, out_features):
        super(NodeNormalWLearner, self).__init__()
        self.N = N
        self.in_features = in_features
        self.out_features = out_features
        self.WLearner = nn.Linear(in_features=args.dmodel, out_features=in_features*out_features, bias=True)
        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, NMC):
        """
        NMC : [N*D]
        """
        W = self.dropout(self.WLearner(NMC)) # N*(in_features*out_features)
        W = W.view(self.N, self.in_features, -1) # N*in_features*out_features
        return W
