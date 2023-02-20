import torch.nn as nn


# Node Meta Characteristics Learner 节点元特性学习器
class NCL(nn.Module):
    def __init__(self, args, N):
        super(NCL, self).__init__()
        self.N = N
        self.D = args.dmodel
        self.NCLLinear = nn.Linear(in_features=989, out_features=self.D, bias=True)
        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, NC):
        """
        NC: N*989
        return: NMC [N*D]
        """
        return self.dropout(self.NCLLinear(NC))  # N*D
