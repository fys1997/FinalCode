import torch.nn as nn


class ECL(nn.Module):
    def __init__(self, args, N):
        super(ECL, self).__init__()
        self.N = N
        self.D = args.dmodel
        self.ECLLinear = nn.Conv2d(in_channels=32, out_channels=self.D, kernel_size=(1,1), bias=True)

    def forward(self, EC):
        """
        EC: 边特性 [N*N*32]
        """
        EMC = self.ECLLinear(EC.transpose(0,2).unsqueeze(0))

        return EMC # 1*D*N*N

