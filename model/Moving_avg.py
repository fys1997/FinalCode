import torch
import torch.nn as nn


class Moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(Moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self,x):
        """
        x:[B*N,T,D]
        """
        front = x[:,0:1,:].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0,2,1).contiguous()) # BN*D*T
        x = x.permute(0, 2, 1).contiguous() # BN*T*D
        return x