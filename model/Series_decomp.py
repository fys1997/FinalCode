import torch.nn as nn
from model.Moving_avg import Moving_avg


class Series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(Series_decomp, self).__init__()
        self.moving_avg = Moving_avg(kernel_size=kernel_size,stride=1)

    def forward(self,x):
        """
        x [B*N,T,D]
        """
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean