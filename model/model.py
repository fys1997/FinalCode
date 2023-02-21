# 此处定义自己的模型
import torch.nn as nn
import model.GmanGcn as GG


class mixNet(nn.Module):
    def __init__(self, args, device, T, N, outputT):
        '''

        :param args: 一些设置的参数
        :param device: GPU or CPU
        '''
        super(mixNet, self).__init__()

        self.GcnAtteNet = GG.GcnAtteNet(N=N, Tin=T, Tout=outputT, args=args)

    def forward(self, X):
        """

        :param X: 输入数据，X:batch*node*T*2
        :return: 输出数据: Y:batch*node*T*2
        """
        result = self.GcnAtteNet(X)

        return result
