# 此处定义自己的模型
import torch.nn as nn
import model.GmanGcn as GG
import model.timeEmbedding as TE


class mixNet(nn.Module):
    def __init__(self, args, device, T, N, outputT):
        '''

        :param args: 一些设置的参数
        :param data: 训练的数据
        '''
        super(mixNet, self).__init__()

        # 设置时间embedding与GCN matrix的学习
        self.timeEmbed = TE.timeEmbedding(num_embedding=args.num_embedding, embedding_dim=args.dmodel,
                                          dropout=args.dropout)

        self.GcnAtteNet = GG.GcnAtteNet(N=N, Tin=T, Tout=outputT, args=args)

    def forward(self, X, Y, teacher_forcing_ratio):
        """

        :param X: 输入数据，X:batch*node*T*2
        :param Y: 真实值，Y:batch*node*outputT*2
        :return: 输出数据: Y:batch*node*T
        """
        vx = X[..., 0]  # batch*node*Tin 表示X车流量
        tx = X[:, 0, :, 1]  # batch*Tin 表示输入X的时间index
        ty = Y[:, 0, :, 1]  # batch*Tout 表示Y的时间index
        # 取得TimeEmbedding值
        tx = self.timeEmbed(tx)  # batch*Tin*dmodel
        ty = self.timeEmbed(ty)  # batch*Tout*dmodel

        # 开始encoder
        result = self.GcnAtteNet(vx, tx, ty)
        # result=torch.cat([result,vx[...,-self.arSize:]],dim=2)
        # result=self.predict(result)

        return result
