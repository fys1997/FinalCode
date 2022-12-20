import torch
import torch.nn as nn


class timeEmbedding(nn.Module):
    def __init__(self, num_embedding, embedding_dim,dropout):
        super(timeEmbedding, self).__init__()
        self.t2vLinear=nn.Linear(in_features=1,out_features=1,bias=True)
        self.t2vSin=nn.Linear(in_features=1,out_features=embedding_dim-1,bias=True)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,x):
        """

        :param x: 时间：[batch*N*Tin]
        :return:batch*N*T*embedding_dim
        """
        x=x.long()
        x=x.unsqueeze(dim=3) # batch*N*Tin*1
        # 作normalize，给所有数除以一个最大值
        x=torch.div(x,288)
        x1=self.t2vLinear(x) # batch*N*Tin*1
        x2=torch.sin(self.t2vSin(x)) # batch*N*Tin*(dmodel-1)
        x=torch.cat((x1,x2),dim=3) # batch*N*Tin*dmodel
        return self.dropout(x)
