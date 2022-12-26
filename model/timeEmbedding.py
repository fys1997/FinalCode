import torch
import torch.nn as nn


class timeEmbedding(nn.Module):
    def __init__(self, num_embedding, embedding_dim,dropout):
        super(timeEmbedding, self).__init__()
        self.embed = nn.Embedding(num_embeddings=num_embedding,embedding_dim=embedding_dim)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,x):
        """

        :param x: 时间：[batch*N*Tin]
        :return:batch*N*T*embedding_dim
        """
        x=x.long()
        x=self.embed(x) # batch*N*Tin*dmodel
        # x1=x[:,:,:,0:1] # batch*N*Tin*1
        # x2=torch.sin(x[:,:,:,1:]) # batch*N*Tin*(dmodel-1)
        # x=torch.cat((x1,x2),dim=3) # batch*N*Tin*dmodel
        return self.dropout(x)
