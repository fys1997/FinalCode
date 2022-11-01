
import torch.nn as nn


class timeEmbedding(nn.Module):
    def __init__(self, num_embedding, embedding_dim,dropout):
        super(timeEmbedding, self).__init__()
        self.embed = nn.Embedding(num_embeddings=num_embedding, embedding_dim=embedding_dim)
        self.dropout=nn.Dropout(p=dropout)

    def forward(self,x):
        """

        :param x: 时间：[batch*T]
        :return:batch*T*embedding_dim
        """
        x=x.long()
        x=self.embed(x)
        return self.dropout(x)
