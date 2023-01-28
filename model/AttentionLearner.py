import torch
import torch.nn as nn
import pandas as pd


class AttentionMeta(nn.Module):
    def __init__(self, args, N, T):
        """
        args: 输入参数
        N: sensor数量
        """
        super(AttentionMeta, self).__init__()
        self.location_file = args.location_file
        self.N = N
        self.num_heads = args.head
        self.dmodel = args.dmodel
        self.d_keys = 2 * self.dmodel // self.num_heads
        self.device = args.device
        self.T = T

        df = pd.read_csv(self.location_file, header=None, usecols=[2, 3])
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
        self.locations = torch.from_numpy(df.values[1:]).to(self.device).to(torch.float32)  # N*2 (经纬度)

        self.locationLinear = nn.Linear(in_features=2, out_features=self.dmodel)
        self.KLinear1 = nn.Linear(in_features=T + N, out_features=N)
        self.KLinear2 = nn.Linear(in_features=1, out_features=2*self.d_keys*self.num_heads*T)
        self.QLinear1 = nn.Linear(in_features=T + N, out_features=N)
        self.QLinear2 = nn.Linear(in_features=1, out_features=2 * self.d_keys * self.num_heads * T)
        self.VLinear1 = nn.Linear(in_features=T + N, out_features=N)
        self.VLinear2 = nn.Linear(in_features=1, out_features=2 * self.d_keys * self.num_heads * T)

    def forward(self, t):
        """
        tX: 时间嵌入向量 batch*T*dmodel
        return：
        K (N*T*2dmodel*(dkeys*num_heads))
        Q (N*T*2dmodel*(dkeys*num_heads))
        V (N*T*2dmodel*(dkeys*num_heads))
        out_pro 可能存在
        """

        locations = self.locationLinear(self.locations)  # N*dmodel
        batch = t.shape[0]
        locations = locations.unsqueeze(0)  # 1*N*dmodel
        locations = locations.repeat(batch, 1, 1)  # batch*N*dmodel
        metaInfo = torch.cat((t, locations), 1)  # batch*(T+N)*dmodel
        metaInfo = metaInfo.permute(0, 2, 1).contiguous()  # batch*dmodel*(T+N)

        K = self.KLinear1(metaInfo)  # batch*dmodel*N
        K = torch.reshape(K,(batch,self.dmodel*self.N,1)) # batch*(dmodel*N)*1
        K = self.KLinear2(K) # batch*(dmodel*N)*(2*d_keys*num_heads*T)
        K = torch.reshape(K,(batch,self.N,self.T,2*self.dmodel,-1)) # batch*N*T*2dmodel*(d_keys*num_heads)

        Q = self.QLinear1(metaInfo)  # batch*dmodel*N
        Q = torch.reshape(Q, (batch, self.dmodel * self.N, 1))  # batch*(dmodel*N)*1
        Q = self.QLinear2(Q)  # batch*(dmodel*N)*(2*d_keys*num_heads*T)
        Q = torch.reshape(Q, (batch, self.N, self.T, 2 * self.dmodel, -1))  # batch*N*T*2dmodel*(d_keys*num_heads)

        V = self.VLinear1(metaInfo)  # batch*dmodel*N
        V = torch.reshape(V, (batch, self.dmodel * self.N, 1))  # batch*(dmodel*N)*1
        V = self.VLinear2(V)  # batch*(dmodel*N)*(2*d_keys*num_heads*T)
        V = torch.reshape(V, (batch, self.N, self.T, 2 * self.dmodel, -1))  # batch*N*T*2dmodel*(d_keys*num_heads)
        return K, Q, V
