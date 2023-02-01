import torch
import torch.nn as nn
import pandas as pd
import numpy as np


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

        if N == 207:
            df = pd.read_csv(self.location_file, header=None, usecols=[2, 3])
            df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
            self.locations = torch.from_numpy(df.values[1:]).to(self.device).to(torch.float32)  # N*2 (经纬度)
            mean = self.locations.mean()
            std = self.locations.std()
            self.locations = (self.locations - mean) / std
        else:
            df = pd.read_csv('data/graph_sensor_locations_bay.csv', header=None, usecols=[1, 2])
            df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
            self.locations = torch.from_numpy(df.values).to(torch.float32).to(self.device) # N*2 (经纬度)
            mean = self.locations.mean()
            std = self.locations.std()
            self.locations = (self.locations - mean) / std

        self.spatialEmbed = np.loadtxt(args.distance_file, skiprows=1)
        self.spatialEmbed = self.spatialEmbed[self.spatialEmbed[..., 0].argsort()]
        self.spatialEmbed = torch.from_numpy(self.spatialEmbed[..., 1:]).float().to(
            args.device)  # 对应文件的space embed [N*64]
        mean = self.spatialEmbed.mean()
        std = self.spatialEmbed.std()
        self.spatialEmbed = (self.spatialEmbed - mean) / std
        self.spatialEmbedLinear = nn.Linear(in_features=64, out_features=self.dmodel)

        self.locationLinear = nn.Linear(in_features=2, out_features=self.dmodel)
        self.KLinear = nn.Linear(in_features=1, out_features=self.d_keys * self.num_heads)
        self.QLinear = nn.Linear(in_features=1, out_features=self.d_keys * self.num_heads)
        self.VLinear = nn.Linear(in_features=1, out_features=self.d_keys * self.num_heads)
        self.outLinear = nn.Linear(in_features=1, out_features=self.d_keys * self.num_heads // 2)

        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, t):
        """
        tX: 时间嵌入向量 batch*T*dmodel
        return：
        K (N*2dmodel*(dkeys*num_heads))
        Q (N*2dmodel*(dkeys*num_heads))
        V (N*2dmodel*(dkeys*num_heads))
        out_pro (N*(dkeys*num_heads)*dmodel)
        """

        locations = self.dropout(self.locationLinear(self.locations))  # N*dmodel
        spatialEmbed = self.dropout(self.spatialEmbedLinear(self.spatialEmbed))  # N*dmodel
        metaInfo = torch.cat((locations, spatialEmbed), 0)  # 2N*(dmodel)
        metaInfo = torch.reshape(metaInfo, (2 * self.N * self.dmodel, 1))  # (2N*dmodel)*1

        K = torch.tanh(self.KLinear(metaInfo)) # (2N*dmodel)*(d_keys*num_heads)
        K = torch.reshape(K, (self.N, 2 * self.dmodel, -1))  # N*2dmodel*(dKeys*num_heads)
        Q = torch.tanh(self.dropout(self.QLinear(metaInfo)))
        Q = torch.reshape(Q, (self.N, 2 * self.dmodel, -1))
        V = torch.tanh(self.dropout(self.VLinear(metaInfo)))
        V = torch.reshape(V, (self.N, 2 * self.dmodel, -1))
        out = torch.tanh(self.dropout(self.outLinear(metaInfo)))  # N*(dmodel*dkeys*num_heads)
        out = torch.reshape(out, (self.N, self.d_keys * self.num_heads, -1))  # N*(dkeys*num_heads)*dmodel

        return K, Q, V, out
