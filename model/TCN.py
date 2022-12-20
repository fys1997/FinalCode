import torch
import torch.nn as nn
import torch.nn.functional as F
import model.GCN as GCN


# no usage
class TCN(nn.Module):
    def __init__(self,Tin,Tout,trainMatrix1,trainMatrix2,channels,device, hops, tradGcn=False,
                 blocks=4,kerner_size=2,dropout=0.1,layers=2):
        """
        输入数据：[batch*N*T]
        :param Tin: 输入数据的time length
        :param Tout: 输出数据的time length
        :param blocks: TCN的隐藏层数
        :param kerner_size:
        :param dropout:
        """
        super(TCN, self).__init__()
        self.blocks=blocks
        self.layers=layers
        self.Tin=Tin
        self.Tout=Tout
        self.dropout=nn.Dropout(p=dropout)

        self.filter_convs=nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        # self.residual_cons=nn.ModuleList()
        # self.skip_convs=nn.ModuleList()
        self.gcn_convs=nn.ModuleList()
        self.bn=nn.ModuleList()
        receptive_filed=1
        T=Tin+1
        for b in range(blocks):
            additional_scope=kerner_size-1
            new_dilation=1
            for i in range(layers):
                self.filter_convs.append(nn.Conv1d(in_channels=channels,out_channels=channels,kernel_size=kerner_size,dilation=new_dilation))
                self.gate_convs.append(nn.Conv1d(in_channels=channels,out_channels=channels,kernel_size=kerner_size,dilation=new_dilation))
                # self.residual_convs.append(nn.Conv1d(in_channels=channels,out_channels=channels,kernel_size=1))
                # self.skip_convs.append(nn.Conv1d(in_channels=channels,out_channels=channels,kernel_size=1))
                self.bn.append(nn.BatchNorm2d(1))
                T=T-kerner_size+1-new_dilation+1
                new_dilation*=2
                receptive_filed+=additional_scope
                additional_scope*=2
                self.gcn_convs.append(GCN.GCN(T=T,trainMatrix1=trainMatrix1,trainMatrix2=trainMatrix2,device=device,hops=hops,tradGcn=tradGcn,dropout=dropout))
        self.receptive_field=receptive_filed
        self.end_conv=nn.Conv1d(in_channels=1,out_channels=Tout,kernel_size=1,bias=True)

    def forward(self,input):
        """

        :param input: [batch*N*Tin]
        :return:
        """
        in_len=input.size(2)
        if in_len<self.receptive_field:
            x=nn.functional.pad(input,(self.receptive_field-in_len,0))
        else:
            x=input
        skip=0
        # TCN layers
        for i in range(self.blocks*self.layers):
            residual=x
            # dilated convolution
            filter=self.filter_convs[i](residual)
            filter=torch.tanh(filter)
            gate=self.gate_convs[i](residual)
            gate=torch.sigmoid(gate)
            x=filter*gate

            # skip connection
            s=x
            try:
                skip=skip[:,:,-s.size(2):]
            except:
                skip=0
            skip=s+skip
            x=self.gcn_convs[i](x)# T*batch*N
            x=x.permute(1,2,0).contiguous()
            x=x+residual[:,:,-x.size(2):]
            x=self.bn[i](x.unsqueeze(dim=1)).squeeze(dim=1)
        x=F.relu(skip) # skip:[batch*N*1]
        x=self.end_conv(x.permute(0,2,1).contiguous()) # batch*Tout*N
        return x.permute(0,2,1).contiguous()