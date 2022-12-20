import torch
import torch.nn as nn
import model.PositionEmbedding as PE
import model.timeEmbedding as TE

# no usage

class Transformer(nn.Module):
    def __init__(self, dmodel, cnn_in_channels, cnn_out_channels, nhead,
                 num_layers, dropout, device,Tout,Tin,num_embedding):
        super().__init__()
        self.transformerCNN1 = nn.Conv1d(in_channels=cnn_in_channels, out_channels=cnn_out_channels, kernel_size=1)
        self.transformerCNN2 = nn.Conv1d(in_channels=cnn_out_channels, out_channels=cnn_in_channels, kernel_size=1)
        # encoder
        self.encoderLayer = nn.TransformerEncoderLayer(d_model=dmodel, nhead=nhead, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoderLayer, num_layers=num_layers)
        # decoder
        self.decoderLayer = nn.TransformerDecoderLayer(d_model=dmodel, nhead=nhead, dropout=dropout)
        self.decoder = nn.TransformerDecoder(self.decoderLayer, num_layers=num_layers)
        self.device = device
        self.positionEmbedding = PE.PositionalEncoding(d_model=dmodel, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.Tout=Tout
        # 输出层
        self.projection=nn.Linear(Tin,Tout)
        # timeEmbedding
        self.timeEmbed=TE.timeEmbedding(num_embedding=num_embedding,embedding_dim=dmodel,dropout=dropout)

    def forward(self, X, Y,tx,ty, teacher_forcing_ratio):
        """

        :param X: [batch*N*Tin]
        :param Y: [batch*N*Tout]
        :param tx: [batch*Tin] x时间index
        :param ty: [batch*Tout] y时间index
        :param teacher_forcing_ratio: decide train or val
        :return:
        """
        X=X.permute(2,1,0).contiguous() # Tin*N*batch
        X = self.transformerCNN1(X)  # Tin*dmodel*batch
        X=X.permute(0,2,1).contiguous() # Tin*batch*dmodel

        xin = self.positionEmbedding(X)  # Tin*batch*dmodel
        tx=self.timeEmbed(tx) # bacth*Tin*dmodel
        xin=xin+tx.permute(1,0,2).contiguous()
        encoder_output = self.encoder(xin) # Tin*batch*dmodel

        # decoder 一步预测
        # ty=self.timeEmbed(ty) # batch*Tout*dmodel
        # ty=ty.permute(1,0,2).contiguous() # Tout*batch*dmodel
        # decoder_input = torch.cat([encoder_output,ty],dim=0) # (Tin+Tout)*batch*dmodel
        # target_len = Y.shape[0]
        # Y=self.positionEmbedding(Y)
        # tgt_mask = nn.Transformer().generate_square_subsequent_mask(target_len).to(self.device)
        # decoder_output=self.decoder(tgt=Y,tgt_mask=tgt_mask,memory=encoder_output)
        decoder_output=self.projection(encoder_output.permute(1,2,0).contiguous()) # batch*dmodel*(Tin+Tout)


        decoder_output = self.transformerCNN2(decoder_output)  # batch*N*(Tin+Tout)
        return decoder_output # batch*N*Tout
