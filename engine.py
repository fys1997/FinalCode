import torch.nn.utils
import torch.optim as optim
from model.model import mixNet
import util
import torch.nn as nn


class trainer():
    def __init__(self, device, args, scaler, T, N, outputT):

        # 加载模型，根据preTrain决定加载已经训练好的best model或是重新开始训练模型
        if args.preTrain is not None and args.preTrain:
            model = mixNet(args=args, device=args.device, T=T, N=N, outputT=outputT)
            model.to(device)
            model.load_state_dict(torch.load(args.save), strict=True)
            model.eval()
            print("model load successfully")
        else:
            self.model = mixNet(args, device, T, N, outputT)
            self.model.to(device)
        # self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lrate, weight_decay=args.wdeacy)
        self.loss = util.masked_mae
        self.scaler = scaler

    def train(self, X, real_val):
        """

        :param X: 输入:batch*T*N*2
        :param real_val: 输入:batch*outputT*N*2
        :return:
        """
        self.model.train()
        self.optimizer.zero_grad()

        X = X.permute(0, 2, 1, 3).contiguous()  # batch*N*T*2
        Y = real_val.permute(0, 2, 1, 3).contiguous()  # batch*N*outputT*2

        output = self.model(X, Y, 0.5)  # batch*N*T
        output = output.permute(0, 2, 1).contiguous()  # batch*T*N
        predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real_val[:, :, :, 0], 0.0)
        loss.backward()
        # if self.clip is not None:
        #     torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clip)
        self.optimizer.step()
        mape = util.masked_mape(predict, real_val[:, :, :, 0], 0.0).item()
        rmse = util.masked_rmse(predict, real_val[:, :, :, 0], 0.0).item()
        return loss.item(), mape, rmse

    def eval(self, X, real_val):
        self.model.eval()
        X = X.permute(0, 2, 1, 3).contiguous()  # batch*N*T*2
        Y = real_val.permute(0, 2, 1, 3).contiguous()  # batch*N*outputT*2

        output = self.model(X, Y, 0)
        output = output.permute(0, 2, 1).contiguous()  # batch*T*N
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real_val[:, :, :, 0], 0.0)
        mape = util.masked_mape(predict, real_val[:, :, :, 0], 0.0).item()
        rmse = util.masked_rmse(predict, real_val[:, :, :, 0], 0.0).item()
        return loss.item(), mape, rmse
