import torch.nn.utils
import torch.optim as optim
from model.model import mixNet
import util
import torch.nn as nn


class trainer():
    def __init__(self,device,args,scaler,T,N,outputT):
        #self.data=data.permute(0,2,1).contiguous() #batch*node*T
        if args.preTrain is not None and args.preTrain:
            with open(args.save, 'rb') as f:
                self.model = torch.load(f)
                self.model.to(device)
                print("model load successfully")
        else:
            self.model=mixNet(args,device,T,N,outputT)
            self.model=nn.DataParallel(self.model.cuda(),device_ids=[0,1,2,3])
        # self.model.to(device)
        self.optimizer=optim.Adam(self.model.module.parameters(),lr=args.lrate,weight_decay=args.wdeacy)
        self.loss=util.masked_mae
        self.scaler=scaler
        self.clip=5

        self.schedule=optim.lr_scheduler.StepLR(self.optimizer,step_size=5,gamma=0.5)

    def train(self,X,real_val):
        """

        :param X: 输入:batch*T*N*2
        :param real_val: 输入:batch*outputT*N*2
        :return:
        """
        self.model.train()
        self.optimizer.zero_grad()

        X=X.permute(0,2,1,3).contiguous() # batch*N*T*2
        Y=real_val.permute(0,2,1,3).contiguous() # batch*N*outputT*2

        output=self.model(X,Y,0.5) # batch*N*T
        output=output.permute(0,2,1).contiguous() # batch*T*N
        predict=self.scaler.inverse_transform(output)

        loss=self.loss(predict,real_val[:,:,:,0],0.0)
        loss.backward()
        # if self.clip is not None:
        #     torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clip)
        self.optimizer.step()
        mape=util.masked_mape(predict,real_val[:,:,:,0],0.0).item()
        rmse=util.masked_rmse(predict,real_val[:,:,:,0],0.0).item()
        return loss.item(),mape,rmse

    def eval(self,X,real_val):
        self.model.eval()
        X=X.permute(0,2,1,3).contiguous() # batch*N*T*2
        Y = real_val.permute(0,2,1,3).contiguous()  # batch*N*outputT*2

        output=self.model(X,Y,0)
        output = output.permute(0, 2, 1).contiguous()  # batch*T*N
        predict=self.scaler.inverse_transform(output)
        loss=self.loss(predict,real_val[:,:,:,0],0.0)
        mape=util.masked_mape(predict,real_val[:,:,:,0],0.0).item()
        rmse=util.masked_rmse(predict,real_val[:,:,:,0],0.0).item()
        return loss.item(),mape,rmse

    def adjust_lr(self,i,epochs):
        if(i%epochs==0):
            for param_group in self.optimizer.param_groups:
                param_group["lr"]/=2

