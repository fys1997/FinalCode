import torch
import numpy as np
import argparse
import util
import time
from engine import trainer

parser=argparse.ArgumentParser()
parser.add_argument('--M',type=int,default=10,help='GCN matrix W dimensions')
parser.add_argument('--device',type=str,default='cuda:0',help='GPU cuda')
parser.add_argument('--hops',type=int,default=4,help='GCN hops')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout')
parser.add_argument('--head',type=int,default=8,help='the multihead count of attention')
parser.add_argument('--lrate',type=float,default=0.001,help='learning rate')
parser.add_argument('--wdeacy',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--data',type=str,default='data/BJ-Flow/',help='data path')
parser.add_argument('--batch_size',type=int,default=1,help='batch size')
parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=1,help='')
parser.add_argument('--save',type=str,default='modelSave/metr.pth',help='save path')
parser.add_argument('--tradGcn',type=bool,default=False,help='whether use tradGcn')
parser.add_argument('--dmodel',type=int,default=64,help='transformerEncoder dmodel')
parser.add_argument('--encoderBlocks',type=int,default=2,help=' encoder block numbers')
parser.add_argument('--preTrain',type=bool,default=False,help='whether use preTrain model')
parser.add_argument('--seed',type=int,default=3407,help='random seed')
parser.add_argument('--clip',type=int,default=5,help='gradient norm clip')
parser.add_argument('--decoderBlocks',type=int,default=2,help='decoder block number')

args=parser.parse_args()


def main():
    args.device=torch.device(args.device if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    else:
        torch.manual_seed(args.seed)
    dataloader=util.load_dataset(args.data,args.batch_size,args.batch_size,args.batch_size)
    scaler=dataloader['scaler']
    T=dataloader['T']
    N=dataloader['N']
    outputT=dataloader['outputT']

    print(args)
    engine=trainer(device=args.device,args=args,scaler=scaler,T=T,N=N,outputT=outputT)
    print("start training...",flush=True)
    his_loss=[]
    val_time=[]
    train_time=[]
    # 加载预训练模型设置最小loss初始值
    if args.preTrain is not None and args.preTrain:
        loss=[]
        for iter,(x,y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx=torch.Tensor(x).to(args.device)
            testy=torch.Tensor(y).to(args.device)
            metrics=engine.eval(testx,testy)
            loss.append(metrics[0])
        best_valid_loss=np.mean(loss)
        print("preTrain best valid loss:",best_valid_loss)
    else:
        best_valid_loss = 10000000

    for param_tensor in engine.model.state_dict():
        print(param_tensor,'\t',engine.model.state_dict()[param_tensor].size())
    # 查看parameter 大小
    total_num = sum(p.numel() for p in engine.model.parameters())
    trainable_num = sum(p.numel() for p in engine.model.parameters())
    print('Total par num:{}, Trainable par num:{}'.format(total_num,trainable_num))
    # 开始训练
    for i in range(1,args.epochs+1):
        train_loss=[]
        train_mape=[]
        train_rmse=[]
        t1=time.time()
        dataloader['train_loader'].shuffle()
        for iter,(x,y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx=torch.Tensor(x).to(args.device)
            trainy=torch.Tensor(y).to(args.device)
            metrics=engine.train(trainx,trainy)
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every==0:
                log='Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter,train_loss[-1],train_mape[-1],train_rmse[-1]),flush=True)



        t2=time.time()
        train_time.append(t2-t1)


        valid_loss=[]
        valid_mape=[]
        valid_rmse=[]
        s1=time.time()
        for iter,(x,y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx=torch.Tensor(x).to(args.device)
            testy=torch.Tensor(y).to(args.device)
            metrics=engine.eval(testx,testy)
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2=time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss=np.mean(train_loss)
        mtrain_mape=np.mean(train_mape)
        mtrain_rmse=np.mean(train_rmse)

        mvalid_loss=np.mean(valid_loss)
        mvalid_mape=np.mean(valid_mape)
        mvalid_rmse=np.mean(valid_rmse)
        his_loss.append(mvalid_loss)
        if mvalid_loss<best_valid_loss:
            # 保存最佳模型
            torch.save(engine.model.state_dict(), args.save)
            print("best model saved")
            best_valid_loss = mvalid_loss

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),
              flush=True)
        #torch.save(engine.model.state_dict(),args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    # testing

    print("Training finished")
    print("The valid loss on best model is", str(round(best_valid_loss, 4)))


if __name__=="__main__":
    t1=time.time()
    main()
    t2=time.time()
    print("Total time spent: {:.4f}".format(t2-t1))

