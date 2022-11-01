import util
import argparse
import numpy as np
import pandas as pd
import torch


parser=argparse.ArgumentParser()
parser.add_argument('--M',type=int,default=10,help='preTrained matrix dimensions')
parser.add_argument('--device',type=str,default='cuda:0',help='GPU cuda')
parser.add_argument('--hops,',type=int,default=10,help='GCN hops')
parser.add_argument('--arSize',type=int,default=6,help='AutoRegressive window')
parser.add_argument('--dropout',type=float,default=0.1,help='dropout')
parser.add_argument('--head',type=int,default=8,help='the multihead count of transformer')
parser.add_argument('--transformerLayers',type=int,default=1,help='the layer count of transformerEncode')
parser.add_argument('--lrate',type=float,default=0.001,help='learning rate')
parser.add_argument('--wdeacy',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--data',type=str,default='data/METR-LA-12/',help='data path')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--epochs',type=int,default=20,help='')
parser.add_argument('--print_every',type=int,default=100,help='')
parser.add_argument('--save',type=str,default='modelSave/metr-12.pt',help='save path')
parser.add_argument('--tradGcn',type=bool,default=False,help='whether use tradGcn')
parser.add_argument('--horzion',type=int,default=12,help='output sequenth length')
args=parser.parse_args()


def main():
    device=torch.device(args.device if torch.cuda.is_available() else "cpu")

    # load the best saved model
    with open(args.save,'rb') as f:
        model=torch.load(f)
        model.to(device)
        model.eval()
        print("model load successfully")
    dataloader=util.load_dataset(args.data,args.batch_size,args.batch_size,args.batch_size)
    scaler=dataloader['scaler']

    mae=[]
    mape=[]
    rmse=[]
    for iter,(x,y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx=torch.Tensor(x).to(device) # batch*T*N*2
        testy=torch.Tensor(y).to(device) #batch*T*N*2
        with torch.no_grad():
            preds=model(testx.permute(0,2,1,3).contiguous(),testy.permute(0,2,1,3).contiguous(),0).permute(0,2,1).contiguous()
        pred=scaler.inverse_transform(preds) #batch*T*N
        metrics=util.metric(pred,testy[:,:,:,0])
        mae.append(metrics[0])
        mape.append(metrics[1])
        rmse.append(metrics[2])
    log='Test average loss: {:.4f} Test average mape: {:.4f} Test average rmse: {:.4f}'
    print(log.format(np.mean(mae),np.mean(mape),np.mean(rmse)),flush=True)

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device) # batch_size*T*N*2

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device) # batch*T*N*2
        testy=torch.Tensor(y).to(device) # batch*outputT*N*2
        with torch.no_grad():
            preds = model(testx.permute(0,2,1,3).contiguous(),testy.permute(0,2,1,3).contiguous(),0).permute(0,2,1).contiguous() # batch*T*N
        outputs.append(preds)

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...] #batch*T*N

    amae = []
    amape = []
    armse = []
    for i in range(args.horzion):
        pred = scaler.inverse_transform(yhat[:, i, :])
        real = realy[:, i, :,0]
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(np.mean(amae), np.mean(amape), np.mean(armse)))


if __name__== "__main__":
    main()


