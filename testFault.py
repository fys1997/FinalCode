import util
import argparse
import numpy as np
import pandas as pd
import torch
from model.model import mixNet
import torch.nn.functional as F


parser=argparse.ArgumentParser()
parser.add_argument('--M',type=int,default=10,help='GCN matrix W dimensions')
parser.add_argument('--device',type=str,default='cuda:0',help='GPU cuda')
parser.add_argument('--hops',type=int,default=4,help='GCN hops')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout')
parser.add_argument('--head',type=int,default=8,help='the multihead count of attention')
parser.add_argument('--lrate',type=float,default=0.001,help='learning rate')
parser.add_argument('--wdeacy',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--data',type=str,default='data/METR-LA-12/',help='data path')
parser.add_argument('--batch_size',type=int,default=1,help='batch size')
parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=100,help='')
parser.add_argument('--save',type=str,default='modelSave/metr.pth',help='save path')
parser.add_argument('--tradGcn',type=bool,default=False,help='whether use tradGcn')
parser.add_argument('--dmodel',type=int,default=64,help='transformerEncoder dmodel')
parser.add_argument('--num_embedding',type=int,default=288,help='')
parser.add_argument('--encoderBlocks',type=int,default=2,help=' encoder block numbers')
parser.add_argument('--preTrain',type=bool,default=False,help='whether use preTrain model')
parser.add_argument('--seed',type=int,default=3407,help='random seed')
parser.add_argument('--location_file',type=str,default='data/graph_sensor_locations.csv',help='sensor location csv')
parser.add_argument('--distance_file',type=str,default='data/sensor_graph/SE(METR).txt',help='distance adjacency matrix file')
parser.add_argument('--clip',type=int,default=8,help='gradient norm clip')
parser.add_argument('--decoderBlocks',type=int,default=2,help='decoder block number')
parser.add_argument('--onlyEC', type=bool, default=False, help="only EMC used in adj")
parser.add_argument('--onlyAdp', type=bool, default=False, help="only Adp use in adj")
parser.add_argument('--normalTA', type=bool, default=False, help="normal attention or meta Attention")
args=parser.parse_args()


def main():
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    args.device=torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(args)
    # load the best saved model
    T = dataloader['T']
    N = dataloader['N']
    outputT = dataloader['outputT']
    model=mixNet(args=args,device=args.device,T=T,N=N,outputT=outputT)
    model.load_state_dict(torch.load(args.save),strict=True)
    model.eval()
    model.to(args.device)
    print("model load successfully")
    for param_tensor in model.state_dict():
        print(param_tensor,'\t',model.state_dict()[param_tensor].size())

    scaler=dataloader['scaler']

    mae=[]
    mape=[]
    rmse=[]
    for iter,(x,y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx=F.dropout3d(torch.Tensor(x).to(args.device),p=args.dropout) # batch*T*N*2
        testy=torch.Tensor(y).to(args.device) #batch*T*N*2
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
    realy = torch.Tensor(dataloader['y_test']).to(args.device) # batch_size*T*N*2

    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = F.dropout3d(torch.Tensor(x).to(args.device),p=args.dropout) # batch*T*N*2
        testy=torch.Tensor(y).to(args.device) # batch*outputT*N*2
        with torch.no_grad():
            preds = model(testx.permute(0,2,1,3).contiguous(),testy.permute(0,2,1,3).contiguous(),0).permute(0,2,1).contiguous() # batch*T*N
        outputs.append(preds)

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...] #batch*T*N

    amae = []
    amape = []
    armse = []
    for i in range(outputT):
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


