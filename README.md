# FinalCode
This is the pytorch code for my graduation paper of master's degree

python train.py --device cuda:2 --head 8 --data data/METR-LA-12/ --batch_size 32 --print_every 100 --save modelSave/MetaLearningONT/Adj/metr-ECAdp.pth --dmodel 64 --clip 5  --seed 1000 --onlyEC False --onlyAdp False --normalTA False