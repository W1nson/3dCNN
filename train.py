import h5py  
import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch 
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import Adam, RMSprop, lr_scheduler
import torch.nn as nn
from torchsummary import summary
import random
from model import model_3DCNN 
from data_util import * 
import argparse
from sklearn.metrics import f1_score, recall_score, precision_score


def set_seed(seed=42): 
    
    os.environ['PYTHONHASHSEED'] = str(seed) 
    random.seed(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed) 


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=False, action="store_true", help="training") 
    parser.add_argument("--eval", default=False, action="store_true", help="evaluating") 
    parser.add_argument("--test", default=False, action="store_true", help="testing")
    parser.add_argument("--seed", type=int, default=42, help="random seed") 
    parser.add_argument("--device-name", default="cuda:0", help="use cpu or cuda:0, cuda:1 ...")
    parser.add_argument("--data-dir", default=".", help="dataset directory")
    parser.add_argument("--model-path", default="./ckpts/best_ckpt.bin", help="model checkpoint file path")
    parser.add_argument("--output-path", default="./result", help="store the prediction files") 
    parser.add_argument("--epoch-count", type=int, default=50, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=50, help="mini-batch size")
    parser.add_argument("--learning-rate", type=float, default=0.0007, help="initial learning rate")
    parser.add_argument("--decay-rate", type=float, default=0.95, help="learning rate decay")
    parser.add_argument("--decay-iter", type=int, default=100, help="learning rate decay")
    parser.add_argument("--checkpoint-iter", type=int, default=50, help="checkpoint save rate")
    parser.add_argument("--verbose", type=int, default=0, help="print all input/output shapes or not")
    parser.add_argument("--sigma", type=int, default=0, help="sigma for gaussian filter") 
    args = parser.parse_args()
    

    # setting random seed 
    set_seed(args.seed) 

    # hyperparameters 
    # epoch_count = 5 #up epoch count once model is properly running
    # batch_size = 32
    # learning_rate = 0.0005
    # decay_rate = 0.95
    # decay_iter = 100
    # checkpoint_iter = 30
    # verbose = 0
    # sigma = 0 
    # model_path = './ckpts/best_ckpt.bin'

    use_cuda = torch.cuda.is_available()
    cuda_count = torch.cuda.device_count()
    device = torch.device('cpu') 
    if use_cuda: 
        device = torch.device(args.device_name)
    
    
    file_name = 'mpro_exp_data2_rdkit_feat.csv'
    df = pd.read_csv(os.path.join(args.data_dir, file_name))  # remember to change the path 
    df = df.drop(columns=['lib_name', 'Unnamed: 0'])


    df = df.drop_duplicates(subset="smiles", keep="first")


    
    # [16, 19, 48, 48, 48] for model summary 
    model = model_3DCNN(verbose=args.verbose, use_cuda=use_cuda)
     
    if use_cuda: 
        model.to(device)
    summary(model, (19, 48, 48, 48))
    
    # create new model just in case 
    model = model_3DCNN(verbose=args.verbose, use_cuda=use_cuda) 
    model.to(device) 


    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_iter, gamma=args.decay_rate)
    
    if args.train:
        
        print("loading training data") 
        train_data = Dataset_ligand(df=df, types='train', sigma=args.sigma)
        train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        
        if args.eval: 
            print("loading validation data") 
            val_data = Dataset_ligand(df=df, types='val', sigma=args.sigma) 
            val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True) 


        min_loss = float('inf')
        step = 0
        print("start training") 
        for epoch_ind in range(args.epoch_count):
            model.train()
            losses = []
            bar = tqdm(train_dataloader)
            avg_loss = 0 
            for batch_ind, batch in enumerate(bar):
                x, y = batch
                x_gpu, y_gpu = x.to(device), y.to(device)
                
        
                prob, ypred_batch, _ = model(x_gpu)
                # print(ypred_batch)
               
                y_pred = prob.argmax(1)  
                # loss_fn = nn.MSELoss().float()
                loss_fn = nn.CrossEntropyLoss().float() 
                loss = loss_fn(ypred_batch.cpu(), y.long()) 
        
        
                losses.append(loss.cpu().data.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step() 

                step+=1
                bar.set_description("epoch %d loss %.3f" % (epoch_ind+1,loss)) 
                avg_loss = np.mean(losses) 
            print("[%d/%d] training, epoch loss: %.3f" % (epoch_ind+1, args.epoch_count, avg_loss))
            
            
            if val_data and args.eval: 
                val_losses = [] 
                model.eval()
                bar = tqdm(val_dataloader) 
                avg_loss = 0 
                with torch.no_grad(): 
                    for batch_ind, batch in enumerate(bar): 
                        x, y = batch 
                        x_gpu, y_gpu = x.to(device), y.to(device) 
                        prob, ypred_batch, _ = model(x_gpu)

                        loss = loss_fn(ypred_batch.cpu(), y.long()) 

                        val_losses.append(loss.cpu().data.item()) 
                        
                        bar.set_description("validation epoch %d loss %.3f" % (epoch_ind+1, loss))
                        avg_loss = np.mean(val_losses)  
            print("[%d/%d] validation, epoch loss: %.3f" % (epoch_ind+1, args.epoch_count, avg_loss))


            if avg_loss < min_loss:
               min_loss = avg_loss 
               checkpoint_dict = {
                   "model_state_dict": model.state_dict(),
                   "optimizer_state_dict": optimizer.state_dict(),
                   "loss": min_loss,
                   "step": step,
                   "epoch": epoch_ind
               }
               torch.save(checkpoint_dict, args.model_path)
               print("checkpoint saved: %s" % args.model_path) 

    
    
    if args.train: 
        train_data.close() 
    
    if args.eval: 
        val_data.close() 
    
    
    
    
    if args.test: 
        
        print("loading test set")  
        test_data = Dataset_ligand(df=df, types='test', sigma=args.sigma)
        test_dataloader = DataLoader(test_data, batch_size=args.batch_size) 
    

            
        print("loading model") 
        checkpoint = torch.load(args.model_path, map_location=device)
        
        model = model_3DCNN(verbose=args.verbose, use_cuda=use_cuda)
        model_state_dict = checkpoint.pop("model_state_dict")
        model.load_state_dict(model_state_dict, strict=False)
        model.to(device)

        pred_list = []
        y_true = [] 
        y_pred = []
        test_losses = [] 
        model.eval()
        print("start testing") 
        bar = tqdm(test_dataloader) 
        avg_loss = 0
        with torch.no_grad(): 
            for batch_ind, batch in enumerate(bar): 
                x, y = batch 
                x_gpu = x.to(device)
                prob, ypred_batch, _ = model(x_gpu)

                for i in range(prob.shape[0]):
                    pred_list.append([str(batch_ind*prob.shape[0]+ i), str(y[i].item()), str(torch.argmax(prob[i].cpu()).item())])
                    y_true.append(y[i].item())
                    y_pred.append(torch.argmax(prob[i].cpu()).item())
           

        
        with open(os.path.join(args.output_path, 'predictions.txt'), 'w') as f: 
            for i in pred_list: 
                f.write('\t'.join(i) + '\n') 

        f1 = f1_score(y_true, y_pred) 
        precision = precision_score(y_true, y_pred) 
        recall = recall_score(y_true, y_pred) 
        
        print('f1: %.3f' % f1) 
        print('precision: %.3f' % precision)
        print('recall: %.3f' % recall) 

        test_data.close() 



if __name__ == '__main__': 
    main()
