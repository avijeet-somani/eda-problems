import torch
from torch_geometric.data import Data,  Dataset
from torch_geometric.loader import DataLoader
import graph_gen
import torch.nn as nn
import math
#from modules import Attention, GraphEmbedding
from torch.distributions import Categorical
#from solver import Solver, solver_LSTM, solver_Attention
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import optuna
from optuna.trial import TrialState
import optuna.visualization as vis
import matplotlib.pyplot as plt
#import tsp_heuristic
import networkx as nx
from torchviz import make_dot
from fp_model import FPWithLSTM
from graph_gen import GraphDataset
from torch_geometric.nn import GCNConv
from solver import solverLSTM
from place_env import Placement , get_place_env
import utils

def get_model(model_type, 
            membedding_size,
            hidden_size,
            ) : 
    if model_type == "LSTM" : 
        model = solverLSTM(membedding_size,
            hidden_size
            )
    elif model_type == "Attention" : 
        model = solverAttention(membedding_size,
            hidden_size)    
    else : 
        assert 0
    return model
        

def create_datasets(args) : 
    train_dataset = GraphDataset(args.seq_len, args.num_tr_dataset)
    test_dataset = GraphDataset(args.seq_len, args.num_te_dataset)
    return train_dataset, test_dataset



def batch_train(args, train_dataset, model, optimizer) :
    
    beta = args.beta
    grad_clip = args.grad_clip
    num_epochs = args.num_epochs

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        )
    moving_avg = torch.zeros(args.num_tr_dataset)
    
    
    #generating first baseline
    for (indices,sample_batch) in tqdm(train_data_loader):
        #print('baseline generation : ', sample_batch.num_graphs)
        rewards, _, _ = model(sample_batch)
        moving_avg[indices] = rewards
    

    # Train loop
    model.train()
    writer = SummaryWriter()
    for epoch in range(num_epochs):
        for batch_idx, (indices, sample_batch) in enumerate(train_data_loader):
            #print('in training loop : ' , batch_idx , indices, sample_batch ) 
            rewards, log_probs, action = model(sample_batch)
            #print('log_probs, action , rewards:' ,  log_probs.shape, action.shape , rewards.shape)
            log_probs = torch.sum(log_probs, dim=-1)
            log_probs[log_probs < -100] = -100
            
            moving_avg[indices] = moving_avg[indices] * beta + rewards * (1.0 - beta)
            #penalize if rewards > moving_avg . if rewards < moving_avg should i make the loss negative or zero ??
            #advantage = rewards - moving_avg[indices]
            #loss = (advantage * log_probs).mean()

            penalty = rewards - moving_avg[indices]
            loss = (penalty * log_probs).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        #print("loss , epoch ", loss , epoch)
        writer.add_scalar("Loss vs epoch", loss, epoch)
        writer.flush()
    
    writer.close()
    return model



def batch_test(test_dataset, model, display=False) : 
    model.eval()
    batch_size = len(test_dataset)
    distance = torch.zeros(batch_size)
    tour_list = []

    with torch.no_grad() :
        eval_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  
        for indices, sample_batch in eval_loader: 
            #print('batch_test in loop' , sample_batch.num_graphs)
            R, _, action  = model(sample_batch) 
            if display : 
                for i in range(sample_batch.num_graphs) :
                    original_graph_data = sample_batch[i]
                    sample_solution = action[i]
                    placement = get_place_env(original_graph_data )
                    placement.place_row_wise(sample_solution)
                    placement.display()
            R_avg = R.mean().detach().numpy()
    print('Average Tour length : ', R_avg)       
    return R_avg


def eval_model(args, model) : 
    model_random = get_model(
            args.model_type,
            args.embedding_size,
            args.hidden_size,
           )
   

    train_dataset, test_dataset = create_datasets(args)
    R_avg_random = batch_test(test_dataset, model_random)
    R_avg_model = batch_test(test_dataset, model)
    print('eval model before/after : ',  R_avg_random, R_avg_model)


def place_and_display(args, model) :
    dataset = graph_gen.GraphDataset(args.seq_len, 1)
    batch_test(dataset, model, display=True)


def FP_RL(args, num_train_data, num_test_data):
    #train_dataset, test_dataset = create_datasets(args)
    train_dataset = GraphDataset(args.seq_len, num_train_data  )
    test_dataset = GraphDataset(args.seq_len, num_test_data )
    
   
    model = get_model(
            args.model_type,
            args.embedding_size,
            args.hidden_size,
           )
    optimizer = torch.optim.Adam(model.parameters(), lr=3.0 * 1e-4)    
    reward = batch_test(test_dataset, model)
    print("AVG Tour Distance before Training", reward)

    model = batch_train(args, train_dataset, model, optimizer)
    utils.save_model(model, 'test_model.h5')

    reward = batch_test(test_dataset, model) #reward is the tour distance     
    print("AVG Tour Distance after Training", reward)   
    return model


def incremental_train(args, model) : 
    optimizer = torch.optim.Adam(model.parameters(), lr=3.0 * 1e-4)  
    train_dataset, test_dataset = create_datasets(args)
    reward = batch_test(test_dataset, model)
    print("AVG Tour Distance before Training", reward)

    model = batch_train(args, train_dataset, model, optimizer)
    utils.save_model(model, 'incremental_train_model.h5')

    reward = batch_test(test_dataset, model) #reward is the tour distance     
    print("AVG Tour Distance after Training", reward)   
    return model

