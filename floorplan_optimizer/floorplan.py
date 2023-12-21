
import argparse
import os
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
import utils
import hp_optimizer
import train
import hier_partition


def get_full_path(relative_path) : 
    full_path = os.path.join(os.getcwd() , relative_path)
    return full_path

def parse_arguments() : 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="LSTM")
    parser.add_argument("--seq_len", type=int, default=25)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--num_tr_dataset", type=int, default=10000)
    parser.add_argument("--num_te_dataset", type=int, default=200)
    parser.add_argument("--embedding_size", type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_clip", type=float, default=1.5)
    parser.add_argument("--use_cuda", type=bool, default=False)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--train", type=bool, default=False) 
    parser.add_argument("--model_path", type=str, default="") 
    parser.add_argument("--run_dir", type=str, default="floorplan_runs/fp_run") 
    parser.add_argument("--hp_trials", type=int, default=12) 
    parser.add_argument("--hp_parallelism", type=int, default=4) 
    parser.add_argument("--eval", type=bool, default=False)  
    parser.add_argument("--display_placement", type=bool, default=False)  
    parser.add_argument("--blank", type=bool, default=False)
    parser.add_argument("--incremental_train", type=bool, default=False) 
    parser.add_argument("--cluster_and_place", type=bool, default=False) 
    args = parser.parse_args()
    return args





def main() : 
    args = parse_arguments() 
    run_path = args.run_dir
    if args.model_path : 
        full_model_path = get_full_path(args.model_path)
        print('full_model_path : ', full_model_path )
        assert full_model_path != None
        assert os.path.exists(full_model_path) 
       
    if args.train : 
        run_path = run_path + '_train'
    if args.incremental_train : 
        run_path = run_path + 'incr_train'

    new_directory_name = utils.create_unique_directory(run_path)
    os.chdir(new_directory_name)
    if args.train : 
        hp_opt = hp_optimizer.HPOptimizer(args)    
        hp_opt.kickstart() 
    elif args.display_placement  : 
        model = torch.load(full_model_path)
        train.place_and_display(args, model)      
    elif args.eval : 
        model = torch.load(full_model_path)
        train.eval_model(args, model)
    elif args.blank : 
        train.FP_RL(args, 5, 2)
    elif args.incremental_train : 
        model = torch.load(full_model_path)
        train.incremental_train(args, model)
    elif args.cluster_and_place : 
        model = torch.load(full_model_path)
        hier_partition.HierarchicalFP(args, model)
    else : 
        assert 0
        


if __name__ == "__main__":
    main()      
