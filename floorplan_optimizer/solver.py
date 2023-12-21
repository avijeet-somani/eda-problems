import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset
from fp_model  import FPWithLSTM
from place_env import Placement , get_place_env
import math


class Solver(nn.Module):
    def __init__(self):
        super(Solver, self).__init__()

    def reward(self, placement) :
        tour_len = placement.compute_total_distance()
        return tour_len
    




    def forward(self, inputs):
        """
        Args:
            inputs: [batch_size, input_size, seq_len]
        """
        batch = inputs
        probs, actions = self.actor(batch)
        #print('actions, probs , : ', actions.shape , probs.shape )
        #print(batch.num_graphs) 
        rewards = []
        for i in range(batch.num_graphs) :
            original_graph_data = batch[i]
            sample_solution = actions[i]
            placement = get_place_env(original_graph_data )
            placement.place_row_wise(sample_solution)
            #print( 'original_graph_data , actions ' , original_graph_data , actions[i] ) 
            sample_reward = self.reward(placement )
            rewards.append(sample_reward)
        R = torch.tensor(rewards)
        #print('Rewards: ', R)
        return R, probs, actions
       
    


class solverLSTM(Solver):
    def __init__(self, embedding_size,
            hidden_size,
            n_glimpses=1,
            tanh_exploration=10,
            ):
        super(solverLSTM, self).__init__()
        
        #start_index = None : just find the best route connecting all the nodes
        #start_index = index : start with index , end with index

        self.actor = FPWithLSTM(embedding_size,
                                hidden_size,
                                n_glimpses,
                                tanh_exploration
                                )
        