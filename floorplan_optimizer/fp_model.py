import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import sys
from modules import Attention, GCN

import torch.nn.functional as F


class FPWithLSTM(nn.Module) : 
    def __init__(self, embedding_size,
            hidden_size,
            n_glimpses=1,
            tanh_exploration=10,
            ):
        super(FPWithLSTM, self).__init__()  
        self.n_glimpses = 1
        self.graph_embedding = GCN(1, hidden_size, embedding_size)
        self.encoder = nn.Linear(embedding_size, hidden_size)
        self.decoder = nn.LSTM(embedding_size, hidden_size, batch_first=True)
        embedding_size = embedding_size
        self.hidden_size = hidden_size
        
        self.decoder_start_input = nn.Parameter(torch.FloatTensor(embedding_size))
        self.decoder_start_input.data.uniform_(-(1. / math.sqrt(embedding_size)), 1. / math.sqrt(embedding_size))
        #print('decoder input dummy data: ', self.decoder_start_input.data)
        #self.encoder = nn.LSTM(embedding_size, self.hidden_size, batch_first=True)
       
        
        #self.glimpse = Attention(self.hidden_size)
        self.pointer = Attention(self.hidden_size)


    def reshape_graph_embedding(self, embedded, batch) : 
         # Reshape node embeddings using Batch information
        num_nodes_per_graph = [ptr_end - ptr_start for ptr_start, ptr_end in zip(batch.ptr, batch.ptr[1:])]

        # Stack embeddings for each graph
        reshaped_embeddings_list = [embedded[ptr_start:ptr_end] for ptr_start, ptr_end in zip(batch.ptr, batch.ptr[1:])]
        #print(num_nodes_per_graph)
        # Stack into a tensor of shape (batch_size, max_num_nodes, embedding_size)
        reshaped_embedded = torch.nn.utils.rnn.pad_sequence(reshaped_embeddings_list, batch_first=True)
        return reshaped_embedded



       
    def forward(self, batch ) : 
        #each batch will have similar node-sizes
        #print('FPWithLSTM forward : batch: ', batch)
        x = batch.x
        edge_index = batch.edge_index

        #x, edge_index, batch = inputs 
        batch_size = batch.num_graphs
        seq_len = int(x.shape[0] / batch_size)

        #print('forward : batch_size, seq_len' , batch_size, seq_len )
        embedded = self.graph_embedding(x, edge_index)
        # Reshape node_embeddings using batch information
        embedded = self.reshape_graph_embedding(embedded, batch)
        encoder_outputs = self.encoder(embedded)
        #print('encoder_output : ', encoder_outputs.shape)
        
        
        
        #encoder_outputs, (hidden, context) = self.encoder(embedded) #encoder_outputs hold the hidden state for all time-steps . Output shape : encoder_outputs :-(batch, seq-length, hidden-size), hidden :- (batch, hidden-size)
        
        prev_chosen_logprobs = []
        preb_chosen_indices = []
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        num_nodes_to_explore = seq_len
        decoder_input = self.decoder_start_input.unsqueeze(0).repeat(batch_size, 1)
        
        # Initialize hidden and context to zeros for the first iteration
        num_layers = 1
        hidden = torch.zeros(num_layers , batch_size,  self.hidden_size)
        context = torch.zeros(num_layers, batch_size,  self.hidden_size)
      
        
        #print('Shapes : encoder_outputs.shape, hidden.shape, context.shape , mask.shape , decoder_input.shape ' , encoder_outputs.shape, hidden.shape, context.shape , mask.shape , decoder_input.shape )
        for index in range(num_nodes_to_explore):
            
            _, (hidden, context) = self.decoder(decoder_input.unsqueeze(1), (hidden, context))
            query = hidden.squeeze(0)
           

            #query attends to the encoder_outputs
            _, logits = self.pointer(query, encoder_outputs)


            _mask = mask.clone()
            #logits are scores/unnormalized log probablities
            logits[_mask] = -100000.0
            probs = torch.softmax(logits, dim=-1) #create probs along the last tensor dimension
            cat = Categorical(probs) 
            chosen = cat.sample()
            #print('categorical , chosen ', cat, chosen.shape)
            mask[[i for i in range(batch_size)], chosen] = True
            log_probs = cat.log_prob(chosen)
            decoder_input = embedded[torch.arange(batch_size), chosen, :] #chose the embeddings for the chosen index
            #decoder_input = encoder_outputs[torch.arange(batch_size), chosen, :] #chose the embeddings for the chosen index
            
            #print('decoder_input : ', decoder_input.shape)
            prev_chosen_logprobs.append(log_probs)
            preb_chosen_indices.append(chosen)
        
        return torch.stack(prev_chosen_logprobs, 1), torch.stack(preb_chosen_indices, 1)







