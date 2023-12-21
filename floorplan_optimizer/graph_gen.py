import networkx as nx
import matplotlib.pyplot as plt
import random

import torch
#from torch.utils.data import  DataLoader, Dataset
from torch_geometric.data import Data,  Dataset, Batch
from torch_geometric.loader import DataLoader

class GraphGenerator : 
    def __init__() : 
        pass

    @staticmethod
    def skew_graph(graph) : 
        #few nodes are highly connected compared to others.
        # Select 20% of nodes to be high-degree nodes
        num_nodes = graph.number_of_nodes()
        num_high_degree_nodes = int(0.2 * num_nodes)
        high_degree_nodes = random.sample(range(num_nodes), num_high_degree_nodes)
        #print('high_degree_nodes  : ', high_degree_nodes )
        # Set the number of edges to add for each high-degree node
        num_edges_per_high_degree_node = 4

        # Set the degree for the high-degree nodes
        for node in high_degree_nodes:
            for _ in range(num_edges_per_high_degree_node):
                target_node = random.choice(list(set(range(num_nodes)) - set([node])))
                graph.add_edge(node, target_node)

        return graph
        

    @staticmethod
    def generate_random_graph(num_nodes, probability_of_edge=0.2, seed=None) :
        random_graph = nx.erdos_renyi_graph(num_nodes, probability_of_edge, seed=seed)
        random_graph = GraphGenerator.skew_graph(random_graph)
        data = GraphGenerator.graph_to_edge_index(random_graph) 
        #print('node feature : ', x)
        #print(data)
        return data, random_graph
    
    @staticmethod
    def graph_to_edge_index(graph) : 
        degrees = [val for (node, val) in graph.degree()]
        node_feature = torch.tensor(degrees, dtype=torch.float )
        x = torch.tensor(torch.unsqueeze(node_feature, dim=1), dtype=torch.float)
        edge_index = torch.tensor(list(graph.edges), dtype=torch.long).t().contiguous()
        
        data = Data(x, edge_index=edge_index)
        return data

    


    # Visualize the graph
    @staticmethod
    def visualize_graph(self, graph) : 
        nx.draw(graph, with_labels=True, font_weight='bold')
        plt.show()


class GraphDataset(Dataset):

    def __init__(self, num_nodes=None, num_samples=None, random_seed=42, graphs=None):
        super(GraphDataset, self).__init__()
        torch.manual_seed(random_seed)
        self.data_set = []
        self.graphs = []
        if graphs != None : 
            num_samples = len(graphs)
        print('GraphDataset :' , num_samples)
        for l in range(num_samples):
            if graphs == None : 
                data, graph = GraphGenerator.generate_random_graph(num_nodes, probability_of_edge=0.1)
            else : 
                graph = graphs[l]
                data = GraphGenerator.graph_to_edge_index(graph)

            self.data_set.append(data)
            self.graphs.append(graph)
        self.size = len(self.data_set)

  

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return idx, self.data_set[idx]




def test():
    train_loader = DataLoader(GraphDataset(3, 10), batch_size=8, shuffle=True)  
    #print('train_loader' , train_loader)
    # Iterate over the DataLoader

    '''
    for batch in train_loader:
    # Access the original graphs in the batch
        for i in range(batch.num_graphs):
        # Access the i-th graph in the batch
            original_graph_data = batch[i]
    '''
  
    for batch_idx, (indices,sample_batch) in enumerate(train_loader):
        print(batch_idx, indices, sample_batch , sample_batch.num_graphs )
        for i in range(sample_batch.num_graphs):
            original_graph_data = sample_batch[i]
            #print( original_graph_data )
       
        
       


   

        ''' 
        for idx in enumerate(sample_batch):
            # Access the original graph data using the index
            original_graph_data = sample_batch[idx]
            print(original_graph_data)
        '''
    # Access batches from the DataLoader
    #for batch in train_loader:
    #    print("Batch - Edge Index:", batch.edge_index)
    
    
    return train_loader




#test()



