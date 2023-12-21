import networkx as nx
import metis
import matplotlib.pyplot as plt
import graph_gen
import train
from torch_geometric.loader import DataLoader


class HierarchicalFP :
    def __init__(self, args, model=None) : 
        self.dataset = graph_gen.GraphDataset(args.seq_len, 1)
        self.model = model
        self.args = args
        self.place()

    

    def place(self) : 
        random_model = train.get_model(
            self.args.model_type,
            self.args.embedding_size,
            self.args.hidden_size,
           )
        
        wl_random_place = train.batch_test(self.dataset, random_model)
        wl_model_place = train.batch_test(self.dataset, self.model)


        wl_partition_with_random_place = self. partition_and_place(random_model)
        wl_partition_with_model_place = self. partition_and_place(self.model)
        
        print('FloorPlan wirelengths : fully random floorplan, fully model floorplan, paritioned with random floorplan, partitioned with model floorplan  :  ', wl_random_place ,
               wl_model_place, wl_partition_with_random_place, wl_partition_with_model_place  )
    

    def partition_and_place(self, model) :
        #print('parition the graph and place by model ')
        #num_partitions = 4  # Number of partitions
        num_partitions = (int)(self.args.seq_len / 25 )
        # Convert the graph to an adjacency list
        G = self.dataset.graphs[0]
        adjacency_list = [list(G.neighbors(node)) for node in G.nodes()]
        (edgecuts, partitions) = metis.part_graph(adjacency_list, nparts=num_partitions, recursive=False)

        # Create subgraphs for each partition
        subgraphs = [G.subgraph([node for node, part_id in enumerate(partitions) if part_id == i]) for i in range(num_partitions)]
        
        #print('after partition : ', len(subgraphs))
        total_wl = 0
        for subgraph in subgraphs : 
            #print('subgraph : ', subgraph)
            H = nx.relabel.convert_node_labels_to_integers(subgraph, first_label=0)
            subgraph_dataset = graph_gen.GraphDataset(graphs = [H]) 
           
            #place partitioned subgraphs
            total_wl += train.batch_test(subgraph_dataset, model)
        return total_wl

        
        


    
