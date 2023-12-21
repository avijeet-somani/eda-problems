import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import math

'''Logic : 
model outputs a node , placement engine puts that in the next available location . Model will try to learn to select nodes such that it minimizes the wirelength.
'''

 
def generate_graph_from_data(data):
    # Extract information from Data object
    x = data.x.numpy()
    edge_index = data.edge_index.numpy().T
    #print('generate_graph_from_data : ', x, edge_index, data)
    # Create a graph using NetworkX
    graph = nx.Graph()
    graph.add_nodes_from(range(x.shape[0]))
    graph.add_edges_from(edge_index)

    return graph


def get_place_env(graph_data) : 
    graph = generate_graph_from_data(graph_data)
    num_nodes = graph.number_of_nodes()
    nrows = math.ceil(math.sqrt(num_nodes))
    ncols = nrows
    placement = Placement(nrows, ncols, graph)
    return placement


class Placement :
    def __init__(self, nrows, ncols, graph) :
        #print('Placement rows cols : ', nrows, ncols)
        self.num_rows = nrows
        self.num_cols = ncols
        self.placement = dict()
        self.graph = graph

    def assert_valid_position(self, ri, ci) : 
        assert ri >= 0 and ri < self.num_rows , ri
        assert ci >= 0 and ci < self.num_cols , ci

    def place(self, block_index, ri, ci):
        assert block_index not in self.placement
        self.assert_valid_position(ri, ci)
        self.placement[block_index] = (ri,ci)
    
    def get_row_col(self, index) : 
        row_id = math.floor(index / self.num_rows)
        col_id = index % self.num_cols
        return row_id, col_id
    
    def get_distance_between_placed_nodes(self, n1, n2) :
        p1 = self.placement[n1]
        p2 = self.placement[n2]
        self.assert_valid_position(p1[0], p1[1])  
        self.assert_valid_position(p2[0], p2[1])   
        #manhattan distance 
        weight = 1
        manhattan_dist = ( abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) ) * weight
        assert manhattan_dist > 0 and manhattan_dist <= ( self.num_rows + self.num_rows )
        return float(manhattan_dist)

    
    def place_row_wise(self, sample_solution) : 
        for index, block_id in enumerate(sample_solution) : 
            ri, ci = self.get_row_col(index)
            #print('block_id, ri, ci :' , block_id.item(), ri, ci)
            self.place(block_id.item(), ri, ci)


    def compute_total_distance(self, display=False) : 
        total_distance = 0
        #print('self.graph.edges :', len(self.graph.edges) )
        for src,dst in self.graph.edges : 
            total_distance += self.get_distance_between_placed_nodes(src, dst)
        if display : 
            self.display()
        return total_distance
       

    def display(self) : 
        # Create a grid to visualize the placement
        grid_shape = (self.num_rows , self.num_cols)
        grid = np.zeros(grid_shape, dtype=int)
        #for coord in zip(*object_coordinates):
        for node, coord in self.placement.items() :
            grid[coord] = node
            degree = self.graph.degree()[node]
            plt.text(coord[0], coord[1], 'N:' + str(node) + ' D:' + str(degree), color='red', ha='center', va='center')
        

   

        # Visualize the grid with placed objects
        plt.imshow(grid, cmap='viridis', interpolation='nearest')
        plt.title('Grid with Placed Objects')
        
        plt.show()
