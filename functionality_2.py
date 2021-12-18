## Import Utils
import matplotlib.pyplot as plt
import networkx as nx
import math
from tqdm import tqdm
from datetime import timedelta
from random import sample



def my_subgraph(G, time_start, time_end):
    '''
    Input:
        - G: the initial graph
        - time_start: the starting time of the interval (date object)
        - time_end: the ending time of the interval (date object)
    Output:
        - The subgraph for the given interval of time
    '''
    # Init a new graph
    subG = nx.DiGraph()
    # For each edge of the original graph
    for edge in list(G.edges):
        # Add it to the subgraph if it is in the given time interval
        if time_start <= G.edges[edge[0], edge[1]]['timestamp'][0] <= time_end:
            subG.add_edge(edge[0], edge[1])
            subG[edge[0]][edge[1]]['weight'] = G.edges[edge[0], edge[1]]['weight']
            subG[edge[0]][edge[1]]['timestamp'] = G.edges[edge[0], edge[1]]['timestamp']

    return subG

def shortest_path(G, source, target):
    unvisited = list(nx.nodes(G)) ## all nodes
    dist = {node : float('infinity') for node in list(nx.nodes(G))} ## distances from each node to source
    previous = {node : None for node in list(nx.nodes(G))} ## previous node  
    dist[source] = 0 ## The distance of the source node is 0
    
    while unvisited :  
        dist_min = math.inf
        for node in unvisited:
            if dist[node] <= dist_min :
                dist_min = dist[node]
                u = node
        unvisited.remove(u) ## removing the node from the list of the unvisited nodes
        
        ## if the visited node is the target stop the visits
        if u == target:
            break
            
        ## Examine its neighbours
        for neighbor in G.neighbors(u):
            new_dist = dist[u] + G[u][neighbor]['weight']
            if dist[neighbor] > new_dist:
                ## update the shortest distance
                dist[neighbor] = new_dist
                ## update the previous node
                previous[neighbor] = u
                
    if previous[target] == None:
        return ('Not Connected',[])
    
    path = [target]
    while source not in path:
        path.append(previous[path[-1]])
    path.reverse()

    return dist[target], path # return the distance

'''
------- METRICS -------
'''

def Beetweenness(G, node):
    '''
    Input:
        - G: a graph
        - node: a node of the graph G
    Output:
        - The value of the beetweenness centrality
    '''
    # Initialize the sum for all the shortest paths found
    s = 0   
    # Initialize the sum for all the shortest paths that pass through the node
    s_node = 0 
    
    # Get a sample of 100 nodes for computational reasons
    # We will consider all the possible shortest paths between this set of nodes
    # and not between the entire graph.
    sample_nodes = sample(list(G.nodes), 100)
    
    # If the requested node is not in the sample, we put it in.
    if node not in sample_nodes:
        sample_nodes.append(node)

    # Main double-loop
    # For each pair of nodes of the sample
    for n1 in tqdm(sample_nodes):
        for n2 in sample_nodes:
            # Compute the shortest path
            dist = shortest_path(G, n1, n2)
            
            # If a shortest path between the nodes exists, add 1 to s
            if(len(dist[1]) != float('inf') or len(dist[1]) != 0):
                s += 1
                # If the requested node is in that path, add 1 to s_node
                if node in dist[1]:
                    s_node += 1
            else:
                return("There aren't shorthest paths here!")

    return s_node/s

def PageRank(G, alpha, node, weight = 'weight'):
    '''
    Input:
        - G: the graph
        - alpha: the damping parameter
        - node: the requested node
        - weight: the edges' weight
    Output:
        - The PageRank score for the given node
    '''

    # First of all we want to have a stochastic graph were the sum of the weights
    # for each node's edge is equal to one.
    # In this way we have the probabilities to take each edge as possible link
    # from a node to an other one.
    
    # Create a copy of the graph
    W = G.copy()
    
    # Now we compute the outer degree of each node, so the number of edges that
    # starts from it.
    
    d = dict() # Init a dictionary
    # Consider each node of the graph as a key with value 0.
    for i in W:
        d[i] = 0
        # For each edge (with the features) of the graph
        for u, v, w in W.edges(data=True):
            # When the starting node is the key one add the weight of the edge as value
            if u == i:
                d[i] += w[weight]

    # Stochastic Graph
    # For each edge (with the features) of the graph
    for u, v, w in W.edges(data=True):
        # If the outer degree of the starting node is 0
        if d[u] == 0:
            # Assign 0 to the edge's weight
            w[weight] = 0
        else:
            # Assign as new weight, the ratio between the original one and the
            # total weights of the edges that start from that node
            w[weight] = w.get(weight, 1) / d[u]
    
    N = len(W) # Number of nodes

    # Initialize a dictionary of ones / N
    q = dict.fromkeys(W, 1.0/N)

    # Store the starting dictionary
    start_q = q

    # Nodes with zero edges that point out to them
    nodes = set()
    aa = set() # just a set
    # For each node in the graph
    for n in W:
        nodes.add(n)
        # For each edge in the graph
        for _, ign in W.edges(n):
            aa.add(ign)
    # Consider the nodes that have no edges that point out to them
    ignored_nodes = nodes.difference(aa)
    
    # Now, we implement the formula of the Page Rank algorithm
    
    # Power Iteration: make up to 100 iterations
    for _ in range(100):
        qlast = q # Probability to be at a certain state after t-1 steps
        qt = dict.fromkeys(qlast.keys(), 0) 
        # Score for the ignored nodes
        # Only the damping parameter is considered because teleportation is the
        # only way to reach them
        ignored_sum = alpha * sum(qlast[n] for n in ignored_nodes)
        
        for n in qt: # For each node
            # Extract the destination node of the edges and the weight of the edge
            for _, dest_node, wt in W.edges(n, data = weight):
                # Score of the destination node = alpha*q(t-1)
                qt[dest_node] += alpha * qlast[n] * wt 
            # Score of each node = q[0]*alpha + (1 - alpha)*q[0]
            # Total probability to get at the nodes with damping or from another node
            qt[n] += ignored_sum * start_q.get(n, 0) + (1.0 - alpha) * start_q.get(n, 0)
        
        # Check convergence using l1 norm
        err = sum([abs(qt[n] - qlast[n]) for n in qt])
        # Fix tolerance value to check convergence at 0.0001
        if err < N * 0.0001: 
            return qt[node] # Return the PageRank score for the requested node
        
def Closeness(G, u):
    '''
    Input:
        - G: a graph
        - u: a node of the graph
    '''
    # Initialize the sum for the shortest path distances
    s = 0
    
    # Initialize the number of shortest paths of the node
    paths = 0
    for node in tqdm(G.nodes):
        # Compute the shortest path for each node
        dist = shortest_path(G, u, node)
        # If a path exists add its distance to the sum and 1 to 'paths'
        if (dist[0] != 'Not Connected'):
            paths += 1.0
            s += dist[0]
   
    # Compute the closeness centrality score
    return ((paths - 1.0) / s)

def Degree(G, node):
    '''
    Input:
        - G: the graph
        - node: the given node
    Output:
        - The degree centrality of the given node
    '''
    # Init a counter
    count = 0
    # For each edge in the graph
    for edge in list(G.edges):
        # When one node of the edge is equal to the given node, add 1 to the counter
        if edge[0] == str(node) or edge[1] == str(node):
            count += 1
        # If a node is connected with itself, subtract 1 to the counter
        if edge[0] == edge[1] == str(node):
            count -= 1

    return count / (len(G) - 1)

'''
------- VISUALIZATION -------
'''

def my_neighs(G, node):
    '''
    Input:
        - G: the graph
        - node: the requested node
    Output:
        - A plot with the requested node and all its neighbours
    '''
    # List of the neighborhood of the node
    neighborhood = [node] + list(G.neighbors(node))

    # Consider a subgraph given by that neighborhood
    N = G.subgraph(neighborhood)
    
    # Plot the subgraph
    plt.figure(1)
    plt.figure(figsize=(15,10))
    plt.title(f'Node {node} and its neighbours')
    
    # Map of the colors:
        # Red for the node
        # Blue for the neighbours
    color_map = ['red' if n == node else 'blue' for n in N]
    # Consider all the weights in the subgraph to get different shapes of color
    e_colors = [w[2]['weight'] for w in N.edges(data=True)]
    
    # Give to the nodes a nice and adjusted position in the plot
    pos = nx.drawing.nx_agraph.graphviz_layout(N)
    
    # Draw
    nx.draw(N, pos = pos, with_labels = True, 
            node_color = color_map, edge_color = e_colors, node_size = 1800, 
            width = 4.0, edge_cmap = plt.cm.Blues)
    
def metric_evolution(G, time_start, time_end, node, metric):
    '''
    Input:
        - G: the graph
        - time_start: the starting date of the interval of time
        - time_end: the ending date of the interval of time
        - node: the requested node
        - metric: the user can choose between the 4 metrics of the functionality
    Output:
        - Plot of the evolution of the metric in the chosen days
    '''
    
    # Init values and dates for the axis of the plot
    values = []
    dates = []
    # Compute the lenght of the interval of time in days
    interval = (time_end - time_start).days
    
    # For each day in the interval
    for i in range(interval+1):
        # Every sub-interval is one day long
        time_change = timedelta(days=i)
        # Compute the new dates and append to the list
        new_date = time_start + time_change
        dates.append(new_date)
        
        # Compute a subgraph for the new sub-interval
        H = my_subgraph(G, new_date, new_date)

        # Check if the given node is in that specific sub-interval
        if node in H.nodes:
            # Compute the value for the requested metric
            # Append 0 if we get None as result
            if metric == 1:
                val = Beetweenness(H, node)
                if val == None:
                    values.append(0)
                else:
                    values.append(val)

            elif metric == 2:
                val = PageRank(H, 0.25, node)
                if val == None:
                    values.append(0)
                else:
                    values.append(val)

            elif metric == 3:
                val = Closeness(H, node)
                if val == None:
                    values.append(0)
                else:
                    values.append(val)

            elif metric == 4:
                val = Degree(H, node)
                if val == None:
                    values.append(0)
                else:
                    values.append(val)
            
            else:
                print('Please enter a valid metric!')
                
        # If the requested node is not in the subgraph append 0 to the values list
        else:
            values.append(0)

    # List of the metrics
    metrics = ['Beetweenness Centrality', 'PageRank', 'Closeness Centrality', 
               'Degree Centrality']
    
    # Plot
    plt.figure(2)
    plt.figure(figsize=(10,7))
    # Assign the name of the requested metric to the plot title
    plt.title(f'{metrics[metric-1]} score during time')
    plt.xticks(rotation=45, ha='right')
    plt.bar(dates, values)
    plt.show()


    
