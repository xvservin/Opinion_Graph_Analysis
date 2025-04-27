# If needed, write here your additional fuctions/classes with their signature and use them in the exercices:
# a specific place is available to copy them at the end of the Inginious task.

# First, import the libraries needed for your helper functions
import numpy as np
import pandas as pd

def degrees_attached_to_local_bridges(adj, local_bridges):
    degrees = []
    for u, v in local_bridges:
        degrees.append(len(adj[u]))  # Degree of node u
        degrees.append(len(adj[v]))  # Degree of node v
    return degrees

def perform_t_test(degrees, avg_degree, p_value):
    sample_mean = np.mean(degrees)
    sample_std = np.std(degrees, ddof=1)  # Sample standard deviation
    n = len(degrees)


    if (sample_std / np.sqrt(n) == 0):
        t_stat = 0
    else :
        t_stat = (sample_mean - avg_degree) / (sample_std / np.sqrt(n))
        print(f"avg_degree: {avg_degree}")
        print(f"sample_std: {sample_std}")
    p_val = 2 * p_value(np.absolute(t_stat), df=n-1)

    
    return p_val
    
def Dictionary_creation(dataframe):
    dataframe.columns = ['source', 'target', 'weight']
    adj = {}

    for index, row in dataframe.iterrows():
        u, v = row['source'], row['target']
        if u not in adj:
            adj[u] = []
        if v not in adj:
            adj[v] = []
        if v not in adj[u]:
            adj[u].append(v)
        if u not in adj[v]:
            adj[v].append(u)
            
    return adj

def count_local_bridges(adj):
    count = 0
    checked_edges = set()
    
    for key, value in adj.items():
        for v in value:
            if (key, v) not in checked_edges and (v, key) not in checked_edges:
                if is_local_bridge(key, v, adj):
                    count += 1
                checked_edges.add((key, v))
                checked_edges.add((v, key))
    
    return count

def return_local_bridges(adj):
    checked_edges = set()
    local_bridges = []
    
    for key, value in adj.items():
        for v in value:
            if (key, v) not in checked_edges and (v, key) not in checked_edges:
                if is_local_bridge(key, v, adj):
                    local_bridges.append((key, v))
                checked_edges.add((key, v))
                checked_edges.add((v, key))
    
    return local_bridges

def is_local_bridge(u, v, adj):
    shortest_path_length_before_removal = shortest_path_length(u, v, adj)
    if v == u :
        adj[u].remove(v)
    else :
        adj[u].remove(v)
        adj[v].remove(u)
    is_bridge = shortest_path_length(u, v, adj) > shortest_path_length_before_removal + 2
    
    if v == u :
        adj[u].append(v)
    else :
        adj[u].append(v)
        adj[v].append(u)
    
    return is_bridge


def shortest_path_length(u, v, adj):
    if u == v:
        return 0  

    queue = [(u, 0)]  
    visited = set()   
    visited.add(u)

    while queue:
        current_node, distance = queue.pop(0)  

        for neighbor in adj[current_node]:
            if neighbor == v:
                return distance + 1
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))

    return float('inf')  # Retourner infini si aucun chemin n'existe


def Avg_degree(adj):
            
    number_of_nodes = len(adj)
    number_of_neighbour = 0
    for key, value in adj.items() :
        number_of_neighbour += len(value)

    avd_degree = number_of_neighbour / (number_of_nodes)

    return avd_degree
          
def count_bridges(adj):
    time = [-1]
    disc = {}
    low = {}
    visited = set()
    bridges = []

    def dfs(u, parent):
        visited.add(u)
        time[0] += 1
        disc[u] = low[u] = time[0]

        for v in adj[u]:
            if v == parent:
                continue
            if v not in visited:
                dfs(v, u)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    bridges.append((u, v))
            else:
                low[u] = min(low[u], disc[v])

    for u in adj:
        if u not in visited:
            dfs(u, -1)

    return len(bridges)
  
def my_combinations(iterable, r):
    """
    Génère toutes les combinaisons de r éléments parmi les éléments de iterable.
    """
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield tuple(pool[i] for i in indices)