import numpy as np
import pandas as pd
import sys 
from template_utils import *
from scipy.stats import t
import matplotlib.pyplot as plt

p_value = t.sf

sys.setrecursionlimit(6000)
# Undirected graph
# Task 1: Basic graph properties

def Q1(dataframe):
    # Name the columns since there’s no header
    adj = Dictionary_creation(dataframe)

    avg_degree = Avg_degree(adj)
    
    nb_bridges = count_bridges(adj)

    nb_local_bridges = count_local_bridges(adj)
    
    local_bridges = return_local_bridges(adj)

    degrees = degrees_attached_to_local_bridges(adj, local_bridges)

    p_val = perform_t_test(degrees, avg_degree, p_value)

    return [avg_degree, nb_bridges, nb_local_bridges, p_val]


def Q2(dataframe):
    dataframe.columns = ['source', 'target', 'weight']
    scores = df.groupby('target')['weight'].sum()
    max_node = scores.idxmax()
    max_score = scores.max()
    # in_edges_count = df.groupby('target').size()
    # merged_df = pd.DataFrame({'score': scores, 'in_edges': in_edges_count})
    # avg_in_edges_per_score = merged_df.groupby('score')['in_edges'].mean()
    # plt.figure(figsize=(8,5))
    # avg_in_edges_per_score.plot(kind='bar')
    # plt.xlabel('Score')
    # plt.ylabel('Average number of incoming edges')
    # plt.title('Average Number of Incoming Edges per Score')
    # plt.grid(axis='y')
    # plt.show()
    # x = range(-5, 6)
    # y = [abs(val) for val in x]
    # plt.figure(figsize=(8,5))
    # plt.plot(x, y, marker='o')
    # plt.xlabel('x')
    # plt.ylabel('|x|')
    # plt.title('Plot of f(x) = |x|')
    # plt.grid(True)
    # plt.show()
    return [max_node, max_score]

# Undirected graph
# Task 3: Paths lengths analysis
def bfs(adj, start):
    visited = {start: 0}  # Node -> distance from start
    queue = [start]

    while queue:
        node = queue.pop(0)  # no deque, just list pop(0)
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited[neighbor] = visited[node] + 1
                queue.append(neighbor)
    return visited

def Q3(dataframe):
    adj = Dictionary_creation(dataframe)

    visited_global = set()
    components = []

    # Find all connected components
    for node in adj:
        if node not in visited_global:
            visited = bfs(adj, node)
            components.append(list(visited.keys()))
            visited_global.update(visited.keys())

    # Find largest component
    largest_component = max(components, key=lambda x: len(x))

    # Compute diameter of the largest component
    diameter = 0
    for node in largest_component:
        distances = bfs(adj, node)
        for other_node in largest_component:
            if other_node in distances:
                diameter = max(diameter, distances[other_node])

    # Count shortest paths lengths across all components
    path_length_counts = {}

    for component in components:
        for i in range(len(component)):
            start = component[i]
            distances = bfs(adj, start)
            for j in range(i + 1, len(component)):
                end = component[j]
                if end in distances:
                    dist = distances[end]
                    if dist not in path_length_counts:
                        path_length_counts[dist] = 0
                    path_length_counts[dist] += 1

    # Prepare the output list
    if path_length_counts:
        max_len = max(path_length_counts.keys())
    else:
        max_len = 0

    result = [0] * (max_len + 1)
    result[0] = diameter  # diameter at index 0

    for length in path_length_counts:
        result[length] = path_length_counts[length]

    return result
     #return [0, 0, 0, 0, 0 ...]# at index 0, the diameter of the largest connected component, at index 1 the total number of shortest paths of length 1 accross all components,
    # at index the 2 the number of shortest paths of length 2...
    

# Directed graph
# Task 4: PageRank
def Q4(dataframe):
    # Your code here
    d = 0.85
    #N = len(pd.unique(df[[0, 1]].values.ravel()))  # Total number of unique nodes
    df.columns = ['source', 'target', 'weight']
    N = len(pd.unique(df[["source", "target"]].values.ravel()))  # Total number of unique nodes

    # Build in-neighbors dict and out-degree dict
    in_neighbors = {}
    out_degree = {}

    for index, row in df.iterrows():
        u, v = row['source'], row['target']
        if v not in in_neighbors:
            in_neighbors[v] = []
        in_neighbors[v].append(u)
        out_degree[u] = out_degree.get(u, 0) + 1

    # Initialize PageRank scores
    PR = {node: 1.0 / N for node in pd.unique(df[["source", "target"]].values.ravel())}

    # Iteratively update PageRank
    converged = False
    while not converged:
        new_PR = {}
        for p in PR:
            sum_incoming = 0
            if p in in_neighbors:
                for n in in_neighbors[p]:
                    sum_incoming += PR[n] / out_degree[n]
            new_PR[p] = (1 - d) / N + d * sum_incoming

        # Check convergence
        total_change = sum(abs(new_PR[n] - PR[n]) for n in PR)
        if total_change <= 1e-6:
            converged = True
        PR = new_PR

    # Normalize so total sum is 1
    total_PR = sum(PR.values())
    for n in PR:
        PR[n] /= total_PR

    # Find node with highest PageRank
    max_node = max(PR, key=PR.get)
    max_value = PR[max_node]
    return [max_node, max_value]
    # the id of the node with the highest pagerank score, the associated pagerank value.
    # Note that we consider that we reached convergence when the sum of the updates on all nodes after one iteration of PageRank is smaller than 10^(-6)

# Undirected graph
# Task 5: Relationship analysis 
def Q5(dataframe):
    # Rename columns for clarity
    dataframe.columns = ['source', 'target', 'weight']
    
    # Create undirected adjacency dictionary
    adj = Dictionary_creation(dataframe)
    
    # Total number of triangles
    num_triangles = 0
    num_balanced = 0
    num_unbalanced = 0

    closed_triplets = 0
    total_triplets = 0

    # Create a dictionary for quick weight lookup
    edge_weights = {}
    for idx, row in dataframe.iterrows():
        u, v, w = row['source'], row['target'], row['weight']
        edge_weights[frozenset([u, v])] = w

    # Count triangles and classify them
    for u in adj:
        neighbors = adj[u]
        if len(neighbors) < 2:
            continue
        # All pairs of neighbors of u
        for v, w in my_combinations(neighbors, 2):
            if v in adj and w in adj[v]:
                # Triangle found
                num_triangles += 1
                closed_triplets += 1  # Each triangle contributes 3 closed triplets
                
                # Check balance: product of the 3 edge weights
                weight_product = (
                    edge_weights[frozenset([u, v])] *
                    edge_weights[frozenset([v, w])] *
                    edge_weights[frozenset([w, u])]
                )
                if weight_product > 0:
                    num_balanced += 1
                else:
                    num_unbalanced += 1

    # Count total triplets
    for u in adj:
        deg = len(adj[u])
        if deg >= 2:
            total_triplets += deg * (deg - 1) // 2

    if total_triplets == 0:
        gcc = 0.0
    else:
        gcc = closed_triplets / total_triplets

    # Since each triangle is counted 3 times, divide total number of triangles by 3
    num_triangles = num_triangles // 3
    if (num_balanced != 0 ):
        num_balanced = num_balanced // 3
    if (num_unbalanced != 0 ):
        num_unbalanced = num_unbalanced // 3

    return [num_triangles, num_balanced, num_unbalanced, gcc]
 # number of triangles, number of balanced triangles, number of unbalanced triangles and the GCC.

# you can write additionnal functions that can be used in Q1-Q5 functions in the file "template_utils.py", a specific place is available to copy them at the end of the Inginious task.

df = pd.read_csv('epinion.txt', header=None,sep="    ", engine="python")
print("Q1", Q1(df))
# print("Q2", Q2(df))
# print("Q3", Q3(df))
# print("Q4", Q4(df))
# print("Q5", Q5(df))