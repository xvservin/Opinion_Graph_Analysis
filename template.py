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
    adj = Dictionary_creation(dataframe)

    avg_degree = Avg_degree(adj)
    
    nb_bridges = count_bridges(adj)

    nb_local_bridges = count_local_bridges(adj)
    
    local_bridges = return_local_bridges(adj)

    degrees = degrees_attached_to_local_bridges(adj, local_bridges)

    p_val = perform_t_test(degrees, avg_degree, p_value)

    # # Graph: Show the histogram of the degree distribution
    # degree_distribution = [len(neighbors) for neighbors in adj.values()]
    # plt.figure(figsize=(8, 6))
    # plt.hist(degree_distribution, bins=range(min(degree_distribution), 30),
    #          edgecolor='black', alpha=0.7)
    # plt.xlabel('Degree')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Degree Distribution')
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.show()

    return [avg_degree, nb_bridges, nb_local_bridges, p_val]

def Q2(dataframe):
    dataframe.columns = ['source', 'target', 'weight']
    scores = dataframe.groupby('target')['weight'].sum()
    max_node = scores.idxmax()
    max_score = scores.max()
    in_edges_count = dataframe.groupby('target').size()
    merged_df = pd.DataFrame({'score': scores, 'in_edges': in_edges_count})
    avg_in_edges_per_score = merged_df.groupby('score')['in_edges'].mean().sort_index()
    # plt.figure(figsize=(12, 6))
    # bars = plt.bar(avg_in_edges_per_score.index, avg_in_edges_per_score.values, width=0.8, alpha=0.7, label='Avg Incoming Edges')
    # x = np.linspace(min(scores.min(), -25), max(scores.max(), 50), 50)
    # y = np.abs(x)
    # line = plt.plot(x, y, 'r-', linewidth=2, label='f(x) = |x|')
    # plt.xlabel('Score')
    # plt.ylabel('Average Number of Incoming Edges / |x| Value')
    # plt.title('average number of incoming edges for each score and Absolute Value Function')
    # plt.legend()
    # plt.grid(True, linestyle='--', alpha=0.6)
    # plt.xlim(min(scores.min(), -25) - 2, max(scores.max(), 50) + 2)
    # plt.tight_layout()
    # plt.show()
    return [max_node, max_score]

def Q3(dataframe):
    adj = Dictionary_creation(dataframe)
    visited_global = set()
    components = []
    for node in adj:
        if node not in visited_global:
            visited = bfs(adj, node)
            components.append(list(visited.keys()))
            visited_global.update(visited.keys())
    largest_component = max(components, key=lambda x: len(x))
    diameter = 0
    for node in largest_component:
        distances = bfs(adj, node)
        for other_node in largest_component:
            if other_node in distances:
                diameter = max(diameter, distances[other_node])
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
    if path_length_counts:
        max_len = max(path_length_counts.keys())
    else:
        max_len = 0
    result = [0] * (max_len + 1)
    result[0] = diameter
    for length in path_length_counts:
        result[length] = path_length_counts[length]
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(8, 6))
    # x = sorted(path_length_counts.keys())
    # y = [path_length_counts[length] for length in x]
    # plt.bar(x, y, color='skyblue', edgecolor='black', alpha=0.8)
    # plt.xlabel('Shortest Path Length')
    # plt.ylabel('Number of Node Pairs')
    # plt.title('Distribution of Shortest Path Lengths')
    # plt.grid(True, linestyle='--', alpha=0.7)
    # plt.show()
    return result

def Q4(dataframe):
    d = 0.85
    df.columns = ['source', 'target', 'weight']
    N = len(pd.unique(df[["source", "target"]].values.ravel()))
    in_neighbors = {}
    out_degree = {}
    for index, row in df.iterrows():
        u, v = row['source'], row['target']
        if v not in in_neighbors:
            in_neighbors[v] = []
        in_neighbors[v].append(u)
        out_degree[u] = out_degree.get(u, 0) + 1
    PR = {node: 1.0 / N for node in pd.unique(df[["source", "target"]].values.ravel())}
    converged = False
    while not converged:
        new_PR = {}
        for p in PR:
            sum_incoming = 0
            if p in in_neighbors:
                for n in in_neighbors[p]:
                    sum_incoming += PR[n] / out_degree[n]
            new_PR[p] = (1 - d) / N + d * sum_incoming
        total_change = sum(abs(new_PR[n] - PR[n]) for n in PR)
        if total_change <= 1e-6:
            converged = True
        PR = new_PR
    total_PR = sum(PR.values())
    for n in PR:
        PR[n] /= total_PR
    max_node = max(PR, key=PR.get)
    max_value = PR[max_node]
    return [max_node, max_value]

def Q5(df):
    df.columns = ['source', 'target', 'weight']
    
    # Enlever les self-loops dès le départ
    df = df[df['source'] != df['target']]
    
    adj = Dictionary_creation(df)

    # Créer un dictionnaire pour accéder rapidement aux poids
    weights = {}
    for _, row in df.iterrows():
        u, v, w = row['source'], row['target'], row['weight']
        if (u, v) not in weights and (v, u) not in weights:
            weights[(u, v)] = w
            weights[(v, u)] = w


    triangles = set()
    balanced = 0
    unbalanced = 0
    
    # Parcourir en garantissant un ordre croissant
    for u in sorted(adj):
        for v in sorted(adj[u]):
            if v <= u:
                continue
            for w in sorted(adj[v]):
                if w <= v or w == u or w not in adj[u]:
                    continue
                triangle = (u, v, w)  # u < v < w garanti
                if triangle not in triangles:
                    triangles.add(triangle)
                    
                s1 = weights.get((u, v), weights.get((v, u)))
                s2 = weights.get((v, w), weights.get((w, v)))
                s3 = weights.get((w, u), weights.get((u, w)))
                if s1 is not None and s2 is not None and s3 is not None:
                    prod = s1 * s2 * s3
                    if prod == 1:
                        balanced += 1
                    else:
                        unbalanced += 1

    num_triangles = len(triangles)
    closed_triplets = num_triangles * 3

    # Calcul des triplets ouverts
    open_triplets = 0
    for u in adj:
        neighbors = adj[u]
        k = len(neighbors)
        for i in range(k):
            for j in range(i + 1, k):
                v = neighbors[i]
                w = neighbors[j]
                # Vérifie que ce n’est pas un triangle fermé
                if v != w and w not in adj[v]:
                    open_triplets += 1

    gcc = closed_triplets / open_triplets if open_triplets > 0 else 0

    return [num_triangles, balanced, unbalanced, gcc]


df = pd.read_csv('epinion.txt', header=None,sep="    ", engine="python")
print("Q1", Q1(df))
# print("Q2", Q2(df))
# print("Q3", Q3(df))
# print("Q4", Q4(df))
#print("Q5", Q5(df))