import networkx as nx
import json
import pyomo.environ as pe
from pyomo.opt.results import SolverStatus
import pyomo
import sys

def import_graph(experiment):
    G = nx.read_weighted_edgelist(f"experiments/{experiment}/graph.edgelist", create_using=nx.DiGraph())
    for u, v in G.edges:
        G[u][v]['weight'] = int(G[u][v]['weight'])
    with open(f"experiments/{experiment}/interdiction_weight.intvals") as f:
        tmp = json.load(f)
        int_weights = {eval(x): int(tmp[x]) for x in tmp.keys()}
    return G, int_weights

def solve(G, int_weights, budget):
    return nx.shortest_path_length(G, source='s', target='t', weight='weight')

def export_frequency_dictionary(freq_dict, filename):
    with open(filename, 'w') as file:
        for key, value in freq_dict.items():
            file.write(f"{key}: {value}\n")

def shortest_path_length_distribution(graph, int_list, int_weights):
    for u, v in int_list:
        graph[u][v]['weight'] += int_weights[(u, v)]

    shortest_path_lengths = nx.shortest_path_length(graph, weight='weight')
    length_distribution = {}

    for source in shortest_path_lengths:
        for target, length in shortest_path_lengths[source].items():
            if length in length_distribution:
                length_distribution[length] += 1
            else:
                length_distribution[length] = 1

    for u, v in int_list:
        graph[u][v]['weight'] -= int_weights[(u, v)]

    return length_distribution

if __name__ == "__main__":
    G, int_weights = import_graph(sys.argv[1])
    sol = solve(G, int_weights, int(sys.argv[2]))
    print(f"Objective value: {float(sol)}")
