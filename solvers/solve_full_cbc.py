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

def solve(G, int_weights, budget, timeout):
    m = pe.ConcreteModel()
    m.node_set = pe.Set(initialize=list(G.nodes))
    m.edge_set = pe.Set(initialize=list(G.edges))

    m.pi = pe.Var(m.node_set, domain=pe.NonNegativeIntegers)
    m.x = pe.Var(m.edge_set, domain=pe.Binary)

    def obj_rule(model):
        return model.pi['t']
    m.OBJ = pe.Objective(rule=obj_rule, sense=pe.maximize)

    m.start_constraint = pe.Constraint(expr = m.pi["s"] == 0)
    m.budget_constraint = pe.Constraint(expr = sum([m.x[x] for x in m.edge_set]) <= budget)
    def edge_constraint(model, i, j):
        return model.pi[j] - model.pi[i] - int_weights[(i, j)] * model.x[(i, j)] <= G[i][j]['weight']
    m.edge_constraints = pe.Constraint(m.edge_set, rule=edge_constraint)
    optimizer = pyomo.opt.SolverFactory('cbc', executable="/work/acslab/users/hpcguest3004/darwin-nw-int/cbc")
    optimizer.options['seconds'] = timeout
    result = optimizer.solve(m)
    result.Solver.Status = SolverStatus.warning
    m.solutions.load_from(result)

    if result.solver.termination_condition != pyomo.opt.TerminationCondition.optimal:
      print("bad")

    ret = {}
    for i in G.nodes:
        var_name = f'pi_{i}'
        ret[var_name] = m.pi[i].value
    for i, j in G.edges:
        var_name = f'x_{i}_{j}'
        ret[var_name] = m.x[(i, j)].value
    return ret

def export_frequency_dictionary(freq_dict, filename):
    with open(filename, 'w') as file:
        for key, value in freq_dict.items():
            file.write(f"{key}: {value}\n")

def get_int_list(dictionary):
    tuples = []
    for key in dictionary:
        if key.startswith("x_") and dictionary[key] == 1:
            _, u, v = key.split('_')
            tuples.append((u, v))
    return tuples

def shortest_modified_path(G, int_list, int_weights):
    for u, v in int_list:
        G[u][v]['weight'] += int_weights[(u, v)]

    shortest_length = nx.shortest_path_length(G, source='s', target='t', weight='weight')

    for u, v in int_list:
        G[u][v]['weight'] -= int_weights[(u, v)]

    return shortest_length

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
    sol = solve(G, int_weights, int(sys.argv[2]), float(sys.argv[3]))
    int_list = get_int_list(sol)
    obj = shortest_modified_path(G, int_list, int_weights)
    print(f"Objective value: {sol['pi_t']}")
