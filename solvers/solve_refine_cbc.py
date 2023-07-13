import json
import networkx as nx
import nxmetis
import os
import pyomo.environ as pe
import pyomo
import random
import sys
import time

from pyomo.opt import SolverStatus, TerminationCondition

def import_graph(experiment):
    G = nx.read_weighted_edgelist(f"experiments/{experiment}/graph.edgelist", create_using=nx.DiGraph())
    for u, v in G.edges:
        G[u][v]['weight'] = int(G[u][v]['weight'])
    with open(f"experiments/{experiment}/interdiction_weight.intvals") as f:
        tmp = json.load(f)
        int_weights = {eval(x): int(tmp[x]) for x in tmp.keys()}
    return G, int_weights

def add_graph(G, int_weights, int_list):
    for u, v in int_list:
        G[u][v]['weight'] += int_weights[(u, v)]

def sub_graph(g, int_weights, int_list):
    for u, v in int_list:
        G[u][v]['weight'] -= int_weights[(u, v)]
        
def gen_sol(G, int_weights, int_list):
    ret = {}
    add_graph(G, int_weights, int_list)
    for edge in G.edges:
        curr = 0
        if edge in int_list:
            curr = 1
        ret[f"x_{edge[0]}_{edge[1]}"] = curr
    length, path = nx.single_source_dijkstra(G, 's', weight='weight')
    for node in G.nodes:
        ret[f"pi_{node}"] = length[node]
    sub_graph(G, int_weights, int_list)
    return ret
        
def find_int_list(sol):
    ret = []
    for i in filter(lambda x: x[0] == 'x', sol.keys()):
        if sol[i]:
            u, v = [x for x in i[1:].split('_')[1:]]
            ret.append((u, v))
    return ret

def calc_int_shortest_path(G, int_weights, int_list):
    add_graph(G, int_weights, int_list)
    ret = nx.shortest_path(G, source='s', target='t', weight='weight')
    sub_graph(G, int_weights, int_list)
    return ret

def gen_partitions(G, num_in_partition):
    G = G.copy()
    for u, v in G.edges:
        G[u][v]['weight'] = random.randint(1, 10)
    curr_options = nxmetis.MetisOptions()
    curr_options.seed = int.from_bytes(os.urandom(5), 'big')
    curr_options.contig = 1
    ret = nxmetis.partition(G.to_undirected(), len(G.nodes) // num_in_partition, recursive=False, options=curr_options)[1]
    return ret

def inverse_partition(partition_groups):
    ret = {}
    for i in enumerate(partition_groups):
        for j in i[1]:
            ret[j] = i[0]
    return ret

def solve_subproblem(G, int_weights, budget, end_node, curr_sol, curr_partition, partition_groups, partition_inverse):
    pi_ub = sum(int_weights.values()) + sum([G[u][v]['weight'] for u, v in G.edges])

    m = pe.ConcreteModel()
    m.node_set = pe.Set(initialize=partition_groups[curr_partition])
    m.edge_set = pe.Set(initialize=filter(lambda z: partition_inverse[z[1]] == curr_partition, G.edges))

    lb = {}
    ub = {}
    for i in partition_groups[curr_partition]:
        var_name = f'pi_{i}'
        lb[var_name] = 0
        ub[var_name] = pi_ub
    def pib(model, i):
        return (lb[f'pi_{i}'], ub[f'pi_{i}'])
    m.pi = pe.Var(m.node_set, domain=pe.NonNegativeIntegers, bounds=pib)
    m.x = pe.Var(m.edge_set, domain=pe.Binary)

    def obj_rule(model):
        return model.pi[end_node]
    m.OBJ = pe.Objective(rule=obj_rule, sense=pe.maximize)

    if partition_inverse["s"] == curr_partition:
        m.start_constraint = pe.Constraint(expr = m.pi["s"] == 0)
    curr_use = 0
    for i, j in G.edges:
        if partition_inverse[j] != curr_partition and curr_sol[f"x_{i}_{j}"]:
            curr_use += 1
    m.budget_constraint = pe.Constraint(expr = sum([m.x[x] for x in m.edge_set]) <= budget - curr_use)
    def edge_constraint(model, i, j):
        if partition_inverse[i] == curr_partition:
            return model.pi[j] - model.pi[i] - int_weights[(i, j)] * model.x[(i, j)] <= G[i][j]['weight']
        return model.pi[j] - curr_sol[f"pi_{i}"] - int_weights[(i, j)] * model.x[(i, j)] <= G[i][j]['weight']
    m.edge_constraints = pe.Constraint(m.edge_set, rule=edge_constraint)
    optimizer = pyomo.opt.SolverFactory('cbc', executable="/work/acslab/users/hpcguest3004/darwin-nw-int/cbc")

    # optimizer.options['seconds'] = 0.1625
    # optimizer.options['seconds'] = 0.00625
    result = optimizer.solve(m)
    result.Solver.Status = SolverStatus.ok
    result.solver.status = SolverStatus.ok
    result.solver.termination_condition = TerminationCondition.optimal
    m.solutions.load_from(result)

    ret = {}
    for i, j in G.edges:
        var_name = f'x_{i}_{j}'
        if partition_inverse[j] == curr_partition:
            ret[var_name] = m.x[(i, j)].value
        else:
            ret[var_name] = curr_sol[var_name]
    return ret

def solve_refine(G, int_weights, budget, num_iters, num_in_partition):
    curr_sol = gen_sol(G, int_weights, [])
    curr_obj = curr_sol["pi_t"]
    curr_budget = 1
    shortest_path_nodes = None
    prev_sp = set(calc_int_shortest_path(G, int_weights, find_int_list(curr_sol)))
    good_arcs = set()
    cycle = set()

    while curr_budget <= budget:
        partition_groups = gen_partitions(G, num_in_partition)
        partition_inverse = inverse_partition(partition_groups)
        
        new_shortest_path_nodes = set(calc_int_shortest_path(G, int_weights, find_int_list(curr_sol)))
        shortest_path_nodes = prev_sp.union(new_shortest_path_nodes)
        prev_sp = new_shortest_path_nodes
        
        next_sol = []
        next_obj = curr_obj
        for node in shortest_path_nodes:
            curr_use = 0
            for i, j in G.edges:
                if partition_inverse[j] != partition_inverse[node] and curr_sol[f"x_{i}_{j}"]:
                    curr_use += 1
            if curr_use == curr_budget - 1:
                let_budget = curr_budget
            else:
                let_budget = curr_budget - 1
             
            new_sol = solve_subproblem(G, int_weights, let_budget, node, curr_sol, partition_inverse[node], partition_groups, partition_inverse)
            new_sol = gen_sol(G, int_weights, find_int_list(new_sol))
            if new_sol["pi_t"] == next_obj:
                next_sol.append(new_sol)
            if new_sol["pi_t"] > next_obj:
                next_sol = [new_sol]
                next_obj = new_sol["pi_t"]
        if len(next_sol):
            curr_sol = random.choice(next_sol)
            curr_obj = next_obj
        curr_budget += 1
    
    while curr_budget <= budget + num_iters:
        partition_groups = gen_partitions(G, num_in_partition)
        partition_inverse = inverse_partition(partition_groups)
        
        new_shortest_path_nodes = set(calc_int_shortest_path(G, int_weights, find_int_list(curr_sol)))
        shortest_path_nodes = prev_sp.union(new_shortest_path_nodes)
        prev_sp = new_shortest_path_nodes
        
        next_sol = []
        next_obj = curr_obj
        for node in shortest_path_nodes:
            new_sol = solve_subproblem(G, int_weights, min(curr_budget, budget), node, curr_sol, partition_inverse[node], partition_groups, partition_inverse)
            new_sol = gen_sol(G, int_weights, find_int_list(new_sol))
            if new_sol["pi_t"] == next_obj:
                next_sol.append(new_sol)
            if new_sol["pi_t"] > next_obj:
                next_sol = [new_sol]
                next_obj = new_sol["pi_t"]
            
        if next_sol:
            curr_sol = random.choice(next_sol)
            curr_obj = next_obj
        else:
            curr_int_list = find_int_list(curr_sol)
            for edge in set(curr_int_list).difference(good_arcs):
                new_int_list = curr_int_list[:]
                new_int_list.remove(edge)
                tmp_sol = gen_sol(G, int_weights, new_int_list)
                for node in shortest_path_nodes:
                    new_sol = solve_subproblem(G, int_weights, min(budget, curr_budget), node, tmp_sol, partition_inverse[node], partition_groups, partition_inverse)
                    new_sol = gen_sol(G, int_weights, find_int_list(new_sol))
                    if new_sol["pi_t"] > next_obj or (new_sol["pi_t"] == next_obj and next_obj == curr_obj and find_int_list(new_sol) != find_int_list(curr_sol)):
                        if new_sol["pi_t"] == next_obj and edge not in good_arcs:
                            good_arcs.add(edge)
                            cycle.add(edge)
                        else:
                            good_arcs = good_arcs.difference(cycle)
                        next_sol = new_sol
                        next_obj = new_sol["pi_t"]
                        break
                if next_sol:
                    curr_sol = next_sol
                    curr_obj = next_obj
                    break
                else: 
                    good_arcs.add(edge)
        curr_budget += 1
    return curr_sol

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
    sol = solve_refine(G, int_weights, int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    int_list = get_int_list(sol)
    obj = shortest_modified_path(G, int_list, int_weights)
    print(f"Objective value: {obj}.")
