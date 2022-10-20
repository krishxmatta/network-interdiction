import gurobipy as gp
import json
import networkx as nx
import sys

from gurobipy import GRB

def import_graph(experiment):
    G = nx.read_weighted_edgelist(f"experiments/{experiment}/graph.edgelist", create_using=nx.DiGraph())
    for u, v in G.edges:
        G[u][v]['weight'] = int(G[u][v]['weight'])
    with open(f"experiments/{experiment}/interdiction_weight.intvals") as f:
        tmp = json.load(f)
        int_weights = {eval(x): int(tmp[x]) for x in tmp.keys()}
    return G, int_weights

def solve_full(G, int_weights, budget):
    pi_ub = sum([curr[2]['weight'] for curr in G.edges(data=True)] + list(int_weights.values()))

    
    m = gp.Model('networkInterdictionFull')
    m.setParam('OutputFlag', 0)
    
    pi = {}
    for node in G.nodes:
        var_name = f"pi_{node}"
        pi[node] = m.addVar(lb=0, ub=pi_ub, vtype=GRB.INTEGER, name=var_name)
    x = {}
    for edge in G.edges:
        var_name = f"x_{edge[0]}_{edge[1]}"
        x[edge] = m.addVar(vtype=GRB.BINARY, name=var_name)
        
    start_constraint = m.addConstr(pi['s'] == 0, name="startConstraint")
    budget_constraint = m.addConstr(sum([x[edge] for edge in G.edges]) <= budget, name="budgetConstraint")
    edge_constraints = m.addConstrs((pi[j] - pi[i] - int_weights[(i, j)] * x[(i, j)] <= G[i][j]['weight'] for i, j in G.edges), name="edgeConstraints")
    
    m.setObjective(pi['t'], GRB.MAXIMIZE)
    
    m.optimize()
    
    sol = {}
    for var in pi:
        sol[f"pi_{var}"] = pi[var].X
    for var in x:
        sol[f"x_{var[0]}_{var[1]}"] = x[var].X
    return sol

if __name__ == "__main__":
    G, int_weights = import_graph(sys.argv[1])
    sol = solve_full(G, int_weights, int(sys.argv[2]))
    print(sol)
    print(f"Objective value: {sol['pi_t']}")