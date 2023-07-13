import networkx as nx
import json
import os
import signal
import pandas as pd
import random
import re
import shutil
import subprocess
import sys

def gen_grid_graph(dim, weight_range):
    G = nx.grid_2d_graph(*dim).to_directed()
    for u, v in list(G.edges):
        if u[0] > v[0] or u[1] > v[1]:
            G.remove_edge(u, v)
    
    G.add_node('s')
    G.add_node('t')
    for x in G.nodes:
        if x != 's' and x != 't':
            if x[1] == 0:
                G.add_edge('s', x)
            if x[1] == dim[1] - 1:
                G.add_edge(x, 't')
    
    relabel_map = {}
    for x in G.nodes:
        if x != 's' and x != 't':
            relabel_map[x] = str(x[0] * dim[1] + x[1])
    G = nx.relabel_nodes(G, relabel_map)
    
    for u, v in G.edges:
        if u == 's' or v == 't':
            G[u][v]['weight'] = 0
        else:
            G[u][v]['weight'] = random.randint(*weight_range)
    return G

def gen_grid_graph(dim, weight_range):
    G = nx.watts_strogatz_graph(dim[0] * dim[1], 4, 0.2)
    G = G.to_directed()
    mn = min(G.nodes)

    shortest_path_lengths = nx.shortest_path_length(G, source=mn)
    max_length = -1
    node_with_max_length = None
    for node, length in shortest_path_lengths.items():
        if length > max_length and length != float("inf"):
            max_length = length
            node_with_max_length = node


    relabel_map = {u: str(u) for u in G.nodes}
    G = nx.relabel_nodes(G, relabel_map)
    G.add_node('s')
    G.add_node('t')
    G.add_edge('s', str(mn))
    G.add_edge(str(node_with_max_length), 't')
    
    for u, v in G.edges:
        if u == 's' or v == 't':
            G[u][v]['weight'] = 0
        else:
            G[u][v]['weight'] = random.randint(*weight_range)
    return G

def gen_grid_graph(dim, weight_range):
    G = nx.grid_2d_graph(*dim).to_directed()
    for u, v in list(G.edges):
        if u[0] > v[0] or u[1] > v[1]:
            G.remove_edge(u, v)
    
    G.add_node('s')
    G.add_node('t')
    for x in G.nodes:
        if x != 's' and x != 't':
            if x[1] == 0:
                G.add_edge('s', x)
            if x[1] == dim[1] - 1:
                G.add_edge(x, 't')
    
    relabel_map = {}
    for x in G.nodes:
        if x != 's' and x != 't':
            relabel_map[x] = str(x[0] * dim[1] + x[1])
    G = nx.relabel_nodes(G, relabel_map)
    
    for u, v in G.edges:
        if u == 's' or v == 't':
            G[u][v]['weight'] = 0
        else:
            G[u][v]['weight'] = random.randint(*weight_range)
    return G

def gen_int_weights(G, weight_range):
    int_weights = {}
    for u, v in G.edges:
        if u == 's' or v == 't':
            int_weights[(u, v)] = 0
        else:
            int_weights[(u, v)] = random.randint(*weight_range)
    return int_weights

class ExperimentLogger:
    global_info = ['Experiment Name', '|V|', '|E|', 'Interdiction Budget', 'Number of Refinement Iterations', 'Number of Nodes Per Refinement Partition', 'Timeout']
    def __init__(self, solvers, export_path):
        self.info = self.global_info + sum([[f"{x[0]} Objective Value", f"{x[0]} Time"] for x in solvers], [])
        if os.path.exists(export_path):
            self.df = pd.read_csv(export_path)
        else:
            self.df = pd.DataFrame(columns=self.info)
        self.solvers = solvers
        self.export_path = export_path
        
    def log_experiment(self, experiment_name, G, int_weights, budget, refine_iters, refine_partitions):
        curr_experiment = [experiment_name, len(G.nodes), len(G.edges), budget, refine_iters, refine_partitions, 1]
        
        if os.path.exists(f"experiments/{experiment_name}"):
            shutil.rmtree(f"experiments/{experiment_name}")
        os.mkdir(f"experiments/{experiment_name}")
        nx.write_weighted_edgelist(G, f"experiments/{experiment_name}/graph.edgelist")
        with open(f"experiments/{experiment_name}/interdiction_weight.intvals", 'w+') as f:
            f.write(json.dumps({str(x): int_weights[x] for x in int_weights.keys()}))

        prev_time = None
        
        for i, solver in enumerate(self.solvers):
            print(f"Using {solver[0]} solver")
            
            try:
                if i == 2:
                    cmd = f"time python3 {solver[1]} {experiment_name} {str(budget)} {str(refine_iters)} {str(refine_partitions)} {str(prev_time)}"
                else:
                    cmd = f"time python3 {solver[1]} {experiment_name} {str(budget)} {str(refine_iters)} {str(refine_partitions)}"
                problem = subprocess.Popen(cmd, shell=True, start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                output, error = problem.communicate()
                tim = error.decode().replace('\t','').replace('\n','').split('user')[1].split('sys')[0]
                m = float(tim.split('m')[0])
                s = float(tim.split('m')[1][:-1])
                tim = 60 * m + s
                print("out: " + output.decode())
                print("err: " + str(tim))
                problem_obj = re.search(r'Objective value: (.*?)\.', output.decode()).group(1)
                problem_timeout = re.search(r'bad', output.decode())
                problem_time = tim
                curr_experiment.append(problem_obj)
                if problem_timeout != None:
                  curr_experiment.append(f">")
                else:
                  curr_experiment.append(problem_time)
                  prev_time = problem_time
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(problem.pid), signal.SIGTERM)
                curr_experiment.append("?")
                curr_experiment.append(f">")
            
        self.df.loc[len(self.df.index)] = curr_experiment
        self.df.to_csv(self.export_path, index=False)

    def log_experiment_old(self, experiment_name, nodes, edges, budget, refine_iters, refine_partitions):
        curr_experiment = [experiment_name, nodes, edges, budget, refine_iters, refine_partitions, 1]

        prev_time = None
        
        for i, solver in enumerate(self.solvers):
            print(f"Using {solver[0]} solver")
            
            try:
                cmd = ["/usr/bin/time", "-f", "%U", "python3", f"{solver[1]}", experiment_name, str(budget), str(refine_iters), str(refine_partitions)]
                if i == 2:
                    cmd = f"/bin/time -f %U python3 {solver[1]} {experiment_name} {str(budget)} {str(refine_iters)} {str(refine_partitions)} {prev_time}"
                else:
                    cmd = f"/bin/time -f %U python3 {solver[1]} {experiment_name} {str(budget)} {str(refine_iters)} {str(refine_partitions)}"
                problem = subprocess.Popen(cmd, shell=True, start_new_session=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                output, error = problem.communicate()
                print("out: " + output.decode())
                print("err: " + error.decode())
                problem_obj = re.search(r'Objective value: (.*?)\.', output.decode()).group(1)
                problem_timeout = re.search(r'bad', output.decode())
                problem_time = float(error.decode())
                curr_experiment.append(problem_obj)
                if problem_timeout != None:
                  curr_experiment.append(f">")
                else:
                  curr_experiment.append(problem_time)
                  prev_time = problem_time
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(problem.pid), signal.SIGTERM)
                curr_experiment.append("?")
                curr_experiment.append(f">")
            
        self.df.loc[len(self.df.index)] = curr_experiment
        self.df.to_csv(self.export_path, index=False)

if __name__ == "__main__":
    experiment_logger = ExperimentLogger([("No Change", "solvers/solve_none.py"), ("Refine Problem", "solvers/solve_refine.py"), ("Solve Full Problem (CBC)", "solvers/solve_full_cbc.py")], "out/graph_size.csv")

    for i in [95, 100]:
      G = gen_grid_graph((i, i), (1, 10))
      int_weights = gen_int_weights(G, (1, 10))

      budget = round(.005 * len(G.edges))

      experiment_logger.log_experiment(f"random_{i}x{i}", G, int_weights, budget, 50, 20)

    for i in range(50, 105, 5):
      G = gen_grid_graph((i, i), (1, 10))
      int_weights = gen_int_weights(G, (1, 10))

      budget = round(.005 * len(G.edges))

      experiment_logger.log_experiment(f"random_{i}x{i}", G, int_weights, budget, 50, 40)
