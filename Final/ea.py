# CS 420/CS 527 Final: Genetic Algorithms Car
# Author: Gaddiel Morales
# February 2022

# Adapted code from Lab 2

import os
import numpy as np
from toolz import pipe
import simulator
import graphing
from distributed import Client, LocalCluster

from leap_ec import Individual, context, test_env_var
from leap_ec import ops, probe, util
from leap_ec.decoder import IdentityDecoder
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.real_rep.ops import mutate_gaussian
from leap_ec.problem import ScalarProblem
from leap_ec.distrib import asynchronous
from leap_ec.distrib import evaluate
from leap_ec.distrib.individual import DistributedIndividual
import argparse
import sys

def fix_csv(path):
    """
    gets rid of extra line breaks
    """
    lines = []
    with open(path,'r') as file:
        lines = file.readlines()
    str_start = ""
    with open(path,'w') as writable:
        writable.write("step,fitness,genome\n")
        for line in lines:
            if line.replace("\n","")[-1] != '"':
                str_start = line
            else:
                modified_line = str_start.replace("\n","") + line
                writable.write(modified_line)
        writable.close()

# Implementation of a custom problem
class CarProblem(ScalarProblem):
    accel = 0
    def __init__(self):
        super().__init__(maximize=True)
        
    def evaluate(self, phenome):
        #run the simulator
        simulator.World.load_map()
        #simulation occurs for 500 steps by default
        results = simulator.World.simulate(phenome[0],phenome[1],phenome[2],phenome[3],phenome[4],phenome[5],phenome[6],phenome[7],accel=CarProblem.accel)
        return results[0]
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Final Genetic Algorithms")
    parser.add_argument("--n", default=100, help="population size", type=int)
    parser.add_argument("--p_m", default=0.3, help="probability of mutation", type=float)
    parser.add_argument("--p_c", default=0.3, help="probability of crossover", type=float)
    parser.add_argument("--trn_size", default=5, help="tournament size", type=int)
    parser.add_argument("--csv_output", required=True, help="csv output file name", type=str)
    parser.add_argument("--accel",default=2, help="max simulator accelration",type=float)
    args = parser.parse_args()    

    N = args.n
    p_m = args.p_m
    p_c = args.p_c
    trn_size = args.trn_size
    CarProblem.accel = args.accel

    # cluster = LocalCluster()
    # client = Client(cluster)

    max_generation = 50
    l = 8   #vector length
    spread = 1.0    #initial spread of values
    bounds = [(-spread,spread) for i in range(l)]
    parents = Individual.create_population(N,
                                           initialize=create_real_vector(bounds),
                                           decoder=IdentityDecoder(),
                                           problem=CarProblem())

    # Evaluate initial population
    parents = Individual.evaluate_population(parents)

    generation_counter = util.inc_generation()
    out_f = open(args.csv_output, "w")
    while generation_counter.generation() < max_generation:
        offspring = pipe(parents,
                         ops.tournament_selection(k=trn_size),
                         ops.clone,
                         mutate_gaussian(std=1.0,expected_num_mutations='isotropic'),
                         ops.uniform_crossover(p_xover=p_c),
                         ops.evaluate,
                         ops.pool(size=len(parents)),  # accumulate offspring
                         probe.AttributesCSVProbe(stream=out_f, do_fitness=True, do_genome=True)
                        )
        
        parents = offspring
        generation_counter()  # increment to the next generation

    out_f.close()

    #fix csv from LEAP
    fix_csv(args.csv_output)

    #graphing
    graphing.graph_results(args.csv_output)

