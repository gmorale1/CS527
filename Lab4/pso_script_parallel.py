# CS 420/CS 527 Lab 4: Particle Swarm Optimization 
# Gaddiel Morales
# March 2023

import numpy as np
import concurrent.futures
import subprocess
import pandas as pd
import seaborn as sns

results = pd.DataFrame({
    "num_particles":[],
    "inertia":[],
    "cognition":[],
    "social":[],
    "epoch_stop":[],
    "solution_found":[],
    "fitness":[],
    "test_num":[]
})


def run_pso(args):
    result = subprocess.run(args, capture_output=True, text=True)
    lines = result.stdout.strip().split("\n")

    # Parse the output and extract the variable values
    output_dict = {}
    for line in lines:
        name, value = line.split(":")
        output_dict[name] = [value]
    return pd.DataFrame.from_dict(output_dict)

def test_func(func):
    global results
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=16)
    tasks = []

    #test particles
    inertia = 0.5
    cognition = 1
    social = 1
    for particles in np.arange(10,110,10):
        #for 20 different tests
        for test in range(20):
            args = [
                "python", "./pso.py",
                "--num_particles", str(particles),
                "--inertia", str(inertia),
                "--cognition", str(cognition),
                "--social", str(social),
                "--func", func
            ]
            tasks.append(executor.submit(run_pso, args))

        #save at each cognition value: 800 rows
        print("function: ",func, "particles: ",particles)
        for future in concurrent.futures.as_completed(tasks):
            test_df = future.result()
            #print(test_df)
            results = pd.concat([results, test_df], ignore_index=True)
    
    tasks = []
    #save progress
    results.to_csv("./results.csv")

    #test inertia
    particles = 40
    cognition = 1
    social = 1
    for inertia in np.arange(0.1,1.1,0.1):
        #for 20 different tests
        for test in range(20):
            args = [
                "python", "./pso.py",
                "--num_particles", str(particles),
                "--inertia", str(inertia),
                "--cognition", str(cognition),
                "--social", str(social),
                "--func", func
            ]
            tasks.append(executor.submit(run_pso, args))

        #save at each value: 20 rows
        print("function: ",func, "inertia: ",inertia)
        for future in concurrent.futures.as_completed(tasks):
            test_df = future.result()
            #print(test_df)
            results = pd.concat([results, test_df], ignore_index=True)
        
        tasks = []
        #save progress
        results.to_csv("./results.csv")

    #test cognition and social combinations
    particles = 40
    inertia = 0.5
    for cognition in np.arange(0.1,4.1,0.1):    #cognition
        for social in np.arange(0.1,4.1,0.1):   #social
            #for 20 different tests
            for test in range(20):
                args = [
                    "python", "./pso.py",
                    "--num_particles", str(particles),
                    "--inertia", str(inertia),
                    "--cognition", str(cognition),
                    "--social", str(social),
                    "--func", func
                ]
                tasks.append(executor.submit(run_pso, args))

        #save at each value: 800 rows
        print("function: ",func,"cognition: ",cognition,"social: ", social)
        for future in concurrent.futures.as_completed(tasks):
            test_df = future.result()
            #print(test_df)
            results = pd.concat([results, test_df], ignore_index=True)
        
        tasks = []
        #save progress
        results.to_csv("./results.csv")

#run all PSO combinations
test_func("Rosenbrock")
test_func("Booth")

