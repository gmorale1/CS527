import pandas as pd 
import matplotlib.pyplot as plt
import simulator as sim
import numpy as np



def graph_results(path:str, map_path="maps/circle.map"):

    # read in the CSV file
    output = pd.read_csv(path)
    file = path.split("/")[1]

    #graph average fitness over time
    # group the data by step and calculate the mean fitness
    grouped = output.groupby("step").mean()["fitness"]

    # plot the average fitness by generation
    plt.plot(grouped.index, grouped.values, label="Average")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")

    #graph best fitness
    # group the data by step and calculate the mean fitness
    grouped = output.groupby("step").max()["fitness"]

    # plot the average fitness by generation
    plt.plot(grouped.index, grouped.values, label="Best")
    plt.legend()
    plt.savefig('simulation_results/graphs/'+file+'_fitness.png')
    plt.clf()

    # Load the terrain map from a text file
    with open(map_path, 'r') as f:
        terrain_data = f.read().strip().split('\n')
    terrain_map = np.array([list(map(int, row)) for row in terrain_data])

    #re-simulate first and last gen to compare paths
    #graph first gen
    car_paths = pd.DataFrame(columns=["car_num","x","y"])
    gen = output[output['step'] == 0]
    car_num = 0
    for genome_str in gen['genome'].values:
        genome = [float(val) for val in genome_str.strip("[]").split()]
        #car_paths = car_paths.append(pd.DataFrame{"car_num": car_num, "x": [], "y": []}, ignore_index = True)
        car_paths.loc[car_num]=[car_num,[sim.World.start_position[0]],[sim.World.start_position[1]]]
        car_paths = sim.World.simulate(genome[0],genome[1],genome[2],genome[3],genome[4],genome[5],genome[6],genome[7],record_frame=car_paths)[1]
        car_num+=1
        
    # Group the data by car number
    grouped = car_paths.groupby('car_num')

    # Plot the paths of each car
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(terrain_map, cmap='gray', origin='lower')
    for car_number, group in grouped:
        x = group['x'].values
        y = group['y'].values
        ax.plot(x[0], y[0], label=f'Car {car_number}')
    #ax.legend(loc="upper left")
    ax.invert_yaxis()
    plt.savefig('simulation_results/graphs/'+file+'_gen1_fit.png')
    plt.clf()

    #graph last gen
    car_paths = pd.DataFrame(columns=["car_num","x","y"])
    gen = output[output['step'] == output['step'].max()]
    car_num = 0
    for genome_str in gen['genome'].values:
        genome =[float(val) for val in genome_str.strip("[]").split()]
        #car_paths.append({"car_num": car_num, "x": [], "y": []}, ignore_index = True)
        car_paths.loc[car_num]=[car_num,[sim.World.start_position[0]],[sim.World.start_position[1]]]
        car_paths = sim.World.simulate(genome[0],genome[1],genome[2],genome[3],genome[4],genome[5],genome[6],genome[7],record_frame=car_paths)[1]
        car_num+=1
        
    # Group the data by car number
    grouped = car_paths.groupby('car_num')

    # Plot the paths of each car
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(terrain_map, cmap='gray', origin='lower')
    for car_number, group in grouped:
        x = group['x'].values
        y = group['y'].values
        ax.plot(x[0], y[0], label=f'Car {car_number}')
    #ax.legend(loc="upper left")
    ax.invert_yaxis()
    plt.savefig('simulation_results/graphs/'+file+'_gen_final_fit.png')
    plt.clf()

    #graph best car of all time
    car_paths = pd.DataFrame(columns=["car_num","x","y"])
    car_paths.loc[0]=[0,[sim.World.start_position[0]],[sim.World.start_position[1]]]
    best_car_index = output['fitness'].idxmax()
    best_car = output.loc[best_car_index]
    genome_str = best_car['genome']
    genome =[float(val) for val in genome_str.strip("[]").split()]
    car_paths = sim.World.simulate(genome[0],genome[1],genome[2],genome[3],genome[4],genome[5],genome[6],genome[7],record_frame=car_paths)[1]
    grouped = car_paths.groupby('car_num')

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(terrain_map, cmap='gray', origin='lower')
    for car_number, group in grouped:   #should only be one anyway, but the way the simulation returns results assumes multiple cars
        x = group['x'].values
        y = group['y'].values
        ax.plot(x[0], y[0], label="best_car")
    ax.set_title("best car")
    #ax.legend(loc="upper left")
    ax.invert_yaxis()
    plt.savefig('simulation_results/graphs/'+file+'_best_car.png')
    plt.clf()