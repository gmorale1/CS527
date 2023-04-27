"""
Programmer: Gaddiel Morales
Date: Feb 24, 2023
Project: Lab2
Purpose: Make use of evolutionary algorithms to find a genome of all 1's. 
    Evaluate different parameter combinations when simulating evolution.

Personal bonus: data parsing is done in parallel threads to speed up computation.
"""

import lab2
import numpy as np
import subprocess
import os
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing

#variables
pop_sizes = [25, 50, 75, 100]
mut_probs = [0, 0.01, 0.03, 0.05]
cross_probs = [0,0.1,0.3,0.5]
tourn_sizes = [2,3,4,5]
pops = 20      
solution = "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
pd_solution = "[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]"
out_ct = 0

def calc_diversity(genepool):
    """
    Calculates the diversity of a given genepool
    """
    diversity = 0
    
    #for each gene
    for cur_gene in enumerate(genepool):

        #compare distance to other genes
        for gene in enumerate(genepool):
            if gene[0] != cur_gene[0]:
                distance = np.sum(np.square(cur_gene[1] - gene[1])) ** 0.5
                diversity += distance / len(genepool)
    diversity = diversity / len(genepool)

    return diversity

def pd_calc_diversity(genepool):
    """
    Calculates the diversity of a given genepool (pandas)
    """
    diversity = 0
    
    #for each gene
    for cur_gene in enumerate(genepool):

        #compare distance to other genes
        for gene in enumerate(genepool):
            if gene[0] != cur_gene[0]:
                #gene = np.fromstring(gene_str,sep = " ")
                distance = np.sum(np.square(np.fromstring(cur_gene[1].replace("[","").replace("]",""),sep=" ") - np.fromstring(gene[1].replace("[","").replace("]",""),sep=" "))) ** 0.5
                diversity += distance / (genepool.shape[0])
    diversity = diversity / genepool.shape[0]

    return diversity


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
    

def simulate_all():
    """
    runs all simulations
    """
    global out_ct
    param_file = open("sim_params.csv",'w')
    param_file.write("iteration,n,p_m,p_c,trn_size")

    #for each population size
    for pop_size in pop_sizes:
        #for each mutation probability
        for mut_prob in mut_probs:

            #for each uniform crossover probability
            for cross_prob in cross_probs:

                #for each tournament size
                for tourn_size in tourn_sizes:

                    #for each random population
                    for i in range(pops): 
                        # if(out_ct % 100 == 0):
                        #     print("running population: ",out_ct)

                        #run simulation for 30 generations
                        exec_str = "python lab2.py --n " + str(pop_size) + " --p_m " + str(mut_prob) + " --p_c " + str(cross_prob) + " --trn_size " + str(tourn_size) + " --csv_output ./simulations/output_" + str(out_ct)
                        subprocess.call(exec_str)

                        #save params used
                        param_file.write(str(out_ct)+","+str(pop_size)+","+str(mut_prob)+","+str(cross_prob)+","+str(tourn_size)+"\n")
                        
                        #fix_csv("./simulations/output_" + str(out_ct))

                        #return for testing small set
                        # if(out_ct == 3):
                        #     return

                        out_ct += 1
    param_file.close()

def pd_parallel_parse(simulation):

    param_file = open("sim_params.csv",'r')
    param_lines = param_file.readlines()
    param_file.close()

    results = pd.DataFrame({
        "iteration num":[],
        "generation":[],
        "avg fitness":[],
        "best fitness":[],
        "best genome":[],
        "solved":[],
        "num solutions":[],
        "diversity":[],
        "n":[],
        "p_m":[],
        "p_c":[],
        "trn_size":[]
    })

    #print(simulation)
    sim_frame = pd.read_csv("./simulations/"+simulation)
    #print(sim_frame.head(5).to_string())
    params=""
    try:
        params = param_lines[int(simulation.split("_")[1])+1].split(",")
    except:
        print("End of File")
        return
    

    #sort generations
    for i in range(sim_frame["step"].max()):
        generation = sim_frame[sim_frame["step"] == i]

        solved = "no"
        if(generation[generation["genome"] == pd_solution].shape[0] > 0):
            solved = "yes"

        #fill entry
        gen_result = pd.DataFrame({
            "iteration num":[simulation.split("_")[1]],
            "generation":[i],
            "avg fitness":[generation["fitness"].mean()],
            "best fitness":[generation["fitness"].max()],
            "best genome":[generation[generation["fitness"] == generation["fitness"].max()]["genome"].max()],
            "solved":[solved],
            "num solutions":[generation[generation["genome"] == pd_solution].shape[0]],
            "diversity":[pd_calc_diversity(generation["genome"])],
            "n":[params[1]],
            "p_m":[params[2]],
            "p_c":[params[3]],
            "trn_size":[params[4].replace("\n","")]
        })
        #append entry to results
        results = pd.concat([results,gen_result])
        #print(results.to_string())
    return results

#globalized for parallelization
def pd_parse_results():
    '''
    compiles data into results.csv
    '''

    results = pd.DataFrame({
        "iteration num":[],
        "generation":[],
        "avg fitness":[],
        "best fitness":[],
        "best genome":[],
        "solved":[],
        "num solutions":[],
        "diversity":[],
        "n":[],
        "p_m":[],
        "p_c":[],
        "trn_size":[]
    })

    iter_results = []
    #parallelize
    if __name__ == '__main__':
        pool = multiprocessing.Pool()
        simulations = os.listdir('simulations')
        iter_results = pool.map(pd_parallel_parse,simulations)

    for iter_result in iter_results:
        if iter_result is not None:
            results = pd.concat([results,iter_result])
        else: 
            continue

        
    results.to_csv("results.csv",index=False)

def parse_results():
    param_file = open("sim_params.csv",'r')
    param_lines = param_file.readlines()
    param_file.close()

    with open("results.csv","w") as results:
        '''
        DEPRECATED see pd_parse_results

        results will be a csv with columns:
            iteration num
            generation
            avg fitness
            best fitness
            best genome
            solved
            num solutions
            diversity
            n
            p_m
            p_c
            trn_size
        '''
        results.write("iteration num,generation,avg fitness,best fitness,best genome,solved,num solutions,diversity,n,p_m,p_c,trn_size\n")

        #i = 0
        #for each iteration
        for simulation in os.listdir('simulations'):
            with open("./simulations/" + str(simulation), "r") as file:
                lines = file.readlines()
                
                #generation variables
                gen = 0
                avg_fit = 0
                best_fit = 0
                best_genome = ""
                solved = "no"
                sols = 0

                genepool = np.empty(shape=[0,40]) #for diversity calculation

                for line in lines:
                    
                    content = line.split(",")
                    if content[0] != "step" and len(simulation.split("_")) > 0:    #skip header/footer
                        #for each individual in generation
                        if gen == int(content[0]):

                            #parse line values
                            fitness = float(content[1])

                            gene_str = content[2].replace("\"","").replace("[","").replace("]","").replace("\n","")
                            gene = np.fromstring(gene_str,sep = " ")

                            #avg fit
                            avg_fit += fitness

                            #best fit, best genome
                            if(fitness > best_fit) and sols == 0:
                                best_fit = fitness
                                best_genome = gene

                            #solutions, solved
                            if(gene_str == solution):
                                sols += 1
                                if solved == "no":
                                    solved = "yes" 

                            #diversity
                            genepool = np.vstack((genepool, [gene]))


                        else:   #new gen
                            #write stats of last gen

                            if (avg_fit / (len(genepool))) > 1:
                                print("Impossible avg fit in gen: ",gen," file: ",simulation)

                            gen_data = str(simulation.split("_")[1])+","+str(gen)+","+str(avg_fit/(len(genepool)))+","+str(best_fit)+","+str(best_genome).replace("\n","")+","+str(solved)+","+str(sols)+","+str(calc_diversity(genepool))
                            params = param_lines[int(simulation.split("_")[1])+1].split(",")
                            sim_data = params[1]+","+params[2]+","+params[3]+","+params[4]
                            results.write(gen_data + "," + sim_data)

                            #parse line values
                            fitness = float(content[1])

                            gene_str = content[2].replace("\"","").replace("[","").replace("]","").replace("\n","")
                            gene = np.fromstring(gene_str,sep = " ")

                            #init variables for new gen using current line
                            gen = int(content[0])
                            avg_fit = fitness
                            best_fit = fitness
                            best_genome = gene
                            sols = 0
                            
                            #solutions, solved
                            if(gene_str == solution):
                                sols += 1
                                if solved == "no":
                                    solved = "yes"

                            #diversity
                            genepool = np.array([gene])


def graph_results():
    """
    graphs simulation results.

    y: fitness, x: generation
    defaults: pop size = 50, mutation: 0.01, crossover: 0.3, tournament: 2

    results columns: iteration num,generation,avg fitness,best fitness,best genome,solved,num solutions,diversity,n,p_m,p_c,trn_size
    """
    results = pd.read_csv("results.csv")
    figure, graphs = plt.subplots(2,2)

    #graph populations
    # populations = []
    # populations.append(results[(results["n"] == 25 ) & (results["p_m"] == 0.01) & (results["p_c"] == 0.3) & (results["trn_size"] == 2)])
    # populations.append(results[(results["n"] == 50 ) & (results["p_m"] == 0.01) & (results["p_c"] == 0.3) & (results["trn_size"] == 2)])
    # populations.append(results[(results["n"] == 75 ) & (results["p_m"] == 0.01) & (results["p_c"] == 0.3) & (results["trn_size"] == 2)])
    # populations.append(results[(results["n"] == 100 ) & (results["p_m"] == 0.01) & (results["p_c"] == 0.3) & (results["trn_size"] == 2)])
    # pop_div = []
    # for population in enumerate(populations):        
    #     for iteration in population[1]["iteration num"]:
    #         x = population[1][population[1]["iteration num"] == iteration]["generation"]
    #         y = population[1][population[1]["iteration num"] == iteration]["avg fitness"]
    #         yb = population[1][population[1]["iteration num"] == iteration]["best fitness"]
    #         div = population[1][population[1]["iteration num"] == iteration]["diversity"]
    #         pop_div = []

    #         #plot population
    #         graphs[population[0],0].plot(x,y, alpha = 0.002)
    #         graphs[population[0],0].set(xlabel="generation")
    #         graphs[population[0],0].set(ylabel="avg fitness")
    #         graphs[population[0],0].set(title="Population "+str(pop_sizes[population[0]])+", Avg Fitnesses")

    #         graphs[population[0],1].plot(x,yb, alpha = 0.002)
    #         graphs[population[0],1].set(xlabel="generation")
    #         graphs[population[0],1].set(ylabel="best fitness")
    #         graphs[population[0],1].set(title="Population "+str(pop_sizes[population[0]])+", Best Fitnesses")

    # plt.show()

    #  #graph mutation
    # mutations = []
    # mutations.append(results[(results["n"] == 50 ) & (results["p_m"] == 0.0) & (results["p_c"] == 0.3) & (results["trn_size"] == 2)])
    # mutations.append(results[(results["n"] == 50 ) & (results["p_m"] == 0.01) & (results["p_c"] == 0.3) & (results["trn_size"] == 2)])
    # mutations.append(results[(results["n"] == 50 ) & (results["p_m"] == 0.03) & (results["p_c"] == 0.3) & (results["trn_size"] == 2)])
    # mutations.append(results[(results["n"] == 50 ) & (results["p_m"] == 0.05) & (results["p_c"] == 0.3) & (results["trn_size"] == 2)])

    # mut_div = []

    # for mutation in enumerate(mutations):        
    #     for iteration in mutation[1]["iteration num"]:
    #         x = mutation[1][mutation[1]["iteration num"] == iteration]["generation"]
    #         y = mutation[1][mutation[1]["iteration num"] == iteration]["avg fitness"]
    #         yb = mutation[1][mutation[1]["iteration num"] == iteration]["best fitness"]
    #         div = mutation[1][mutation[1]["iteration num"] == iteration]["diversity"]
    #         mut_div.append(div)

    #         #plot population
    #         graphs[mutation[0],0].plot(x,y, alpha = 0.002)
    #         graphs[mutation[0],0].set(xlabel="generation")
    #         graphs[mutation[0],0].set(ylabel="avg fitness")
    #         graphs[mutation[0],0].set(title="Mutation "+str(mut_probs[mutation[0]])+", Avg Fitnesses")

    #         graphs[mutation[0],1].plot(x,yb, alpha = 0.002)
    #         graphs[mutation[0],1].set(xlabel="generation")
    #         graphs[mutation[0],1].set(ylabel="best fitness")
    #         graphs[mutation[0],1].set(title="Mutation "+str(mut_probs[mutation[0]])+", Best Fitnesses")

            

    # plt.show()

    # #graph crossover
    # crossovers = []
    # crossovers.append(results[(results["n"] == 50 ) & (results["p_m"] == 0.01) & (results["p_c"] == 0.0) & (results["trn_size"] == 2)])
    # crossovers.append(results[(results["n"] == 50 ) & (results["p_m"] == 0.01) & (results["p_c"] == 0.1) & (results["trn_size"] == 2)])
    # crossovers.append(results[(results["n"] == 50 ) & (results["p_m"] == 0.01) & (results["p_c"] == 0.3) & (results["trn_size"] == 2)])
    # crossovers.append(results[(results["n"] == 50 ) & (results["p_m"] == 0.01) & (results["p_c"] == 0.5) & (results["trn_size"] == 2)])
    # cros_div = []

    # for crossover in enumerate(crossovers):        
    #     for iteration in crossover[1]["iteration num"]:
    #         x = crossover[1][crossover[1]["iteration num"] == iteration]["generation"]
    #         y = crossover[1][crossover[1]["iteration num"] == iteration]["avg fitness"]
    #         yb = crossover[1][crossover[1]["iteration num"] == iteration]["best fitness"]
    #         div = crossover[1][crossover[1]["iteration num"] == iteration]["diversity"]
    #         cros_div.append(div)

    #         #plot population
    #         graphs[crossover[0],0].plot(x,y, alpha = 0.002)
    #         graphs[crossover[0],0].set(xlabel="generation")
    #         graphs[crossover[0],0].set(ylabel="avg fitness")
    #         graphs[crossover[0],0].set(title="Crossover "+str(cross_probs[crossover[0]])+", Avg Fitnesses")

    #         graphs[crossover[0],1].plot(x,yb, alpha = 0.002)
    #         graphs[crossover[0],1].set(xlabel="generation")
    #         graphs[crossover[0],1].set(ylabel="best fitness")
    #         graphs[crossover[0],1].set(title="Crossover "+str(cross_probs[crossover[0]])+", Best Fitnesses")

    # plt.show()

    # #graph tournaments
    # tournaments = []
    # tournaments.append(results[(results["n"] == 50 ) & (results["p_m"] == 0.01) & (results["p_c"] == 0.3) & (results["trn_size"] == 2)])
    # tournaments.append(results[(results["n"] == 50 ) & (results["p_m"] == 0.01) & (results["p_c"] == 0.3) & (results["trn_size"] == 3)])
    # tournaments.append(results[(results["n"] == 50 ) & (results["p_m"] == 0.01) & (results["p_c"] == 0.3) & (results["trn_size"] == 4)])
    # tournaments.append(results[(results["n"] == 50 ) & (results["p_m"] == 0.01) & (results["p_c"] == 0.3) & (results["trn_size"] == 5)])
    # tourn_div = []

    # for tournament in enumerate(tournaments):        
    #     for iteration in tournament[1]["iteration num"]:
    #         x = tournament[1][tournament[1]["iteration num"] == iteration]["generation"]
    #         y = tournament[1][tournament[1]["iteration num"] == iteration]["avg fitness"]
    #         yb = tournament[1][tournament[1]["iteration num"] == iteration]["best fitness"]
    #         div = tournament[1][tournament[1]["iteration num"] == iteration]["diversity"]
    #         tourn_div.append(div)

    #         #plot population
    #         graphs[tournament[0],0].plot(x,y, alpha = 0.002)
    #         graphs[tournament[0],0].set(xlabel="generation")
    #         graphs[tournament[0],0].set(ylabel="avg fitness")
    #         graphs[tournament[0],0].set(title="Tournament "+str(tourn_sizes[tournament[0]])+", Avg Fitnesses")

    #         graphs[tournament[0],1].plot(x,yb, alpha = 0.002)
    #         graphs[tournament[0],1].set(xlabel="generation")
    #         graphs[tournament[0],1].set(ylabel="best fitness")
    #         graphs[tournament[0],1].set(title="Tournament "+str(tourn_sizes[tournament[0]])+", Best Fitnesses")

    # plt.show()

    #graph diversity
    #high pop
    # div_pop = results[(results["n"] == 100 ) & (results["p_m"] == 0.01) & (results["p_c"] == 0.3) & (results["trn_size"] == 2)]
    # for iteration in set(div_pop["iteration num"].to_list()):
    #     sub_div = div_pop[div_pop["iteration num"] == iteration]
    #     graphs[0,0].plot(sub_div["generation"],sub_div["diversity"])
    # graphs[0,0].set(xlabel="generation")
    # graphs[0,0].set(ylabel="diversity")
    # graphs[0,0].set(title="High Population, diversity vs generations")
    # #high c
    # div_c = results[(results["n"] == 50 ) & (results["p_m"] == 0.01) & (results["p_c"] == 0.5) & (results["trn_size"] == 2)]
    # for iteration in set(div_c["iteration num"].to_list()):
    #     sub_div = div_c[div_c["iteration num"] == iteration]
    #     graphs[0,1].plot(sub_div["generation"],sub_div["diversity"])
    # graphs[0,1].set(xlabel="generation")
    # graphs[0,1].set(ylabel="diversity")
    # graphs[0,1].set(title="High Crossover, diversity vs generations")
    # #high m
    # div_m = results[(results["n"] == 50 ) & (results["p_m"] == 0.05) & (results["p_c"] == 0.3) & (results["trn_size"] == 2)]
    # for iteration in set(div_m["iteration num"].to_list()):
    #     sub_div = div_m[div_m["iteration num"] == iteration]
    #     graphs[1,0].plot(sub_div["generation"],sub_div["diversity"])
    # graphs[1,0].set(xlabel="generation")
    # graphs[1,0].set(ylabel="diversity")
    # graphs[1,0].set(title="High Mutation, diversity vs generations")
    # #high t
    # div_t = results[(results["n"] == 50 ) & (results["p_m"] == 0.01) & (results["p_c"] == 0.3) & (results["trn_size"] == 5)]
    # for iteration in set(div_t["iteration num"].to_list()):
    #     sub_div = div_t[div_t["iteration num"] == iteration]
    #     graphs[1,1].plot(sub_div["generation"],sub_div["diversity"])
    # graphs[1,1].set(xlabel="generation")
    # graphs[1,1].set(ylabel="diversity")
    # graphs[1,1].set(title="Large Tournaments, diversity vs generations")
    # plt.show()

    #low pop
    # div_pop = results[(results["n"] == 25 ) & (results["p_m"] == 0.01) & (results["p_c"] == 0.3) & (results["trn_size"] == 2)]
    # for iteration in set(div_pop["iteration num"].to_list()):
    #     sub_div = div_pop[div_pop["iteration num"] == iteration]
    #     graphs[0,0].plot(sub_div["generation"],sub_div["diversity"])
    
    # graphs[0,0].set(xlabel="generation")
    # graphs[0,0].set(ylabel="diversity")
    # graphs[0,0].set(title="Small Population, diversity vs generations")
    # #low c
    # div_c = results[(results["n"] == 50 ) & (results["p_m"] == 0.01) & (results["p_c"] == 0.0) & (results["trn_size"] == 2)]
    # for iteration in set(div_c["iteration num"].to_list()):
    #     sub_div = div_c[div_c["iteration num"] == iteration]
    #     graphs[0,1].plot(sub_div["generation"],sub_div["diversity"])
    
    # graphs[0,1].set(xlabel="generation")
    # graphs[0,1].set(ylabel="diversity")
    # graphs[0,1].set(title="Low Crossover, diversity vs generations")
    # #low m
    # div_m = results[(results["n"] == 50 ) & (results["p_m"] == 0.0) & (results["p_c"] == 0.3) & (results["trn_size"] == 2)]
    # for iteration in set(div_m["iteration num"].to_list()):
    #     sub_div = div_m[div_m["iteration num"] == iteration]
    #     graphs[1,0].plot(sub_div["generation"],sub_div["diversity"])

    # graphs[1,0].set(xlabel="generation")
    # graphs[1,0].set(ylabel="diversity")
    # graphs[1,0].set(title="Low Mutation, diversity vs generations")
    # #low t
    # div_t = results[(results["n"] == 50 ) & (results["p_m"] == 0.01) & (results["p_c"] == 0.3) & (results["trn_size"] == 2)]
    # for iteration in set(div_t["iteration num"].to_list()):
    #     sub_div = div_t[div_t["iteration num"] == iteration]
    #     graphs[1,1].plot(sub_div["generation"],sub_div["diversity"])

    # graphs[1,1].set(xlabel="generation")
    # graphs[1,1].set(ylabel="diversity")
    # graphs[1,1].set(title="Samll Tournaments, diversity vs generations")
    # plt.show()


    # #table of best performers
    # results[results["num solutions"]>0].sort_values(by=["num solutions"]).to_csv("./processed_data/best_performance.csv")

    #check mutation and crossover values of classes with high diversity
    solutions = results[results["num solutions"]>0].sort_values(by=["num solutions"])

    highcm = solutions[(solutions["p_c"]>0.3) & (solutions["p_m"]>0.1)].shape[0]
    highc = solutions[(solutions["p_c"]>0.3) & (solutions["p_m"]<=0.1)].shape[0]
    highm = solutions[(solutions["p_c"]<=0.3) & (solutions["p_m"]>0.1)].shape[0]
    all = solutions.shape[0]
    other = (solutions[(solutions["p_c"]<=0.3) & (solutions["p_m"]<=0.1)].shape[0])
    none = solutions[(solutions["p_c"]==0.0) & (solutions["p_m"]==0.0)].shape[0]
    one = solutions[(solutions["p_c"]==0.0) | (solutions["p_m"]==0.0)].shape[0]
    anym = solutions[(solutions["p_m"]>0.0)].shape[0]
    anyc = solutions[(solutions["p_c"]>0.0)].shape[0]
    with open("./processed_data/m_c_comparison.txt","w") as file:
        file.write("highcm: " + str(highcm) + "\t")
        file.write("highc: " + str(highc)+ "\t")
        file.write("highm: " + str(highm)+ "\t")
        file.write("all: " + str(all))

    sum = highcm + highc + highm + other

    fig, ax = plt.subplots()
    ax.bar(["All possible solutions","any mutation","any crossover","no crossover or no mutation"],
           [all,anym,anyc,one], color=["red","green","blue","yellow"])
    plt.show()


if __name__ == '__main__':

    SIMULATE = 0b0001
    FIX = 0b0010
    PARSE = 0b0100
    GRAPH = 0b1000

    #select which programs run:
    #   all: modules = SIMULATE | FIX | PARSE | GRAPH
    modules = GRAPH

    if(modules & SIMULATE == SIMULATE):
        simulate_all()

    if(modules & FIX == FIX):
        for simulation in os.listdir("simulations"):
            #fix_csv("./simulations/" + simulation)
            lines = []
            with open("./simulations/" + simulation,'r') as file:
                lines = file.readlines()
            with open("./simulations/" + simulation,'w') as file:
                #fix first line
                lines[0] = lines[0].removeprefix("step,fitness,genome")
                file.write("step,fitness,genome\n")
                file.writelines(lines)

    if(modules & PARSE == PARSE):
        #parse_results()
        pd_parse_results()

    if(modules & GRAPH == GRAPH):
        graph_results()
