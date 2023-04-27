import lab2
import numpy as np
import subprocess
import os
import matplotlib.pyplot as plt
import pandas as pd

#variables
pop_sizes = [25, 50, 75, 100]
mut_probs = [0, 0.01, 0.03, 0.05]
cross_probs = [0,0.1,0.3,0.5]
tourn_sizes = [2,3,4,5]
pops = 20      
solution = "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
pd_solution = "[1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]"
out_ct = 0
figure, graphs = plt.subplots(3)

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
    Calculates the diversity of a given genepool
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

def pd_parse_results():
    '''
    compiles data into results.csv
    '''
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

    for simulation in os.listdir('simulations'):
        #print(simulation)
        sim_frame = pd.read_csv("./simulations/"+simulation)
        #print(sim_frame.head(5).to_string())
        
        try:
            params = param_lines[int(simulation.split("_")[1])+1].split(",")
        except:
            print("End of File")
            continue

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
                "trn_size":[params[4]]
            })
            #append entry to results
            results = pd.concat([results,gen_result])
            if(results.shape[0] % 10000 == 0):
                print("parse progress: ", results.shape[0])
    results.to_csv("results.csv")

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

    y: iteration, x: generation
    defaults: pop size = 50, mutation: 0.01, crossover: 0.3, tournament: 2

    results columns: iteration num,generation,avg fitness,best fitness,best genome,solved,num solutions,diversity,n,p_m,p_c,trn_size
    """
    results = pd.read_csv("results.csv")

    #graph populations
    population = results[results["p_m"] == 0.01 & results["p_c"] == 0.3 & results["trn_size"] == 2]
    graphs[0].plot(population["iteration num"],population["generation"])

    #graph mutation
    mutation = results[results["n"] == 50 & results["p_c"] == 0.3 & results["trn_size"] == 2]
    graphs[0].plot(mutation["iteration num"],mutation["generation"])

    #graph crossover
    crossover = results[results["n"] == 50 & results["p_m"] == 0.01 & results["trn_size"] == 2]
    graphs[0].plot(crossover["iteration num"],crossover["generation"])

    #graph tournament size
    tournament = results[results["n"] == 50 & results["p_m"] == 0.01 & results["p_c"] == 0.3]
    graphs[0].plot(tournament["iteration num"],tournament["generation"])

    #table of best performers
    results[results["sols"]>0].sort_values(by=["sols"]).to_csv("./processed_data/best_performance.csv")


SIMULATE = 0b0001
FIX = 0b0010
PARSE = 0b0100
GRAPH = 0b1000

#select which programs run:
#   all: modules = SIMULATE | FIX | PARSE | GRAPH
modules = PARSE

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
