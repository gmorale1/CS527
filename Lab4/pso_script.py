# CS 420/CS 527 Lab 4: Particle Swarm Optimization 
# Gaddiel Morales
# March 2023

import subprocess
import pandas as pd
import numpy as np

import seaborn as sns

results = pd.DataFrame({
    "particles":[],
    "inertia":[],
    "cognition":[],
    "social":[],
    "epoch_stop":[],
    "solution_found":[],
    "fitness":[],
    "test_num":[]
})

def test_func(func):
    global results
    #for all particle values
    for particles in range(10,101):
        #for all inertias
        for inertia in np.arange(0.1,1.1,0.1):
            #for all cognitions
            for cognition in np.arange(0.1,4.1,0.1):
                #for all social params
                for social in np.arange(0.1,4.1,0.1):
                    #for 20 different tests
                    for test in range(20):
                        args = [
                            "python", "./Lab4/pso.py",
                            "--num_particles", str(particles),
                            "--inertia", str(inertia),
                            "--cognition", str(cognition),
                            "--social", str(social),
                            "--func", func
                        ]
                        result = subprocess.run(args, capture_output=True, text=True)
                        lines = result.stdout.strip().split("\n")

                        # Parse the output and extract the variable values
                        output_list = []
                        for line in lines:
                            name, value = line.split(":")
                            output_list.append({name: value})
                        output_list.append({"test_num":test})
                        test_df = pd.DataFrame(output_list)

                        results = pd.concat([results,test_df], ignore_index=True)

#run all PSO combinations
test_func("Rosenbrock")
test_func("Booth")

results.to_csv("./Lab4/results.csv")

#graphing
converging_df = results[results['fitness'] <= 1e-10]
diverging_df = results[results['fitness'] > 1e-10]
# conv_rosenbrock = converging_df[converging_df['func'] == "Rosenbrock"]
# div_rosenbrock = diverging_df[diverging_df['func'] == "Rosenbrock"]
# conv_booth = converging_df[converging_df['func'] == "Booth"]
# div_booth = diverging_df[diverging_df['func'] == "Booth"]

#converging
#num_particles
plot = sns.boxplot(x="num_particles", y="epoch_stop", hue="func", data=converging_df)
plot.savefig("./Lab4/figures/converging_num_particles.png")

#inertia
plot = sns.boxplot(x="inertia", y="epoch_stop", hue="func", data=converging_df)
plot.savefig("./Lab4/figures/converging_inertia.png")

#cognition
plot = sns.boxplot(x="cognition", y="epoch_stop", hue="func", data=converging_df)
plot.savefig("./Lab4/figures/converging_cognition.png")

#social
plot = sns.boxplot(x="social", y="epoch_stop", hue="func", data=converging_df)
plot.savefig("./Lab4/figures/converging_social.png")

#diverging
#num_particles
plot = sns.boxplot(x="num_particles", y="epoch_stop", hue="func", data=diverging_df)
plot.savefig("./Lab4/figures/diverging_num_particles.png")

#inertia
plot = sns.boxplot(x="inertia", y="epoch_stop", hue="func", data=diverging_df)
plot.savefig("./Lab4/figures/diverging_inertia.png")

#cognition
plot = sns.boxplot(x="cognition", y="epoch_stop", hue="func", data=diverging_df)
plot.savefig("./Lab4/figures/diverging_cognition.png")

#social
plot = sns.boxplot(x="social", y="epoch_stop", hue="func", data=diverging_df)
plot.savefig("./Lab4/figures/diverging_social.png")

#cognition and social heatmap
#converging
heatmap_data = converging_df.pivot_table(index='cognition', columns='social', values='epoch_stop', aggfunc='mean')
plot = sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True)
plot.savefig("./Lab4/figures/converging_heatmap.png")

#diverging
heatmap_data = diverging_df.pivot_table(index='cognition', columns='social', values='epoch_stop', aggfunc='count')
plot = sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True)
plot.savefig("./Lab4/figures/diverging_heatmap.png")