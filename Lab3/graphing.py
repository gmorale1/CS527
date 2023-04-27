'''
Programmer: Gaddiel Morales
Lab3: Hopfield Networks

Purpose: Graphs the results from HopfieldNet.py
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

#save_path = "C:/Users/gaddi_000/Google Drive/UTK/COSC 527 Bio Inspired Computing/Labs/Lab3/results"
save_path = "I:/My Drive/UTK/COSC 527 Bio Inspired Computing/Labs/Lab3/results"


# stability
figure, graphs = plt.subplots(1,2)

i = 0
experiments = []
for filename in os.listdir(save_path):
    f = os.path.join(save_path, filename)
    experiments.append(pd.read_csv(f))
    stability = [(1-s) for s in experiments[i].stability.to_numpy()]

    graphs[0].plot(experiments[i].index, stability, label="experiment_"+str(i))
    i+=1

graphs[0].set_xlabel("Number of imprints")
graphs[0].set_ylabel("Fraction of Unstable Imprints")
graphs[0].legend()

avg = np.zeros(50)
for i in range(len(avg)):
    for j in range(5):
        avg[i] += (1-experiments[j].stability[i])
    avg[i] = avg[i]/5

std = np.zeros(50)
for i in range(len(std)):
    std[i] = np.std([experiments[j].stability[i] for j in range(5)])


graphs[1].plot(experiments[0].index, avg)
graphs[1].fill_between(experiments[0].index,  avg+std, avg-std, alpha=0.3)
graphs[1].set_xlabel("Number of imprints")
graphs[1].set_ylabel("Fraction of Unstable Imprints")

# plt.show()
figure.tight_layout()
plt.savefig('stability.png')

#stable imprints
figure, graphs = plt.subplots(1,2)

i = 0
experiments = []
for filename in os.listdir(save_path):
    f = os.path.join(save_path, filename)
    experiments.append(pd.read_csv(f))
    num_stable = [(s) for s in experiments[i].stable_imprints.to_numpy()]

    graphs[0].plot(experiments[i].index, num_stable, label="experiment_"+str(i))
    i+=1

graphs[0].set_xlabel("Number of imprints")
graphs[0].set_ylabel("Number of Stable Imprints")
graphs[0].legend()

avg = np.zeros(50)
for i in range(len(avg)):
    for j in range(5):
        avg[i] += (experiments[j].stable_imprints[i])
    avg[i] = avg[i]/5

std = np.zeros(50)
for i in range(len(std)):
    std[i] = np.std([experiments[j].stable_imprints[i] for j in range(5)])


graphs[1].plot(experiments[0].index, avg)
graphs[1].fill_between(experiments[0].index,  avg+std, avg-std, alpha=0.3)
graphs[1].set_xlabel("Number of imprints")
graphs[1].set_ylabel("Fraction of Unstable Imprints")

# plt.show()
figure.tight_layout()
plt.savefig('stable_imprints.png')

#basins of attraction histogram
figure, graphs = plt.subplots(5)

i = 0
markers = [
    '.',
    ',',
    'o',
    'v',
    '^',
    "<",
    ">",
    "1",
    "2",
    "3",
    "4",
    "8",
    "s",
    "p",
    "P",
    "*",
    "h",
    "H",
    "+",
    "x",
    "X",
    "D",
    "d",
    "|",
    "_"
]
experiments = []
for filename in os.listdir(save_path):
    f = os.path.join(save_path, filename)
    experiments.append(pd.read_csv(f))

    basins = np.zeros((50,50))
    for j in range(50):
        basins[j] = experiments[i]["imprint_" + str(j)].to_numpy()

        #calc fraction of patterns of even imprints that have the basin size
        frac_basin = np.zeros(50)

        #for each possible basin size
        for k in range(len(basins[j])):
            #increment counter if basin size matches
            if int(basins[k][j]) == k:
                frac_basin[k] += 1
        #turn basin counter into fraction of total patterns
        for k in range(len(basins[j])):
            frac_basin[k] = frac_basin[k] / 50

        if(j != 0 and j%2 == 0):   
            graphs[i].plot(experiments[i].index, frac_basin, marker=markers[(int(j/2))], label="imprint_"+str(j) if i == 0 else "")
            graphs[i].set_xlabel("Basin size")
            graphs[i].set_ylabel("Fraction of patterns")
            
        j += 1
    figure.legend()
    i+=1

figure.set_figheight(11)
figure.set_figwidth(9)
figure.tight_layout()
plt.subplots_adjust(right=0.85)

# plt.show()
plt.savefig('basins_histogram.png')