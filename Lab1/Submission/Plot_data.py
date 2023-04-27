'''
Programmer: Gaddiel Morales
Purpose: plots data for CA parameters
Date: 02/11/2023
'''

import numpy as np
import matplotlib.pyplot as plt

csvfile = open("./MasterExperiment.csv","r")
lines = csvfile.readlines()

figure, graphs = plt.subplots(5)

for i in range(len(lines)):
    classnum = []
    lambdas = []
    lambda_ts = []
    Hs = []
    H_ts = []
    zeta = []
    exp_num = ""

    exper_str = "Experiment #:"
    if exper_str in lines[i]:
        exp_num = lines[i].split(",")[2] 
        
        #read experiment data
        for j in range(13):
            values = lines[i + 3 + j].split(',')

            lambdas.append(float(values[3]))
            lambda_ts.append(float(values[4]))
            Hs.append(float(values[5]))
            H_ts.append(float(values[6]))
            zeta.append(float(values[7]))

            if("1" in values[2]) or ("2" in values[2]):
                classnum.append(0)
            elif( "4" in values[2]):
                classnum.append(1)
            elif( "3" in values[2]):
                classnum.append(2)
    
        #plot experiments
        #lambda
        print(exp_num)
        print(lambdas)
        print(classnum)
        graphs[0].plot(lambdas,classnum,label=exp_num, alpha=0.150)
        graphs[0].set(xlabel = "λ")

        #lambda_t
        graphs[1].plot(lambda_ts,classnum,label=exp_num, alpha=0.150)
        graphs[1].set(xlabel = "λ_t")

        #H
        graphs[2].plot(Hs,classnum,label=exp_num, alpha=0.150)
        graphs[2].set(xlabel = "H")

        #H_t
        graphs[3].plot(H_ts,classnum,label=exp_num, alpha=0.150)
        graphs[3].set(xlabel = "H_t")

        #zeta
        graphs[4].plot(zeta,classnum,label=exp_num, alpha=0.150)
        graphs[4].set(xlabel = "zeta")

        for graph in graphs:
            graph.set(ylabel = "Class")
            plt.setp(graph.get_xticklabels(), rotation=30, horizontalalignment='right')


        i += 15
    #end reading experiment
    

#end reading experiments
figure.legend(labels=np.arange(0,30))
plt.show()

csvfile.close()
exit()