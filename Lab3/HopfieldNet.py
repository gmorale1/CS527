'''
Programmer: Gaddiel Morales
Lab3: Hopfield Networks

Purpose: runs five experiments which create a hopfield network and then imprints 50 random patterns.
'''
import numpy as np
import pandas as pd
import multiprocessing

plen = 100

def imprint(weights,pattern,p):
    '''
    Adds a specified pattern to the network, 
    '''
    for i in range(len(weights)):
        for j in range(len(weights)):
            if i == j:
                continue

            #use pattern to adjust weight
            weights[i][j] += (pattern[i] * pattern[j]) / (p+1)

    return weights

def calc_stability(weights, vectors, p, basin):
    '''
    Calculates the stability of a network for the first p patterns of the provided vectors

    also calculates basins of attraction at the current imprint stage

    variables:
        weights: weight matrix of a hopfield network
        vectors: array of vectors representing the patterns used to generate the hopfield network
        p: number of vectors used to generate the current weight matrix
    '''
    stability = 0
    #for each pattern p imprinted so far
    for i in range(p+1):
        isStable = True #assume stable
        state = vectors[i-1]
        
        #check each neuron
        for j in range(plen):
            if not isStable: break
            #calculate new state h_i for neuron
            new_state = 0
            #sum of state times weights
            for k in range(plen):
                new_state += (weights[k][j] * state[k])
            new_state = new_state/plen

            #sigma operation
            if new_state >= 0:
                new_state = 1
            else:
                new_state = -1

            #compare new state to old state for neuron
            if state[j] != new_state:
                isStable = False    #pattern is unstable
                break

        if isStable:
            stability += 1

            #add score to basin for pattern p
            basin[i] = calc_basin(weights, vectors, i)


    #stability score is number of stable patterns divided by number of patterns
    return (stability / (p + 1)), stability, basin

def calc_basin(weights, vectors, p):
    '''
    calculates the basin of attraction for a hopfield network for a pattern p
    '''
    
    #make a permutation of bits to flip
    perm = np.arange(plen)
    np.random.shuffle(perm)
    for i in range(50):
        pattern = vectors[p]
        state = np.copy(pattern)
    
        #flip a number of bits equal to the size of basin we are analyzing
        for j in range(i):
            if state[perm[j]] == 1:
                state[perm[j]] = -1
            else:
                state[perm[j]] = 1
            
        #do ten updates on the network
        for j in range(10):
            #for each neuron
            for k in range(plen):
                #calculate new state
                new_state = 0
                for l in range(plen):
                    new_state += (weights[l][k] * state[l])
                new_state = new_state/plen
                
                if new_state >= 0:
                    new_state = 1
                else:
                    new_state = -1
                
                #update states
                state[k] = new_state

        #check if pattern persisted
        converge = True
        for n in range(len(pattern)):
            if state[n] != pattern[n]:
                converge = False 
                break
        if not converge:
            return i 

    return 50


def experiment():
    weights = np.zeros((plen,plen),)

    #create bipolar vactors
    vectors = np.array([[ 1 if np.random.rand() <0.5 else -1 for y in range(plen)] for x in range(50)])

    stb_scores = np.empty(50)
    num_stable = np.empty(50)
    basins = np.zeros((50,50,1))
    for p in range(50):
        #imprint
        pattern = vectors[p]
        weights = imprint(weights,pattern,p)

        #test stability
        stb_scores[p], num_stable[p], basins[p] = calc_stability(weights, vectors, p, basins[p])
    
    return [stb_scores, num_stable, basins]
        
def save(results):
    save_path = "I:/My Drive/UTK/COSC 527 Bio Inspired Computing/Labs/Lab3/results"
    for i in range(len(results)):

        result_frame = pd.DataFrame({
            "stability":results[i][0],
            "stable_imprints": results[i][1],
            })
        
        #flatten basins
        for j in range(len(results[i][2])):
            result_frame["imprint_"+str(j)]=results[i][2][j]
            

        result_frame.to_csv(save_path + "/experiment_" + str(i))


if __name__ == '__main__':
    np.random.seed(25565)
    pool = multiprocessing.Pool()
    results = pool.starmap(experiment, [() for _ in range(5)])
    save(results)





