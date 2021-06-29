import numpy as np
import pandas as pd
import time

from TIV.TIVlib import TIV as tiv
from essentia.standard import MonoLoader, Windowing, Spectrum, SpectralPeaks, FrameGenerator, HPCP

from scipy.spatial import distance
from scipy.fftpack import dct

import pickle
import json
import random

from pydub import AudioSegment
from pydub.playback import play

def loadTIV(tiv_dataset, id):
#Load data (deserialize object and retrieve TIV object of filename)
    return tiv_dataset[id-1]


def loadTIVData():
    with open('json_features/tivdata.pickle', 'rb') as handle:
        unserialized_data = pickle.load(handle)
    return unserialized_data

def loadSC(sc_dataset, id):
#Load Spectral centroid data
    return sc_dataset[id-1]['sc']/np.linalg.norm(sc_dataset[id-1]['sc'])

def loadMFCC(mfcc_dataset, id):
#Load MFCC data
    return mfcc_dataset[id-1]['mfcc']/np.linalg.norm(mfcc_dataset[id-1]['mfcc'])

def loadMFCCData():
    with open('json_features/mfccdata.pickle', 'rb') as handle:
        data = pickle.load(handle)
    return data

def loadSCData():
    with open('json_features/scdata.pickle', 'rb') as handle:
        data = pickle.load(handle)
    return data

def loadLoopStructure():
    with open('json_features/harmonysimilarity.pickle', 'rb') as handle:
        datarharmony = pickle.load(handle)
    with open('json_features/rhythmsimilarity.pickle', 'rb') as handle:
        data_rhythm = pickle.load(handle)
    return datarharmony, data_rhythm

def loadHistogram(rh_dataset, id):
#Load data about histogram
    #print(dct(rh_dataset[id-1]['rh'], norm="ortho"))
    return rh_dataset[id-1]['rh']/np.linalg.norm(rh_dataset[id-1]['rh'])

def loadHistogramData():
    with open('json_features/rhdata.json', 'r') as handle:
        data = json.load(handle)
    return data

def loadLabel(json_dataset, id):
#Load data about label
    return json_dataset[id-1]['layer']

def loadLoopCount():
#Load data about filename and if
    with open('json_features/dataset.json', 'r') as handle:
        data = json.load(handle)
        rhythm_nr = []
        harmony_nr = []
        print(len(data))
        for fil in data:
            if(fil['layer'] == "rhythm"):
                print("rhythm")
                rhythm_nr.append(fil['id'])
            else:
                print("harmony")
                harmony_nr.append(fil['id'])
        return data, min(harmony_nr), max(harmony_nr), min(rhythm_nr), max(rhythm_nr)

def loadLoop(id1,id2,id3):
    with open('json_features/dataset.json', 'r') as handle:
        data = json.load(handle)
        filename_data = data[id1-1]['filename']
        filename_data2 = data[id2-1]['filename']
        filename_data3 = data[id3-1]['filename']
        return filename_data, filename_data2, filename_data3

def spectral_balance(value):
    if value >= 1.05:
        return 0
    else:
        return 0.2

def cloning(cells, Nc, beta, fit_array, min_harmony,max_harmony,min_rhythm,max_rhythm, json_dataset, tiv_dataset, rh_dataset, sc_dataset, mfcc_dataset, global_cell, horizontal_stage, harmony_weight,rhythm_weight,sc_weight):
    for i in np.arange(len(cells)):
        new_cells = np.full((Nc, 3), (cells[i]))
        
        new_fit = []

        line = 0
        for l in new_cells:
            element = 0
            for e in l:
                #print(e)
                #print("BETA")
                #print((np.exp(-np.linalg.norm(fit_array[i])*beta)))
                
                random_decimal = random.random()
                #print(random_decimal)
                if (random_decimal > (np.exp(-np.linalg.norm(fit_array[i])*beta))):

                    if(loadLabel(json_dataset, e) == "rhythm"):
                        new_cells[line][element] = np.random.randint(min_rhythm,max_rhythm,dtype=int)
                        element = element + 1
                    else:
                        #print("E MAIS ALTO QUE A FUNCAO, VAI MUTAR")
                        #print(new_cells[line][element])
                        new_cells[line][element] = np.random.randint(min_harmony,max_harmony,dtype=int)
                        element = element + 1
                else:
                    element = element + 1
            line = line + 1

        new_cells_parent = np.insert(new_cells, 0, (cells[i]), axis=0)

        line = 0
        if(horizontal_stage != True):
            for j in new_cells_parent:
                if(j[0] == j[1]):
                    new_cells_parent = np.delete(new_cells_parent, line, axis = 0)
                    continue
                tiv_obj1=loadTIV(tiv_dataset, int(j[0]))
                tiv_obj2=loadTIV(tiv_dataset, int(j[1]))
                rh_distance = distance.cosine(loadHistogram(rh_dataset, int(j[0])), loadHistogram(rh_dataset, int(j[1])))
                rhythm_rh_distance = distance.cosine(np.add(loadHistogram(rh_dataset, int(j[0])),loadHistogram(rh_dataset, int(j[1]))), loadHistogram(rh_dataset, int(j[2])))
                
                sc_distance = spectral_balance(distance.euclidean(loadSC(sc_dataset, int(j[0])), loadSC(sc_dataset, int(j[1]))))
                sc_distance_r = spectral_balance(distance.euclidean(np.add(loadSC(sc_dataset, int(j[0])), loadSC(sc_dataset, int(j[1]))),loadSC(sc_dataset, int(j[2]))))

                new_fit.append((sc_distance + sc_distance_r)*sc_weight + tiv.small_scale_compatibility(tiv_obj1['TIV'],tiv_obj2['TIV'])*harmony_weight + (rh_distance + rhythm_rh_distance)*rhythm_weight)
                line = line + 1 
        else:
            for j in new_cells_parent:
                if(j[0] == j[1]):
                    new_cells_parent = np.delete(new_cells_parent, line, axis = 0)
                    continue
                tiv_obj1=loadTIV(tiv_dataset, int(j[0]))
                tiv_obj2=loadTIV(tiv_dataset, int(j[1]))
                rh_distance = distance.cosine(loadHistogram(rh_dataset, int(j[0])), loadHistogram(rh_dataset, int(j[1])))
                rhythm_rh_distance = distance.cosine(np.add(loadHistogram(rh_dataset, int(j[0])),loadHistogram(rh_dataset, int(j[1]))), loadHistogram(rh_dataset, int(j[2])))
                
                tiv_obj3=loadTIV(tiv_dataset, int(global_cell[0]))
                tiv_obj4=loadTIV(tiv_dataset, int(global_cell[1]))
                rh_distance_global = distance.cosine(loadHistogram(rh_dataset, int(j[2])), loadHistogram(rh_dataset, int(global_cell[2])))
                
                mfcc_current = np.add(loadMFCC(mfcc_dataset, int(j[0])),loadMFCC(mfcc_dataset, int(j[1])), loadMFCC(mfcc_dataset, int(j[2])))
                mfcc_global = np.add(loadMFCC(mfcc_dataset, int(global_cell[0])),loadMFCC(mfcc_dataset, int(global_cell[1])), loadMFCC(mfcc_dataset, int(global_cell[2])))

                mfcc_distance = distance.cosine(mfcc_current,mfcc_global)

                new_fit.append(tiv.small_scale_compatibility(tiv_obj1['TIV'],tiv_obj2['TIV']) + rh_distance + rhythm_rh_distance +
                    tiv.small_scale_compatibility(tiv_obj1['TIV'],tiv_obj3['TIV'])+ tiv.small_scale_compatibility(tiv_obj2['TIV'],tiv_obj4['TIV']) + rh_distance_global +
                    mfcc_distance)
                line = line + 1

        newest_fit_index = np.where(new_fit == np.amin(new_fit))

        if fit_array[i] > np.amin(new_fit):
            cells[i] = new_cells_parent[newest_fit_index[0][0]]

            fit_array[i] = np.amin(new_fit)
    return cells


#Ab_random = cloning2(Ab_random, Nc, beta, fit_random,harmonic_loops,rhythm_loops, min_harmony,min_rhythm, json_dataset, tiv_dataset, rh_dataset, sc_dataset, mfcc_dataset, global_cell, horizontal_stage)
def cloning2(cells, Nc, beta, fit_array, harmonic_loops,rhythm_loops,min_harmony,min_rhythm, json_dataset, tiv_dataset, rh_dataset, sc_dataset, mfcc_dataset, global_cell, horizontal_stage):
    for i in np.arange(len(cells)):
        new_cells = np.full((Nc, 3), (cells[i]))
        new_fit = []

        line = 0
        #print(new_cells)
        for l in new_cells:
            #print(l)
            element = 0
            for e in l:
                #print(e)
                #print("BETA")
                #print((np.exp(-np.linalg.norm(fit_array[i])*beta)))
                
                random_decimal = random.random()
                #print(random_decimal)
                if (random_decimal > (np.exp(-np.linalg.norm(fit_array[i])*beta))):

                    if(loadLabel(json_dataset, e) == "rhythm"):
                        
                        #print(new_cells[line][element])
                        new_cells[line][element] = min_rhythm+np.random.choice(rhythm_loops[element+min_rhythm]['loops'])
                        #print("RHYTHM CELL")
                        #print(rhythm_loops[element+min_rhythm]['loops'])
                        element = element + 1
                    else:
                        #print("E MAIS ALTO QUE A FUNCAO, VAI MUTAR")
                        #print(new_cells[line][element])
                        new_cells[line][element] = min_harmony+np.random.choice(harmonic_loops[element+min_harmony]['loops'])
                        #print("HARMONY CELL")
                        #print(harmonic_loops)
                        #print(harmonic_loops[element+min_harmony]['loops'])
                        #print(new_cells[line][element])
                        element = element + 1
                else:
                    element = element + 1
            line = line + 1

        new_cells_parent = np.insert(new_cells, 0, (cells[i]), axis=0)

        line = 0
        if(horizontal_stage != True):
            for j in new_cells_parent:
                if(j[0] == j[1]):
                    new_cells_parent = np.delete(new_cells_parent, line, axis = 0)
                    continue
                tiv_obj1=loadTIV(tiv_dataset, int(j[0]))
                tiv_obj2=loadTIV(tiv_dataset, int(j[1]))
                rh_distance = distance.cosine(loadHistogram(rh_dataset, int(j[0])), loadHistogram(rh_dataset, int(j[1])))
                rhythm_rh_distance = distance.cosine(np.add(loadHistogram(rh_dataset, int(j[0])),loadHistogram(rh_dataset, int(j[1]))), loadHistogram(rh_dataset, int(j[2])))
                
                sc_distance = spectral_balance(distance.euclidean(loadSC(sc_dataset, int(j[0])), loadSC(sc_dataset, int(j[1]))))

                new_fit.append(sc_distance + tiv.small_scale_compatibility(tiv_obj1['TIV'],tiv_obj2['TIV']) + rh_distance + rhythm_rh_distance)
                line = line + 1 
        else:
            for j in new_cells_parent:
                if(j[0] == j[1]):
                    new_cells_parent = np.delete(new_cells_parent, line, axis = 0)
                    continue
                tiv_obj1=loadTIV(tiv_dataset, int(j[0]))
                tiv_obj2=loadTIV(tiv_dataset, int(j[1]))
                rh_distance = distance.cosine(loadHistogram(rh_dataset, int(j[0])), loadHistogram(rh_dataset, int(j[1])))
                rhythm_rh_distance = distance.cosine(np.add(loadHistogram(rh_dataset, int(j[0])),loadHistogram(rh_dataset, int(j[1]))), loadHistogram(rh_dataset, int(j[2])))
                
                tiv_obj3=loadTIV(tiv_dataset, int(global_cell[0]))
                tiv_obj4=loadTIV(tiv_dataset, int(global_cell[1]))
                rh_distance_global = distance.cosine(loadHistogram(rh_dataset, int(j[2])), loadHistogram(rh_dataset, int(global_cell[2])))
                
                mfcc_current = np.add(loadMFCC(mfcc_dataset, int(j[0])),loadMFCC(mfcc_dataset, int(j[1])), loadMFCC(mfcc_dataset, int(j[2])))
                mfcc_global = np.add(loadMFCC(mfcc_dataset, int(global_cell[0])),loadMFCC(mfcc_dataset, int(global_cell[1])), loadMFCC(mfcc_dataset, int(global_cell[2])))

                mfcc_distance = distance.cosine(mfcc_current,mfcc_global)

                new_fit.append(tiv.small_scale_compatibility(tiv_obj1['TIV'],tiv_obj2['TIV']) + rh_distance + rhythm_rh_distance +
                    tiv.small_scale_compatibility(tiv_obj1['TIV'],tiv_obj3['TIV'])+ tiv.small_scale_compatibility(tiv_obj2['TIV'],tiv_obj4['TIV']) + rh_distance_global +
                    mfcc_distance)
                line = line + 1

        newest_fit_index = np.where(new_fit == np.amin(new_fit))

        if fit_array[i] > np.amin(new_fit):
            cells[i] = new_cells_parent[newest_fit_index[0][0]]

            fit_array[i] = np.amin(new_fit)
    return cells


def opt_music(chosen_cells,global_cell,horizontal_stage):
    begin = time.time()

#USER PARAMETERS
    ts = 0.5 #0.5 para euclidean sem ritmo            #for cosine, 0,001            #threashold supression
    N = 50#10              #number of cells
    Nc = 5 #5              #number of clones
    iterations = 500 #200!!    #number of iterations

    beta =1.8
    memory_cells = 0

    harmony_weight = 1
    rhythm_weight = 1
    sc_weight = 1

    json_dataset, min_harmony,max_harmony,min_rhythm,max_rhythm = loadLoopCount()    #max capacity of loop dataset
    tiv_dataset = loadTIVData()
    rh_dataset = loadHistogramData()
    sc_dataset = loadSCData()
    mfcc_dataset = loadMFCCData()

    harmonic_loops, rhythm_loops = loadLoopStructure()

    print(min_harmony,max_harmony,min_rhythm,max_rhythm)

    Ab = np.zeros((N,2), dtype=int)

#Random population
    Ab_random = np.random.randint(min_harmony,max_harmony, size=(N,2),dtype=int)
    random_column = np.random.randint(min_rhythm,max_rhythm, size=(N,1),dtype=int)
    Ab_random = np.append(Ab_random,random_column, axis=1)
    #print(Ab_random[0]) #print(Ab_random[0][1]) #print(int(Ab_random[0][1])) #print(type(int(Ab_random[0][1])))

    print("INITIAL AB_RANDOM")
    print(Ab_random)
#Fitness of each cell
    fit_random = []
    line = 0
    if(horizontal_stage != True):
        for i in Ab_random:
            if(i[0] == i[1]):
                Ab_random = np.delete(Ab_random, line, axis = 0)
                continue
            tiv_obj1=loadTIV(tiv_dataset, int(i[0]))
            tiv_obj2=loadTIV(tiv_dataset, int(i[1]))
            rh_distance = distance.cosine(loadHistogram(rh_dataset, int(i[0])), loadHistogram(rh_dataset, int(i[1])))
            rhythm_rh_distance = distance.cosine(np.add(loadHistogram(rh_dataset, int(i[0])),loadHistogram(rh_dataset, int(i[1]))), loadHistogram(rh_dataset, int(i[2])))

            sc_distance = spectral_balance(distance.euclidean(loadSC(sc_dataset, int(i[0])), loadSC(sc_dataset, int(i[1]))))
            sc_distance_r = spectral_balance(distance.euclidean(np.add(loadSC(sc_dataset, int(i[0])), loadSC(sc_dataset, int(i[1]))),loadSC(sc_dataset, int(i[2]))))

            fit_random.append((sc_distance_r+sc_distance)*sc_weight + harmony_weight*tiv.small_scale_compatibility(tiv_obj1['TIV'],tiv_obj2['TIV']) + rhythm_weight*(rh_distance + rhythm_rh_distance))
            line = line + 1
    else: #TWO OR MORE ITERATIONS, must check TIV,RH(harmonic and rhythm), MFCC
        for i in Ab_random:
            if(i[0] == i[1]):
                Ab_random = np.delete(Ab_random, line, axis = 0)
                continue
            tiv_obj1=loadTIV(tiv_dataset, int(i[0]))
            tiv_obj2=loadTIV(tiv_dataset, int(i[1]))
            rh_distance = distance.cosine(loadHistogram(rh_dataset, int(i[0])), loadHistogram(rh_dataset, int(i[1])))
            rhythm_rh_distance = distance.cosine(np.add(loadHistogram(rh_dataset, int(i[0])),loadHistogram(rh_dataset, int(i[1]))), loadHistogram(rh_dataset, int(i[2])))

            sc_distance = spectral_balance(distance.euclidean(loadSC(sc_dataset, int(i[0])), loadSC(sc_dataset, int(i[1]))))

            tiv_obj3=loadTIV(tiv_dataset, int(global_cell[0]))
            tiv_obj4=loadTIV(tiv_dataset, int(global_cell[1]))
            rh_distance_global = distance.cosine(loadHistogram(rh_dataset, int(i[2])), loadHistogram(rh_dataset, int(global_cell[2])))
            
            mfcc_current = np.add(loadMFCC(mfcc_dataset, int(i[0])),loadMFCC(mfcc_dataset, int(i[1])), loadMFCC(mfcc_dataset, int(i[2])))
            mfcc_global = np.add(loadMFCC(mfcc_dataset, int(global_cell[0])),loadMFCC(mfcc_dataset, int(global_cell[1])), loadMFCC(mfcc_dataset, int(global_cell[2])))

            mfcc_distance = distance.cosine(mfcc_current,mfcc_global)

            fit_random.append(sc_distance + tiv.small_scale_compatibility(tiv_obj1['TIV'],tiv_obj2['TIV']) + rh_distance + rhythm_rh_distance +
                tiv.small_scale_compatibility(tiv_obj1['TIV'],tiv_obj3['TIV'])+ tiv.small_scale_compatibility(tiv_obj2['TIV'],tiv_obj4['TIV']) + rh_distance_global +
                mfcc_distance)
            line = line + 1

    #!!print(Ab_random)
    #!!print(fit_random)            

    #Average of initial population
    avg_fit_old = np.mean(fit_random)
    avg_fit = avg_fit_old - 1

#Main Loop
    it = 0
    while it <= iterations:
        #Erase same combinations
        Ab_random = np.unique(Ab_random, axis=0)
        #Cloning and mutation
        Ab_random = cloning(Ab_random, Nc, beta, fit_random, min_harmony,max_harmony,min_rhythm,max_rhythm, json_dataset, tiv_dataset, rh_dataset, sc_dataset, mfcc_dataset, global_cell, horizontal_stage, harmony_weight,rhythm_weight,sc_weight)
        #Ab_random = cloning2(Ab_random, Nc, beta, fit_random,harmonic_loops,rhythm_loops, min_harmony,min_rhythm, json_dataset, tiv_dataset, rh_dataset, sc_dataset, mfcc_dataset, global_cell, horizontal_stage)

        if(it > 20):
            if abs(1 - avg_fit_old / avg_fit) <= 0.001:
                #Suppress
                tiv_combinations = []
                line = 0
                if(horizontal_stage != True):
                    for i in Ab_random:
                        if(i[0] == i[1]):
                            Ab_random = np.delete(Ab_random, line, axis = 0)
                            fit_random = np.delete(fit_random,line)
                            continue
                        tiv_obj1=loadTIV(tiv_dataset, int(i[0]))
                        tiv_obj2=loadTIV(tiv_dataset, int(i[1]))
                        tiv_combine = tiv.combine(tiv_obj1['TIV'],tiv_obj2['TIV']).vector
                        rh_combine = np.add(loadHistogram(rh_dataset, int(i[0])), loadHistogram(rh_dataset, int(i[1])))
                        rh_combine_norm = rh_combine / np.linalg.norm(rh_combine)
                        rhythm_rh_combine = np.add(rh_combine, loadHistogram(rh_dataset, int(i[2])))
                        rhythm_rh_combine_norm = rhythm_rh_combine / np.linalg.norm(rhythm_rh_combine)
                        #print(sc_combine_norm)
                        #print(rh_combine_norm)
                        #print(rhythm_rh_combine_norm)
                        sc_distance = np.add(loadSC(sc_dataset, int(i[0])), loadSC(sc_dataset, int(i[1])))
                        sc_distance_norm = sc_distance / np.linalg.norm(sc_distance)


                        rhythm_sc_distance = np.add(sc_distance, loadSC(sc_dataset, int(i[2])))
                        rhythm_sc_distance_norm = rhythm_sc_distance / np.linalg.norm(rhythm_sc_distance)

                        tiv_combinations.append(np.concatenate((tiv_combine, rh_combine_norm, rhythm_rh_combine_norm,sc_distance_norm,rhythm_sc_distance_norm), axis=0))
                        line = line + 1
                else:
                    for i in Ab_random:
                        if(i[0] == i[1]):
                            Ab_random = np.delete(Ab_random, line, axis = 0)
                            fit_random = np.delete(fit_random,line)
                            continue
                        tiv_obj1=loadTIV(tiv_dataset, int(i[0]))
                        tiv_obj2=loadTIV(tiv_dataset, int(i[1]))
                        tiv_combine = tiv.combine(tiv_obj1['TIV'],tiv_obj2['TIV']).vector
                        rh_combine = np.add(loadHistogram(rh_dataset, int(i[0])), loadHistogram(rh_dataset, int(i[1])))
                        rh_combine_norm = rh_combine / np.linalg.norm(rh_combine)
                        rhythm_rh_combine = np.add(rh_combine, loadHistogram(rh_dataset, int(i[2])))
                        rhythm_rh_combine_norm = rhythm_rh_combine / np.linalg.norm(rhythm_rh_combine)

                        tiv_obj3=loadTIV(tiv_dataset, int(global_cell[0]))
                        tiv_obj4=loadTIV(tiv_dataset, int(global_cell[1]))
                        tiv_combine_global_1 = tiv.combine(tiv_obj1['TIV'],tiv_obj3['TIV']).vector
                        tiv_combine_global_2 = tiv.combine(tiv_obj2['TIV'],tiv_obj4['TIV']).vector

                        rh_distance_global = np.add(loadHistogram(rh_dataset, int(i[2])), loadHistogram(rh_dataset, int(global_cell[2])))
                        rh_distance_global_norm = rh_distance_global / np.linalg.norm(rh_distance_global)
                        
                        mfcc_current = np.add(loadMFCC(mfcc_dataset, int(i[0])),loadMFCC(mfcc_dataset, int(i[1])), loadMFCC(mfcc_dataset, int(i[2])))
                        mfcc_global = np.add(loadMFCC(mfcc_dataset, int(global_cell[0])),loadMFCC(mfcc_dataset, int(global_cell[1])), loadMFCC(mfcc_dataset, int(global_cell[2])))

                        mfcc_distance = np.add(mfcc_current,mfcc_global)
                        mfcc_distance_norm = mfcc_distance / np.linalg.norm(mfcc_distance)

                        tiv_combinations.append(np.concatenate((tiv_combine, rh_combine_norm, rhythm_rh_combine_norm, tiv_combine_global_1, tiv_combine_global_2, rh_distance_global_norm, mfcc_distance_norm), axis=0))
                        line = line + 1

                print('COMBINED TIV SPACE and RH OF EACH TWO ELEMENTS INSIDE THE CELL')

                dist_matrix = distance.cdist(tiv_combinations, tiv_combinations, 'cosine')
                #print(dist_matrix)

                row = 0
                cells_to_eliminate = []
                for i in dist_matrix:
                    element = 0
                    for j in i:
                        if (j < ts and row != element) or (j == float(0) and row != element):#SUPPRESSION THRESHOLD
                            print('HA SUPPRESSION THRESHOLD')
                            print(j)
                            #WARNING: PODE TAR MAL AQUI, PORQUE TENHO DE ELIMINAR OS QUE SAO IDENTICOS MAS ORDEM TROCADA [3,4] [4,3]
                            fitness_1 = fit_random[row]
                            #print("INDEX LINHA E ELEMENT")
                            #print(row, element)
                            #print("FIT VEC")
                            #print(fit_random)
                            
                            #print("FIT 1")
                            #print(fitness_1)
                            #print("FIT 2")
                            fitness_2 = fit_random[element]
                            #print(fitness_2)

                            if fitness_1 < fitness_2:
                                #print("ELEMENT")
                                #print(element)
                                cells_to_eliminate.append(element)
                                #Ab_random = np.delete(Ab_random, element, axis = 0)
                                #fit_random = np.delete(fit_random, element)
                            elif fitness_1 == fitness_2:
                                #print("FITNESS IGUAIS")
                                if element < row:
                                    cells_to_eliminate.append(element)
                                else:
                                    cells_to_eliminate.append(row)
                            else:
                                cells_to_eliminate.append(row)
                                #print("ROW")
                                #print(row)
                                #Ab_random = np.delete(Ab_random, row, axis = 0)
                                #fit_random = np.delete(fit_random, row)
                        element = element + 1
                    row = row + 1
                
                cells_to_eliminate = np.unique(cells_to_eliminate)
                #print(cells_to_eliminate)
                ind_cells_to_eliminate = np.argsort(cells_to_eliminate)
                #print(ind_cells_to_eliminate)

                for index in np.arange(len(cells_to_eliminate)):
                    Ab_random = np.delete(Ab_random, cells_to_eliminate[ind_cells_to_eliminate[index]] - index, axis = 0)
                    fit_random = np.delete(fit_random, cells_to_eliminate[ind_cells_to_eliminate[index]] - index)


    #!!            print("DEPOIS SUPRESSION")
    #!!            print(Ab_random)
    #!!            print(fit_random)

                if memory_cells == len(Ab_random) :
                    print("TERMINOU, NAO HA MAIS SOLUCOES")
                    print(len(Ab_random))
                    print(len(fit_random))
                    #No more solutions to cover, end while cycle
                    break
                else:
                    memory_cells = len(Ab_random)
                    print("NOVAS MEMORY CELLS")
                    print(memory_cells)
                
                d = round(0.4*N)

                new_proto_Ab_random = np.random.randint(min_harmony,max_harmony, size=(int(round(d)),2),dtype=int)
                new_column_Ab_random = np.random.randint(min_rhythm,max_rhythm, size=(int(round(d)),1), dtype=int)
                new_Ab_random = np.append(new_proto_Ab_random, new_column_Ab_random, axis=1)
                Ab_random = np.append(Ab_random, new_Ab_random, axis= 0)

                print("AB_RANDOM FINAL")
                print(Ab_random)
                            
                print("ESTABILIZOU")
            
            
        avg_fit_old = avg_fit
        avg_fit = np.mean(fit_random)
        
        #Eliminating duplicates
        Ab_random = np.unique(Ab_random, axis=0)
        
        fit_random = []
        line = 0
        if(horizontal_stage != True): 
            for i in Ab_random:
                if(i[0] == i[1]):
                    Ab_random = np.delete(Ab_random, line, axis = 0)
                    continue
                tiv_obj1=loadTIV(tiv_dataset, int(i[0]))
                tiv_obj2=loadTIV(tiv_dataset, int(i[1]))
                rh_distance = distance.cosine(loadHistogram(rh_dataset, int(i[0])), loadHistogram(rh_dataset, int(i[1])))

                rhythm_rh_distance = distance.cosine(np.add(loadHistogram(rh_dataset, int(i[0])),loadHistogram(rh_dataset, int(i[1]))), loadHistogram(rh_dataset, int(i[2])))

                sc_distance = spectral_balance(distance.euclidean(loadSC(sc_dataset, int(i[0])), loadSC(sc_dataset, int(i[1]))))
                sc_distance_r = spectral_balance(distance.euclidean(np.add(loadSC(sc_dataset, int(i[0])), loadSC(sc_dataset, int(i[1]))),loadSC(sc_dataset, int(i[2]))))
                fit_random.append((sc_distance_r+sc_distance)*sc_weight + tiv.small_scale_compatibility(tiv_obj1['TIV'],tiv_obj2['TIV'])*harmony_weight + (rh_distance + rhythm_rh_distance)*rhythm_weight)
                line = line + 1
        else:
            for i in Ab_random:
                if(i[0] == i[1]):
                    Ab_random = np.delete(Ab_random, line, axis = 0)
                    continue
                tiv_obj1=loadTIV(tiv_dataset, int(i[0]))
                tiv_obj2=loadTIV(tiv_dataset, int(i[1]))
                rh_distance = distance.cosine(loadHistogram(rh_dataset, int(i[0])), loadHistogram(rh_dataset, int(i[1])))
                rhythm_rh_distance = distance.cosine(np.add(loadHistogram(rh_dataset, int(i[0])),loadHistogram(rh_dataset, int(i[1]))), loadHistogram(rh_dataset, int(i[2])))

                sc_distance = spectral_balance(distance.euclidean(loadSC(sc_dataset, int(i[0])), loadSC(sc_dataset, int(i[1]))))

                tiv_obj3=loadTIV(tiv_dataset, int(global_cell[0]))
                tiv_obj4=loadTIV(tiv_dataset, int(global_cell[1]))
                rh_distance_global = distance.cosine(loadHistogram(rh_dataset, int(i[2])), loadHistogram(rh_dataset, int(global_cell[2])))
                
                mfcc_current = np.add(loadMFCC(mfcc_dataset, int(i[0])),loadMFCC(mfcc_dataset, int(i[1])), loadMFCC(mfcc_dataset, int(i[2])))
                mfcc_global = np.add(loadMFCC(mfcc_dataset, int(global_cell[0])),loadMFCC(mfcc_dataset, int(global_cell[1])), loadMFCC(mfcc_dataset, int(global_cell[2])))

                mfcc_distance = distance.cosine(mfcc_current,mfcc_global)

                fit_random.append(sc_distance + tiv.small_scale_compatibility(tiv_obj1['TIV'],tiv_obj2['TIV']) + rh_distance + rhythm_rh_distance +
                    tiv.small_scale_compatibility(tiv_obj1['TIV'],tiv_obj3['TIV'])+ tiv.small_scale_compatibility(tiv_obj2['TIV'],tiv_obj4['TIV']) + rh_distance_global +
                    mfcc_distance)
                line = line + 1
#!!        print('FINAL FIT ITERATED')
#!!        print(fit_random)
        
#!!        print('NEW VECTOR ADDED FOR NEXT ITERATION')
#!!        print(Ab_random)
        it = it+1
        print(it)
    print("ITEROU TUDO")
    
    print(it)
    sorted_fit_random = sorted(fit_random)
    print(sorted_fit_random)
    ind_sorted_Abrandom = np.argsort(fit_random)
    print(ind_sorted_Abrandom)
    
    print(Ab_random)
    value = str(raw_input("Listen to current iteration? Write <Yes>"))
    if(value == "Yes"):
        #se for horizontal, meter aqui o audio das chosen cells. nao ha else
        mix = []
        if(horizontal_stage == True):
            list_audio = AudioSegment.empty()
            for index in np.arange(len(chosen_cells)):
                obj_one = chosen_cells[index][0]
                obj_two = chosen_cells[index][1]
                obj_three = chosen_cells[index][2]
                filenames = loadLoop(obj_one,obj_two,obj_three)
                print(obj_one,obj_two,obj_three)
                print(filenames)
                print()

                audio1 = AudioSegment.from_file(filenames[0]) #your first audio file
                audio2 = AudioSegment.from_file(filenames[1]) #your second audio file
                audio3 = AudioSegment.from_file(filenames[2]) #your third audio file

                second_of_silence = AudioSegment.silent() # use default

                if len(audio2) < len(audio1):
                    audio2 *= int(round(len(audio1)/len(audio2))+1)
                elif len(audio2) > len(audio1):
                    audio1 *= int(round(len(audio2)/len(audio1))+1)

                mixed = audio1.overlay(audio2)        #Further combine , superimpose audio files
                
                if len(audio3) > len(audio1):
                    audio1 *= int(round(len(audio3)/len(audio1))+1)
                elif len(audio3) < len(audio1):
                    audio3 *= int(round(len(audio1)/len(audio3)))

                if len(audio3) > len(audio2):
                    audio2 *= int(round(len(audio3)/len(audio1))+1)
                elif len(audio3) < len(audio2):
                    audio3 *= int(round(len(audio2)/len(audio3)))
                    
                final_mixed = mixed.overlay(audio3)        #Further combine , superimpose audio files

                final_mixed = final_mixed.fade_in(20).fade_out(20)
                final_mixed = final_mixed.append(second_of_silence)
                list_audio = list_audio + final_mixed
                print(list_audio)
            mix = list_audio

            for index in np.arange(len(Ab_random)):
                obj_one = Ab_random[ind_sorted_Abrandom[index]][0]
                obj_two = Ab_random[ind_sorted_Abrandom[index]][1]
                obj_three = Ab_random[ind_sorted_Abrandom[index]][2]
                filenames = loadLoop(obj_one,obj_two,obj_three)
                print(obj_one,obj_two,obj_three)
                print(filenames)
                print()
                
                audio1 = AudioSegment.from_file(filenames[0]) #your first audio file
                audio2 = AudioSegment.from_file(filenames[1]) #your second audio file
                audio3 = AudioSegment.from_file(filenames[2]) #your third audio file
                print(len(audio1))
                print(len(audio2))
                print(len(audio3))
                if len(audio2) < len(audio1):
                    audio2 *= int(round(len(audio1)/len(audio2))+1)
                elif len(audio2) > len(audio1):
                    audio1 *= int(round(len(audio2)/len(audio1))+1)

                mixed = audio1.overlay(audio2)        #Further combine , superimpose audio files
                
                if len(audio3) > len(audio1):
                    audio1 *= int(round(len(audio3)/len(audio1))+1)
                elif len(audio3) < len(audio1):
                    audio3 *= int(round(len(audio1)/len(audio3)))

                if len(audio3) > len(audio2):
                    audio2 *= int(round(len(audio3)/len(audio1))+1)
                elif len(audio3) < len(audio2):
                    audio3 *= int(round(len(audio2)/len(audio3)))

                final_mixed = mixed.overlay(audio3)        #Further combine , superimpose audio files

                if len(audio3) < len(final_mixed):
                    final_mixed = final_mixed[:len(audio3)]
                #Chop at the end of the rhythm layer

                print(mix)
                copy_mix = mix.append(final_mixed)
                print(copy_mix)
                copy_mix.export("mashup" + "%d.wav" % (index), format='wav') #export mixed  audio file
                play(copy_mix.fade_in(20).fade_out(20))
            
        else:
            for index in np.arange(len(Ab_random)):
                obj_one = Ab_random[ind_sorted_Abrandom[index]][0]
                obj_two = Ab_random[ind_sorted_Abrandom[index]][1]
                obj_three = Ab_random[ind_sorted_Abrandom[index]][2]
                filenames = loadLoop(obj_one,obj_two,obj_three)
                print(obj_one,obj_two,obj_three)
                print(filenames)
                print()
                
                audio1 = AudioSegment.from_file(filenames[0]) #your first audio file
                audio2 = AudioSegment.from_file(filenames[1]) #your second audio file
                audio3 = AudioSegment.from_file(filenames[2]) #your third audio file
                print(len(audio1))
                print(len(audio2))
                print(len(audio3))
                if len(audio2) < len(audio1):
                    audio2 *= int(round(len(audio1)/len(audio2))+1)
                elif len(audio2) > len(audio1):
                    audio1 *= int(round(len(audio2)/len(audio1))+1)

                mixed = audio1.overlay(audio2)        #Further combine , superimpose audio files  
                if len(audio3) > len(audio1):
                    audio1 *= int(round(len(audio3)/len(audio1))+1)
                elif len(audio3) < len(audio1):
                    audio3 *= int(round(len(audio1)/len(audio3)))

                if len(audio3) > len(audio2):
                    audio2 *= int(round(len(audio3)/len(audio1))+1)
                elif len(audio3) < len(audio2):
                    audio3 *= int(round(len(audio2)/len(audio3)))

                final_mixed = mixed.overlay(audio3)        #Further combine , superimpose audio files

                if len(audio3) < len(final_mixed):
                    final_mixed = final_mixed[:len(audio3)]
                #Chop at the end of the rhythm layer
                final_mixed.export("loop" + "%d.wav" % (index), format='wav') #export mixed  audio file
                play(final_mixed.fade_in(20).fade_out(20))  

    else:
        print()
        print()
        print("Next opt-AiNet iteration!...")
    
    value = int(input("Which mashup do you pick? Write index of cell."))
    if(value in ind_sorted_Abrandom):
        global_cell = Ab_random[ind_sorted_Abrandom[value]]
        chosen_cells.append(global_cell)
    horizontal_stage = True

    end = time.time()
    print("TOTAL RUNTIME: " + str(end - begin)+ "s")

    return chosen_cells, global_cell, horizontal_stage


global_cell = []
horizontal_stage = False
iterations = int(input("Number of opt-AiNet iterations?"))
chosen_cells = []
it = 0
while(it < iterations):
    chosen_cells, global_cell, horizontal_stage = opt_music(chosen_cells,global_cell,horizontal_stage)
    value = str(raw_input("Listen to horizontal? Write <Yes>"))
    if(value == "Yes"):
        list_audio = AudioSegment.empty()
        for index in np.arange(len(chosen_cells)):
            obj_one = chosen_cells[index][0]
            obj_two = chosen_cells[index][1]
            obj_three = chosen_cells[index][2]
            filenames = loadLoop(obj_one,obj_two,obj_three)
            print(obj_one,obj_two,obj_three)
            print(filenames)
            print()

            audio1 = AudioSegment.from_file(filenames[0]) #your first audio file
            audio2 = AudioSegment.from_file(filenames[1]) #your second audio file
            audio3 = AudioSegment.from_file(filenames[2]) #your third audio file

            second_of_silence = AudioSegment.silent() # use default

            print(len(audio1))
            print(len(audio2))
            print(len(audio3))
            if len(audio2) < len(audio1):
                audio2 *= int(round(len(audio1)/len(audio2))+1)
            elif len(audio2) > len(audio1):
                audio1 *= int(round(len(audio2)/len(audio1))+1)

            mixed = audio1.overlay(audio2)        #Further combine , superimpose audio files
            
            
            if len(audio3) > len(audio1):
                audio1 *= int(round(len(audio3)/len(audio1))+1)
            elif len(audio3) < len(audio1):
                audio3 *= int(round(len(audio1)/len(audio3)))

            if len(audio3) > len(audio2):
                audio2 *= int(round(len(audio3)/len(audio1))+1)
            elif len(audio3) < len(audio2):
                audio3 *= int(round(len(audio2)/len(audio3)))
                
            final_mixed = mixed.overlay(audio3)        #Further combine , superimpose audio files

            final_mixed = final_mixed.fade_in(20).fade_out(20)

            if len(audio3) < len(final_mixed):
                final_mixed = final_mixed[:len(audio3)]
            #Chop at the end of the rhythm layer

            final_mixed = final_mixed.append(second_of_silence)
            final_mixed = final_mixed - audio
            list_audio = list_audio + final_mixed


        list_audio.export("final_mashup" + "%d.wav" % (index), format='wav') #export mixed  audio file
        play(list_audio)
    else:
        print("NEXT OPT-AINET ITERATION")    
    it = it + 1

print("FINISHED EVERY ITERATION")

