import torch
import json
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
from collections import Counter

from tqdm import tqdm

def loadInfo(filename: str):

    with open(filename, 'r') as file:
        data = json.load(file)

    return data

def returnMetricsEvaluation(filename, set = 'test', modo = 'max'):
    data = loadInfo(filename)

    dict_results = {0 : [],
                    1 : [],
                    2 : [],
                    3 : [],
                    4 : [],
                    5 : [],
                    6 : [],
                    7 : [],
                    8 : []}
    
    labels = []

    bar = tqdm(enumerate(data), total= len(data))

    for _ , obj in bar:
        matrix = obj.get('matrix')
        label = obj.get('label')
        labels.append(label)

        if set == 'test':
            matrix.pop(3)
            matrix.pop(1)

        if modo == 'promedio_scores':
            prom_result = [0., 0., 0., 0., 0.]
            for i, model_result in enumerate(matrix):
                for j in range(len(model_result)):
                    prom_result[j] += model_result[j]
            prom_result = list(map(lambda x: x / 9, prom_result))

            pred = int(torch.argmax(torch.FloatTensor(prom_result)))
            dict_results[0].append(pred)
                    


        if modo == 'uno':   
            for i, model_result in enumerate(matrix):
                pred = int(torch.argmax(torch.FloatTensor(model_result)))
                dict_results[i].append(pred)
        
        if modo == 'promedio':
            prom = []
            for i, model_result in enumerate(matrix):
                pred = int(torch.argmax(torch.FloatTensor(model_result)))
                # dict_results[0].append(pred)
                prom.append(pred)

            dict_results[0].append(sum(prom) // 5)
        

        if modo == 'max_elemento':
            prom = []
            for i, model_result in enumerate(matrix):
                pred = int(torch.argmax(torch.FloatTensor(model_result)))
                # dict_results[0].append(pred)
                prom.append(pred)

            dict_results[0].append(max(prom))

        if modo == 'moda':
            prom = []
            
            prom_result = [0., 0., 0., 0., 0.]
            for i, model_result in enumerate(matrix):
                pred = int(torch.argmax(torch.FloatTensor(model_result)))
                prom.append(pred)

                for j in range(len(model_result)):
                    prom_result[j] += model_result[j]

            prom_result = list(map(lambda x: x / 9, prom_result))

            first = Counter(prom).most_common()

            label = prom_result.index(max(prom_result)) # Mayor paso de etiqueta prioridad

            # Verificamos si hay dos clases con la misma cantidad de aciertos
            empatados = []
            ant = 0
            #print(first)
            for indice , tupla in enumerate(first):
                if indice == 0:
                    ant = tupla[1]
                    empatados.append(tupla[0])
                    continue

                if ant == tupla[1]:
                    empatados.append(tupla[0])
            
            #print('empatados: ', empatados)
            #print('mejor_promediada: ', label)

            if len(empatados) > 1:
                dict_results[0].append(label)
            else:
                dict_results[0].append(first[0][0])

    if modo == 'uno':
        for i, res in enumerate(dict_results.values()):
            print('Acc Modelo {} : {}'.format(i,accuracy_score(labels, res)))
        return
    
    mc = confusion_matrix(labels, dict_results.get(0)).tolist()

    train = [3133, 315, 2238, 118, 456]
    valid = [1253, 126, 895, 47, 182]
    test = [1880, 188, 1344, 71, 275]

    dicts = {'test': test,
             'valid': valid,
             'train': train}

    for i in range(5):
        print('{} : {}'.format(i, mc[i][i] / dicts[set][i]))


    #print(confusion_matrix(labels, dict_results.get(0)).tolist())
    print(accuracy_score(labels, dict_results.get(0)))

returnMetricsEvaluation('JSONFiles/predictions/DDR_valid.json', set='valid', modo='uno')
returnMetricsEvaluation('JSONFiles/predictions/DDR_test.json', set='test', modo='uno')
returnMetricsEvaluation('JSONFiles/predictions/DDR_train.json', set='train', modo='uno')