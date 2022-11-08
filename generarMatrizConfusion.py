import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt
import json


def matrizConfucion():

    mc = [[1851,  0, 29,    0,  0],
          [94, 29, 62, 1,  3],
          [302, 25, 991, 3, 23],
          [0, 0, 58, 10, 3],
          [18, 0,  116, 5,  136]]

    # Rendimiento del clasificador visual
    # load libraries
    img_total = [1880, 188, 1344, 71, 275]
    i = 0
    for fila in mc:
        div = []
        for ind in fila:
            div.append(ind/img_total[i])

        mc[i] = div
        i += 1

    class_names = ['None', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
    # crear marco de datos de pandas Crear un conjunto de datos
    dataframe = pd.DataFrame(mc, index=class_names, columns=class_names)

    # crear mapa de calor dibujar mapa de calor
    sns.heatmap(dataframe, annot=True, cbar=None, cmap="Blues", fmt='#.2%')
    plt.title("Confusion Matrix"), plt.tight_layout()
    plt.ylabel("True Class"), plt.xlabel("Predicted Class")
    plt.show()


def loadInfo(filename: str, array: str):

    with open(filename, 'r') as file:
        data = json.load(file)

    return data[array]


def grafLoss(fln1, fln2):
    data1 = loadInfo(fln1, 'loss')
    data2 = loadInfo(fln2, 'loss')

    fig, ax = plt.subplots()
    epocas = range(1, len(data1) + 1, 1)

    ax.plot(epocas, data1, marker='o')
    ax.plot(epocas, data2, marker='o')
    plt.legend(('ResNet', 'ConvNext'),
               prop={'size': 10}, loc='upper right')

    plt.ylabel('Loss')
    plt.xlabel('Epocas')
    plt.title('Perdida con el set de train de kaggle')
    plt.show()


def grafAcc(fln1, fln2):
    data1 = loadInfo(fln1, 'stats')
    data2 = loadInfo(fln2, 'stats')

    fig, ax = plt.subplots()
    print(len(data1))
    print(len(data2))
    epocas = range(1, len(data1) + 1, 1)

    arrays = {0: [], 1: []}

    for obj1, obj2 in zip(data1, data2):
        res, conv = obj1['acc'], obj2['acc']
        arrays[0].append(res)
        arrays[1].append(conv)

    ax.plot(epocas, arrays[0], marker='o')
    ax.plot(epocas, arrays[1], marker='o')

    plt.legend(('ResNet', 'ConvNext'),
               prop={'size': 10}, loc='lower right')

    plt.ylabel('acc')
    plt.xlabel('Epocas')
    plt.title('Acc con el set de valid de kaggle')
    plt.show()


def graficaApInTrain(filename: str):

    data = loadInfo(filename, 'stats')
    arrays = {0: [], 1: [], 2: [], 3: [], 4: []}

    for obj in data:
        none, mild, moderate, severe, pdr = obj['none'], obj['mild'], obj['moderate'], obj['severe'], obj['pro_dr']
        arrays[0].append(none)
        arrays[1].append(mild)
        arrays[2].append(moderate)
        arrays[3].append(severe)
        arrays[4].append(pdr)

    fig, ax = plt.subplots()
    epocas = range(1, len(arrays[0]) + 1, 1)

    ax.plot(epocas, arrays[0], marker='o')
    ax.plot(epocas, arrays[1], marker='o')
    ax.plot(epocas, arrays[2], marker='o')
    ax.plot(epocas, arrays[3], marker='o')
    ax.plot(epocas, arrays[4], marker='o')
    plt.legend(('None', 'Mild', 'Moderate', 'Severe', 'PDR'),
               prop={'size': 8}, loc='lower right')

    plt.ylabel('Precision')
    plt.xlabel('Epocas')
    plt.title('Precision por clase durante el entrenamiento con el set de valid DDR')
    plt.show()


# graficaApInTrain('JSONFiles/convnext/info_train_convnext.json')
# grafAcc('JSONFiles/resnet/info_train_resnet.json',
#         'JSONFiles/convnext/info_train_convnext.json')
matrizConfucion()
