import json
import torch

def loadInfo(filename: str):

    with open(filename, 'r') as file:
        data = json.load(file)

    return data

def getImagenes():

    models_strings = ['0000', '0001', '0011', '0101', '0111', '1001', '1011', '1101', '1111']
    
    data = loadInfo('JSONFiles/predictions/DDR_valid.json')

    pred = []
    
    for i, obj in enumerate(data):
        
        name = obj['filename']
        label = obj['label']
        preds = []
        if label != 0:
            continue
        for v in obj['matrix']:
            preds.append(int(torch.argmax(torch.FloatTensor(v))))

        cont = 0
        str_m = []
        for j ,p in enumerate(preds):
            if p != label:
                str_m.append(models_strings[j])
                cont += 1
        
        if len(str_m) < 1 :# and '0000' in str_m and '0001' in str_m:
            pred.append({
                'modelos_errores' : str_m,
                'name': name
            })
        

    with open('p.json', 'w') as file:
        json.dump(pred, file)
    print(pred)

# DR1
#/home/bringascastle/Documentos/datasets-retina/DDR-dataset/DR_grading/valid/007-2685-100.jpg'
# DR3
# [{'modelos_errores': ['0000', '0001', '0111', '1111'], 'name': '/home/bringascastle/Documentos/datasets-retina/DDR-dataset/DR_grading/valid/007-5573-300.jpg'}, {'modelos_errores': ['0000', '0001', '0011', '1111'], 'name': '/home/bringascastle/Documentos/datasets-retina/DDR-dataset/DR_grading/valid/007-5830-300.jpg'}]

getImagenes()
