
import xlrd
import shutil

archivo = './../Annotation_Base11.xls'

wb = xlrd.open_workbook(archivo)
hoja = wb.sheet_by_index(0)

names = []
grades = []

for fila in range(1, hoja.nrows):
    names.append(str(hoja.cell_value(fila, 0)))
    grades.append(int(hoja.cell_value(fila, 2)))

print(len(names))
archivo = './../Annotation Base12.xls'

wb = xlrd.open_workbook(archivo)
hoja = wb.sheet_by_index(0)

for fila in range(1, hoja.nrows):
    names.append(str(hoja.cell_value(fila, 0)))
    grades.append(int(hoja.cell_value(fila, 2)))


archivo = './../Annotation_Base13.xls'

wb = xlrd.open_workbook(archivo)
hoja = wb.sheet_by_index(0)

for fila in range(1, hoja.nrows):
    names.append(str(hoja.cell_value(fila, 0)))
    grades.append(int(hoja.cell_value(fila, 2)))

archivo = './../Annotation Base14.xls'

wb = xlrd.open_workbook(archivo)
hoja = wb.sheet_by_index(0)

for fila in range(1, hoja.nrows):
    names.append(str(hoja.cell_value(fila, 0)))
    grades.append(int(hoja.cell_value(fila, 2)))


carpetaOrigen = '/home/bringascastle/Vídeos/datasets-retina/drive-download-20220712T015923Z-001/Messidor/Base1/'
carpetaDestino = '/home/bringascastle/Documentos/datasets-dr/messidor/'
i = 0
for name, grad in zip(names, grades):
    i += 1
    try:
        origen = r'' + carpetaOrigen + name
        destino = r'' + carpetaDestino + str(grad) + '/' + name
        shutil.copy(origen, destino)
    except:
        continue


print("Se movieron ", i)
