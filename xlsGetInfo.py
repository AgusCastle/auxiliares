import xlrd
import shutil

archivo = '/Vídeos/Datasets Retina/drive-download-20220712T015923Z-001/Messidor/Base1/Annotation_Base11.xls'

wb = xlrd.open_workbook(archivo)
hoja = wb.sheet_by_index(0)

names = []
grades = []

for fila in range(1, hoja.nrows):
    names.append(str(hoja.cell_value(fila, 0)))
    grades.append(int(hoja.cell_value(fila, 2)))


archivo = '/Vídeos/Datasets Retina/drive-download-20220712T015923Z-001/Messidor/Base1/Annotation_Base12.xls'

wb = xlrd.open_workbook(archivo)
hoja = wb.sheet_by_index(0)

for fila in range(1, hoja.nrows):
    names.append(str(hoja.cell_value(fila, 0)))
    grades.append(int(hoja.cell_value(fila, 2)))


archivo = '/Vídeos/Datasets Retina/drive-download-20220712T015923Z-001/Messidor/Base1/Annotation_Base13.xls'

wb = xlrd.open_workbook(archivo)
hoja = wb.sheet_by_index(0)

for fila in range(1, hoja.nrows):
    names.append(str(hoja.cell_value(fila, 0)))
    grades.append(int(hoja.cell_value(fila, 2)))


carpetaOrigen = '/Vídeos/Datasets Retina/drive-download-20220712T015923Z-001/Messidor/Base1/'
carpetaDestino = '/Documentos/datsets-dr/messidor/'

for name, grad in zip(names, grades):
    origen = r'' + carpetaOrigen + name
    destino = r'' + carpetaDestino + str(grad) + '/' + name

    shutil.move(origen, destino)
