from ssgetpy import fetch
import os
import ssgetpy

result = ssgetpy.fetch(group='Andrianov', name='lp1', format='MM')  # descarga el .mtx

print(result)  # muestra el resultado de la descarga
current_path = os.getcwd()
print("Ruta actual:", current_path)
print(result.download(format='MM', extract=True, destpath=current_path))  # extrae el archivo descargado en la ruta actual