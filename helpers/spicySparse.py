# tools/fetch_ssm.py
import ssgetpy, os, sys, yaml
out = "data/ssm"; os.makedirs(out, exist_ok=True)
cfg = yaml.safe_load(open("./data/matrices.yaml"))
# Se elimina el [0] para iterar sobre todas las matrices
names = [m["name"] for s in cfg["sets"].values() for m in s]
for name in names:
    # ssgetpy.search devuelve una lista, tomamos el primer resultado
    mat = ssgetpy.search(name=name)[0]
    # Descargamos la matriz
    mat.download(destpath=out, format='MM', extract=True)
    
    # Creamos el nuevo nombre y renombramos el directorio
    original_dir = os.path.join(out, name)
    new_name = f"{name}_{mat.kind}"
    new_dir = os.path.join(out, new_name)
    
    # Verificamos si el directorio original existe antes de renombrar
    if os.path.isdir(original_dir):
        os.rename(original_dir, new_dir)
        print("OK", name, "->", new_dir)
    else:
        print(f"WARN: No se encontró el directorio {original_dir} para renombrar.")

