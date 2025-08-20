#!/bin/bash

# --- Configuración ---
# Directorio donde se encuentran las matrices en formato .mtx
# La estructura esperada es: ssm/GROUP/MATRIX.mtx
MATRIX_DIR="data/ssm"
# Ruta al ejecutable del benchmark
EXECUTABLE="./bin/ejemplo"

# --- Script ---

# 1. Verificar que el ejecutable exista
if [ ! -f "$EXECUTABLE" ]; then
    echo "Error: El ejecutable '$EXECUTABLE' no fue encontrado."
    echo "Por favor, compila el proyecto primero ejecutando 'make'."
    exit 1
fi

# 2. Verificar que el directorio de matrices exista
if [ ! -d "$MATRIX_DIR" ]; then
    echo "Error: El directorio de matrices '$MATRIX_DIR' no fue encontrado."
    echo "Asegúrate de que las matrices estén en la carpeta '$MATRIX_DIR' en la raíz del proyecto."
    exit 1
fi

# 3. Encontrar y procesar todos los archivos .mtx en el directorio
# El comando 'find' busca recursivamente todos los archivos que terminen en .mtx
find "$MATRIX_DIR" -type f -name "*.mtx" ! -name "*_b.mtx" ! -name "*_rhs.mtx" ! -name "*_xy.mtx" ! -name "*_coord.mtx" | while read -r matrix_path; do    echo "============================================================"
    echo "Procesando matriz: $matrix_path"
    echo "============================================================"
    
    # Ejecutar el benchmark pasándole la ruta del archivo .mtx
    "$EXECUTABLE" "$matrix_path"
    
    echo "" # Añadir un espacio para mayor legibilidad
done

echo "Todos los benchmarks han sido completados."
