import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def clean_matrix_name(path):
    """Extrae el nombre base de la matriz desde la ruta."""
    return os.path.splitext(os.path.basename(path))[0]

def compare_eigen_times(file1, file2):
    """
    Compara los tiempos de Eigen de dos archivos CSV y genera una tabla y un gráfico.
    """
    try:
        # Cargar los datos
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        print(f"Archivos cargados: '{file1}' y '{file2}'")
    except FileNotFoundError as e:
        print(f"Error: No se pudo encontrar el archivo {e.filename}")
        return

    # Limpiar los nombres de las matrices para poder unirlas
    df1['matrix_name'] = df1['matrix_name'].apply(clean_matrix_name)
    df2['matrix_name'] = df2['matrix_name'].apply(clean_matrix_name)

    # Seleccionar columnas relevantes y renombrar para la unión
    df1_times = df1[['matrix_name', 'tiempoEigen_ms']].rename(columns={'tiempoEigen_ms': 'tiempo_1_hilo'})
    df2_times = df2[['matrix_name', 'tiempoEigen_ms']].rename(columns={'tiempoEigen_ms': 'tiempo_max_hilos'})

    # Unir los dos DataFrames por el nombre de la matriz
    df_comp = pd.merge(df1_times, df2_times, on='matrix_name')

    # Calcular el speedup (evitando división por cero)
    df_comp['speedup'] = df_comp['tiempo_1_hilo'] / df_comp['tiempo_max_hilos'].replace(0, np.nan)

    # --- Imprimir la tabla comparativa ---
    print("\n--- Tabla Comparativa de Tiempos de Eigen ---")
    print(df_comp.to_string())
    
    avg_speedup = df_comp['speedup'].mean()
    print(f"\nSpeedup Promedio: {avg_speedup:.2f}x")
    print("=============================================\n")


    # --- Generar el gráfico de barras ---
    print("Generando gráfico de comparación...")
    df_plot = df_comp.sort_values('tiempo_1_hilo').set_index('matrix_name')

    ax = df_plot[['tiempo_1_hilo', 'tiempo_max_hilos']].plot(kind='bar', figsize=(15, 8), width=0.8)
    
    plt.title('Comparación de Tiempos de Eigen (1 Hilo vs. Máximos Hilos)', fontsize=16)
    plt.ylabel('Tiempo (ms)', fontsize=12)
    plt.xlabel('Matriz', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(['1 Hilo', 'Máx. Hilos'])
    
    # Añadir etiquetas de speedup sobre las barras
    for i, (idx, row) in enumerate(df_plot.iterrows()):
        speedup_val = row['speedup']
        if not pd.isna(speedup_val):
            ax.text(i, row['tiempo_1_hilo'] * 1.05, f'{speedup_val:.2f}x', ha='center', color='blue', fontweight='bold')

    plt.tight_layout()
    
    output_path = '../resultados/graficos/eigen_comparison.png'
    plt.savefig(output_path, dpi=150)
    print(f"Gráfico guardado en: '{output_path}'")
    plt.show()


if __name__ == '__main__':
    # Rutas a los archivos CSV (ajusta si es necesario)
    file_1_thread = '../resultados/resultados_eigen_1.csv'
    file_max_threads = '../resultados/resultados_eigen_2.csv'
    
    compare_eigen_times(file_1_thread, file_max_threads)
