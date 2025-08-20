import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import ssgetpy

def plot_vs_nnz(df, nombre_csv):
    """ Opción B: Por cantidad de nnz """
    print("Generando gráfico: Tiempo vs NNZ...")
    sorted_nnz = df.sort_values("nnz")
    plt.figure(figsize=(10,6))
    plt.plot(sorted_nnz["nnz"], sorted_nnz["tiempoCSR_ms"], marker="o", label="CSR")
    plt.plot(sorted_nnz["nnz"], sorted_nnz["tiempoCOO_ms"], marker="s", label="COO")
    plt.plot(sorted_nnz["nnz"], sorted_nnz["tiempoEigen_ms"], marker="^", label="Eigen")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("NNZ (número de elementos no cero)")
    plt.ylabel("Tiempo (ms)")
    plt.title("Tiempo vs NNZ (CSR, COO, Eigen)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(nombre_csv + "_tiempo_vs_NNZ.png", dpi=160)
    plt.close()

def plot_vs_size_chem(df, nombre_csv):
    """ Opción C: Plotear solo las que son de quimica """
    print("Generando gráfico: Tiempo vs Tamaño para problemas de química...")
    df_chem = df[df["kind"] == "chemical process simulation problem"].sort_values("n_k")
    if df_chem.empty:
        print("No se encontraron matrices de 'chemical process simulation problem'.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(df_chem["n_k"], df_chem["tiempoCSR_ms"], marker="o", linestyle="-", label="(CSR)")
    plt.plot(df_chem["n_k"], df_chem["tiempoCOO_ms"], marker="s", linestyle="--", label="(COO)")
    plt.plot(df_chem["n_k"], df_chem["tiempoEigen_ms"], marker="^", linestyle=":", label=" (Eigen)")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("N*K (tamaño total de la matriz densa)")
    plt.ylabel("Tiempo (ms)")
    plt.title("Tiempo vs Tamaño (N*K) para 'chemical process simulation problem'")
    plt.legend()
    plt.savefig(nombre_csv + "_tiempo_vs_size_chem.png", dpi=160)
    plt.close()

def plot_vs_size_kind(df, nombre_csv):
    """ Opcion D: Plotear por kind """
    print("Generando gráfico: Tiempo vs Tamaño por tipo de problema...")
    df_kind = df[df["kind"] != "unknown"].sort_values("nnz")
    if df_kind.empty:
        print("No se encontraron 'kinds' de matrices para graficar.")
        return

    plt.figure(figsize=(10, 6))
    for kind, group in df_kind.groupby("kind"):
        plt.plot(group["n_k"], group["tiempoCSR_ms"], marker="o", linestyle="-", label=f"(CSR) - {kind}")
        plt.plot(group["n_k"], group["tiempoCOO_ms"], marker="s", linestyle="--", label=f"(COO) - {kind}")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("N*K (tamaño total de la matriz densa)")
    plt.ylabel("Tiempo (ms)")
    plt.title("Tiempo vs Tamaño (N*K) por tipo de problema")
    plt.legend()
    plt.savefig(nombre_csv + "_tiempo_vs_size_kind.png", dpi=160)
    plt.close()

def main():

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = "../resultados/resultados_todas.csv"
    
    if not os.path.exists(csv_path):
        print(f"Error: El archivo {csv_path} no existe.")
        sys.exit(1)

    nombre_csv = os.path.splitext(os.path.basename(csv_path))[0]
    df = pd.read_csv(csv_path)
    df["matrix_name"] = df["matrix_name"].apply(lambda x: os.path.splitext(os.path.basename(x))[0])
    df["matrix_name"] = df["matrix_name"].str.replace("_b$", "", regex=True)

    numeric_cols = ["n", "k", "nnz", "tiempoCSR_ms", "tiempoCOO_ms", "tiempoEigen_ms"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=numeric_cols)
    df["n_k"] = df["n"] * df["k"]
    df = df[df["n"] > 0].sort_values(["n", "nnz"])

    df["kind"] = df["matrix_name"].apply(lambda x: ssgetpy.search(name=x)[0].kind if ssgetpy.search(name=x) else "unknown")
    
    plot_functions = {
        "vs_nnz": plot_vs_nnz,
        "chem": plot_vs_size_chem,
        "kind": plot_vs_size_kind,
    }

    plot_to_run = sys.argv[2] if len(sys.argv) > 2 else None

    if plot_to_run:
        if plot_to_run in plot_functions:
            plot_functions[plot_to_run](df, nombre_csv)
        else:
            print(f"Error: El tipo de gráfico '{plot_to_run}' no es válido.")
            print(f"Opciones válidas: {list(plot_functions.keys())}")
            sys.exit(1)
    else:
        print("Generando todos los gráficos...")
        for func in plot_functions.values():
            func(df, nombre_csv)
    # borra las que tienen nnz = 0
    df = df[df["nnz"] > 0]
    # differences between csr and eigen; and coo with eigen
    df['diff_csr_eigen'] = df['tiempoCSR_ms'] - df['tiempoEigen_ms']
    df['diff_coo_eigen'] = df['tiempoCOO_ms'] - df['tiempoEigen_ms']

    # Mostrar los top 10 con las diferencias más pequeñas (más negativas => método más rápido que Eigen)
    print("\nTop 10: CSR más rápido que Eigen (diff_csr_eigen más negativo):")
    print(df.sort_values('diff_csr_eigen').head(10)[
        ['matrix_name', 'n', 'k', 'nnz', 'tiempoCSR_ms', 'tiempoEigen_ms', 'diff_csr_eigen']
    ].to_string(index=False))

    print("\nTop 10: COO más rápido que Eigen (diff_coo_eigen más negativo):")
    print(df.sort_values('diff_coo_eigen').head(10)[
        ['matrix_name', 'n', 'k', 'nnz', 'tiempoCOO_ms', 'tiempoEigen_ms', 'diff_coo_eigen']
    ].to_string(index=False))

if __name__ == "__main__":
    main()
