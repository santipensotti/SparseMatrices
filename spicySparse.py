import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve_triangular
from scipy.linalg import solve_triangular as dense_solve_triangular
import time
import os

def load_system(path):
    with open(path, "r") as f:
        try:
            header = f.readline().split()
            if len(header) < 3:
                raise ValueError("El encabezado del archivo de matriz es inválido.")
            n, nrhs, nnz = map(int, header)
            
            rowptrs_str = f.readline()
            colidxs_str = f.readline()
            vals_str = f.readline()

            rowptrs = np.fromstring(rowptrs_str, dtype=int, count=n + 1, sep=' ')
            colidxs = np.fromstring(colidxs_str, dtype=int, count=nnz, sep=' ')
            vals = np.fromstring(vals_str, dtype=float, count=nnz, sep=' ')

        except (ValueError, IndexError) as e:
            raise ValueError(f"Error al procesar el archivo {path}: {e}")


    A = csr_matrix((vals, colidxs, rowptrs), shape=(n, n))
    
    rng = np.random.RandomState(42)
    B = rng.rand(n, nrhs)  
    return A, B

def bench_file(path, warmup=2, repeats=10):
    
    print(f"--- Benchmarking: {os.path.basename(path)} ---")
    A_sparse, B = load_system(path)
    n = A_sparse.shape[0]

    # --- 1. Benchmark  (Sparse) ---
    total_sparse = 0.0
    for _ in range(warmup):
        _ = spsolve_triangular(A_sparse, B, lower=True, unit_diagonal=False)
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = spsolve_triangular(A_sparse, B, lower=True, unit_diagonal=False)
        t1 = time.perf_counter()
        total_sparse += (t1 - t0)
    avg_sparse_ms = (total_sparse / repeats) * 1000.0
    print(f"Dispersa (Sparse),{n}x{n},{avg_sparse_ms:.6f}")


    # --- 2. Benchmark (Dense) ---
    A_dense = A_sparse.toarray()

    total_dense = 0.0
    for _ in range(warmup):
        _ = dense_solve_triangular(A_dense, B, lower=True)
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = dense_solve_triangular(A_dense, B, lower=True)
        t1 = time.perf_counter()
        total_dense += (t1 - t0)
    avg_dense_ms = (total_dense / repeats) * 1000.0
    print(f"Densa (Dense),{n}x{n},{avg_dense_ms:.6f}")
    print("-" * (20 + len(os.path.basename(path))))


if __name__ == "__main__":
    files = ["matrix13.txt", "matrix16.txt", "matrix20.txt", "matrix25.txt"]
    print("type,size,time (ms)")
    for path in files:
        if os.path.exists(path):
            bench_file(path)
        else:
            print(f"Error '{path}'.")
