#include "helpers/mtxToCuda.h"          // Debe proveer CSRHost<T> y load_csr_from_mtx<T>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <string>
#include <Eigen/Sparse>

/*
The function performs the multiplication of a sparse matrix matA and a dense matrix matB.
where
op(A) is a sparse matrix of size 
op(B) is a dense matrix of size 
C is a dense matrix of size 
*/

// ---- Chequeo de errores (CUDA + cuSPARSE) ----
#define CHECK_CUDA(x)  do { cudaError_t err = (x); if (err != cudaSuccess) { \
  fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(1);} } while(0)

#define CHECK_CUSPARSE(x) do { cusparseStatus_t st = (x); if (st != CUSPARSE_STATUS_SUCCESS) { \
  fprintf(stderr, "cuSPARSE %s:%d: %d\n", __FILE__, __LINE__, (int)st); exit(1);} } while(0)

// ---- Contenedor device para CSR ----
template<typename T>
struct descriptorDevice {
    int rows=0, cols=0, nnz=0;
    int *d_rowptr=nullptr, 
    *d_colind=nullptr;
    T* d_vals=nullptr;
    T* d_B=nullptr; // Matriz densa B
    int B_cols = 0; // Numero de columnas de B
};



template<typename T>
descriptorDevice<T> to_device(const CSRHost<T>& H, const std::vector<T>& B, int B_cols) {
    descriptorDevice<T> D{H.rows, H.cols, H.nnz};
    D.B_cols = B_cols;
    CHECK_CUDA(cudaMalloc((void**)&D.d_rowptr, (H.rows+1)*sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&D.d_colind, H.nnz*sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&D.d_vals,   H.nnz*sizeof(T)));
    CHECK_CUDA(cudaMemcpy(D.d_rowptr, H.rowptr.data(), (H.rows+1)*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(D.d_colind, H.colind.data(), H.nnz*sizeof(int),      cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(D.d_vals,   H.vals.data(),   H.nnz*sizeof(T),        cudaMemcpyHostToDevice));

    // Allocate and copy B matrix (type T)
    CHECK_CUDA(cudaMalloc((void**)&D.d_B, B.size()*sizeof(T)));
    CHECK_CUDA(cudaMemcpy(D.d_B, B.data(), B.size()*sizeof(T), cudaMemcpyHostToDevice));

    return D;
}

template<typename T>
void destroy_device(descriptorDevice<T>& D) {
    if (D.d_rowptr) cudaFree(D.d_rowptr);
    if (D.d_colind) cudaFree(D.d_colind);
    if (D.d_vals)   cudaFree(D.d_vals);
    if (D.d_B)     cudaFree(D.d_B);
    D = {};
}

double runSpMM(const descriptorDevice<float>& dA, float* dC, cusparseSpMatDescr_t& descr_A, const bool is_coo=false) {
    const int m = dA.rows;
    const int n = dA.cols;
    const int k = dA.B_cols;

    cusparseHandle_t handle;  CHECK_CUSPARSE(cusparseCreate(&handle));
    
    cusparseDnMatDescr_t B_mat, C_mat;
    CHECK_CUSPARSE(cusparseCreateDnMat(&B_mat, n, k, n, (void*)dA.d_B, CUDA_R_32F, CUSPARSE_ORDER_COL));
    CHECK_CUSPARSE(cusparseCreateDnMat(&C_mat, m, k, m, (void*)dC, CUDA_R_32F, CUSPARSE_ORDER_COL));

    float alpha = 1.0f, beta = 0.0f;

    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, descr_A, B_mat, &beta, C_mat,
        CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        &bufferSize
    ));

    void* dBuffer = nullptr;
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    const int iters_warmup = 2, iters = 10;
    for (int i = 0; i < iters_warmup; ++i) {
        CHECK_CUSPARSE(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, descr_A, B_mat, &beta, C_mat, CUDA_R_32F,
            CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    float total_ms = 0.0f;
    for (int i = 0; i < iters; ++i) {
        cudaEvent_t s,e; CHECK_CUDA(cudaEventCreate(&s)); CHECK_CUDA(cudaEventCreate(&e));
        CHECK_CUDA(cudaEventRecord(s));
        CHECK_CUSPARSE(cusparseSpMM(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, descr_A, B_mat, &beta, C_mat, CUDA_R_32F,
            CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));
        CHECK_CUDA(cudaEventRecord(e));
        CHECK_CUDA(cudaEventSynchronize(e));
        float ms=0; CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
        total_ms += ms;
        CHECK_CUDA(cudaEventDestroy(s)); CHECK_CUDA(cudaEventDestroy(e));
    }
    cudaFree(dBuffer);
    cusparseDestroyDnMat(B_mat);
    cusparseDestroyDnMat(C_mat);
    cusparseDestroy(handle);
    double ms = total_ms / iters;
    return ms;
}


void append_row_csv(const std::string& csv_path, const std::string& matrix_name, int m, int n, int k, int nnz, double t_csr, double t_coo, double t_eigen) {
    bool write_header = true;
    {
        std::ifstream test(csv_path, std::ios::binary);
        if (test.good()) {
            test.seekg(0, std::ios::end);
            write_header = (test.tellg() == 0);
        } else {
            write_header = true; // no existe -> escribir cabecera
        }
    }
    std::ofstream f(csv_path, std::ios::app);
    if (!f) { fprintf(stderr, "No pude abrir %s\n", csv_path.c_str()); return; }
    if (write_header) {
        f << "matrix_name,m,n,k,nnz,tiempoCSR_ms,tiempoCOO_ms,tiempoEigen_ms\n";
    }
    f << matrix_name << "," << m << "," << n << "," << k << "," << nnz << ","
      << std::fixed << t_csr << "," << t_coo << "," << t_eigen << "\n";
}

template<typename T>
void errorGpuCpu(const Eigen::Matrix<T, -1, -1>& C_cpu,
                 const Eigen::Matrix<T, -1, -1>& C_gpu) {
    double abs_l2 = (C_cpu.template cast<double>() - C_gpu.template cast<double>()).norm();
    double rel_l2 = abs_l2 / (C_cpu.template cast<double>().norm() + 1e-30);
    double linf    = (C_cpu - C_gpu).template lpNorm<Eigen::Infinity>();

    printf("abs L2=%.3e  rel L2=%.3e  L_inf=%.3e\n", abs_l2, rel_l2, linf);
}

int main(int argc, char** argv) {
    const char* path = (argc > 1 ? argv[1] : "example.mtx");


    // 1) Cargar CSR (host)
    using T = float;
    CSRHost<T> H = load_csr_from_mtx<T>(path);
    printf("Matriz: %d x %d  nnz=%d\n", H.rows, H.cols, H.nnz);
    int k = H.rows;
    if (H.nnz == 0) return 1;
    // 3) Crear B y C en host
    std::vector<T> hB(H.cols * k);
    std::vector<T> hC(H.rows * k, 0.0f);
    std::srand(42);
    for (int i = 0; i < (int)hB.size(); ++i) hB[i] = (T)std::rand() / RAND_MAX;
    
    descriptorDevice<T> dA = to_device(H, hB, k);

    // 4) Reservar C en device y copiar
    T *dC=nullptr, *dC_2=nullptr; 
    CHECK_CUDA(cudaMalloc((void**)&dC, hC.size()*sizeof(T)));
    CHECK_CUDA(cudaMalloc((void**)&dC_2, hC.size()*sizeof(T)));
    CHECK_CUDA(cudaMemcpy(dC, hC.data(), hC.size()*sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC_2, hC.data(), hC.size()*sizeof(T), cudaMemcpyHostToDevice));

    // 5) Ejecutar SpMM con Csr
    printf("Ejecutando SpMM...\n");
    const int m = dA.rows, n = dA.cols, nnz = dA.nnz;    

    cusparseSpMatDescr_t A_csr;
    CHECK_CUSPARSE(cusparseCreateCsr(
        &A_csr, m, n, nnz,
        (void*)dA.d_rowptr, (void*)dA.d_colind, (void*)dA.d_vals,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F
    ));
    
    double time_csr = runSpMM(dA, dC, A_csr);
    printf("SpMM completado.\n");
    CHECK_CUSPARSE(cusparseDestroySpMat(A_csr));

    // 5) COO
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));
    int *A_coo_rows = nullptr;
    CHECK_CUDA(cudaMalloc(&A_coo_rows, nnz * sizeof(int)));
    CHECK_CUSPARSE(cusparseXcsr2coo(handle, dA.d_rowptr, nnz, m,
                           A_coo_rows, CUSPARSE_INDEX_BASE_ZERO));
    
    cusparseSpMatDescr_t A_coo;
    CHECK_CUSPARSE(cusparseCreateCoo(&A_coo, m, n, nnz,
                            A_coo_rows, (void*)dA.d_colind, (void*)dA.d_vals,
                            CUSPARSE_INDEX_32I,
                            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    
    double time_coo = runSpMM(dA, dC_2, A_coo, true);
    CHECK_CUSPARSE(cusparseDestroySpMat(A_coo));

    // 6) Traer resultado y mostrar algunas entradas
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hC.data(), dC, hC.size()*sizeof(T), cudaMemcpyDeviceToHost));
    
    // 7) Cargar Matrix Eigen 
    Eigen::SparseMatrix<T, Eigen::ColMajor, int> A = load_eigen_from_mtx<T>(path);
    A.makeCompressed();
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>> B_cpu(hB.data(), A.cols(), k);

    // 8) Medir tiempo de SpMV en CPU (Eigen)
    auto t0 = std::chrono::high_resolution_clock::now();
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> C_cpu = A * B_cpu;
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_cpu = std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    // 9) Comparar error Cpu y Gpu (Csr y Coo)
    Eigen::Map<const Eigen::Matrix<T, -1, -1, Eigen::ColMajor>> C_gpu(hC.data(), m, k);
    CHECK_CUDA(cudaMemcpy(hC.data(), dC_2, hC.size()*sizeof(T), cudaMemcpyDeviceToHost));
    Eigen::Map<const Eigen::Matrix<T, -1, -1, Eigen::ColMajor>> C_gpu_2(hC.data(), m, k);

    // errorGpuCpu(C_cpu, C_gpu);  
    // errorGpuCpu(C_cpu, C_gpu_2);

    // 10) Limpieza

    CHECK_CUDA(cudaFree(dC));
    CHECK_CUDA(cudaFree(dC_2));
    destroy_device(dA);
    CHECK_CUSPARSE(cusparseDestroy(handle));
    append_row_csv("SpMM_results.csv", path, m, n, k, nnz, time_csr, time_coo, ms_cpu);
    return 0;
}

