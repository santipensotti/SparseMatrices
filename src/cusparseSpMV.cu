%%writefile spMV.cu
#include "mtxToCuda.h"          // Debe proveer CSRHost<T> y load_csr_from_mtx<T>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <Eigen/Sparse>


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
    int* d_x=nullptr;
    T   *d_vals=nullptr;
};



template<typename T>
descriptorDevice<T> to_device(const CSRHost<T>& H, const std::vector<T>& x) {
    descriptorDevice<T> D{H.rows, H.cols, H.nnz};
    CHECK_CUDA(cudaMalloc((void**)&D.d_rowptr, (H.rows+1)*sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&D.d_colind, H.nnz*sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&D.d_vals,   H.nnz*sizeof(T)));
    CHECK_CUDA(cudaMemcpy(D.d_rowptr, H.rowptr.data(), (H.rows+1)*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(D.d_colind, H.colind.data(), H.nnz*sizeof(int),      cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(D.d_vals,   H.vals.data(),   H.nnz*sizeof(T),        cudaMemcpyHostToDevice));

    // Allocate and copy x vector
    CHECK_CUDA(cudaMalloc((void**)&D.d_x, x.size()*sizeof(int)));
    CHECK_CUDA(cudaMemcpy(D.d_x, x.data(), x.size()*sizeof(int), cudaMemcpyHostToDevice));

    return D;
}

template<typename T>
void destroy_device(descriptorDevice<T>& D) {
    if (D.d_rowptr) cudaFree(D.d_rowptr);
    if (D.d_colind) cudaFree(D.d_colind);
    if (D.d_vals)   cudaFree(D.d_vals);
    if (D.d_x)     cudaFree(D.d_x);
    D = {};
}

double runSpMV(const descriptorDevice<float>& dA, const float* dx, float* dy, cusparseSpMatDescr_t& descr_A, const bool is_coo=false) {
    const int m = dA.rows, n = dA.cols, nnz = dA.nnz;

    cusparseHandle_t handle;  CHECK_CUSPARSE(cusparseCreate(&handle));
    const cusparseSpMVAlg_t alg = is_coo ? CUSPARSE_SPMV_COO_ALG1
                                         : CUSPARSE_SPMV_CSR_ALG2;


    cusparseDnVecDescr_t x_vec, y_vec;
    CHECK_CUSPARSE(cusparseCreateDnVec(&x_vec, n, (void*)dx, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&y_vec, m, (void*)dy, CUDA_R_32F));

    float alpha = 1.0f, beta = 0.0f;

    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, descr_A, x_vec, &beta, y_vec,
        CUDA_R_32F,
        alg,
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
    CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, descr_A, x_vec, &beta, y_vec, CUDA_R_32F,
        alg, dBuffer));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    
    float total_ms = 0.0f;
    for (int i = 0; i < iters; ++i) {
        cudaEvent_t s,e; CHECK_CUDA(cudaEventCreate(&s)); CHECK_CUDA(cudaEventCreate(&e));
        CHECK_CUDA(cudaEventRecord(s));
        CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, descr_A, x_vec, &beta, y_vec, CUDA_R_32F,
            alg, dBuffer));
        CHECK_CUDA(cudaEventRecord(e));
        CHECK_CUDA(cudaEventSynchronize(e));
        float ms=0; CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
        total_ms += ms;
        CHECK_CUDA(cudaEventDestroy(s)); CHECK_CUDA(cudaEventDestroy(e));
    }
    cudaFree(dBuffer);
    cusparseDestroyDnVec(x_vec);
    cusparseDestroyDnVec(y_vec);
    cusparseDestroy(handle);
    double ms = total_ms / iters;
    return ms;
}


void append_row_csv(const std::string& filename, int n, int k, int nnz, double t_csr, double t_coo, double t_eigen) {
    bool write_header = !std::filesystem::exists(csv_path)
                      || std::filesystem::file_size(csv_path) == 0;
    std::ofstream f(csv_path, std::ios::app);
    if (!f) { fprintf(stderr, "No pude abrir %s\n", csv_path.c_str()); return; }
    if (write_header) {
        f << "matrix_name,n,k,nnz,tiempoCSR_ms,tiempoCOO_ms,tiempoEigen_ms\n";
    }
    f << matrix_name << "," << n << "," << k << "," << nnz << ","
      << std::fixed << t_csr << "," << t_coo << "," << t_eigen << "\n";
}

template<typename T>
void errorGpuCpu(const Eigen::Map<const Eigen::Matrix<T, -1, 1>>& y_cpu,
                 const Eigen::Map<const Eigen::Matrix<T, -1, 1>>& y_gpu) {
    double abs_l2 = (y_cpu.template cast<double>() - y_gpu.template cast<double>()).norm();
    double rel_l2 = abs_l2 / (y_cpu.template cast<double>().norm() + 1e-30);
    double linf    = (y_cpu - y_gpu).template lpNorm<Eigen::Infinity>();

    printf("abs L2=%.3e  rel L2=%.3e  L_inf=%.3e\n", abs_l2, rel_l2, linf);
}

int main(int argc, char** argv) {
    const char* path = (argc > 1 ? argv[1] : "example.mtx");

    // 1) Cargar CSR (host)
    using T = float;
    CSRHost<T> H = load_csr_from_mtx<T>(path);
    printf("Matriz: %d x %d  nnz=%d\n", H.rows, H.cols, H.nnz);
    descriptorDevice<T> dA = to_device(H);

    // 3) Crear x,y (host) con tamaños correctos
    std::vector<T> hx(H.cols), hy(H.rows, 0.0f);
    std::srand(42);
    for (int i = 0; i < (int)hx.size(); ++i) hx[i] = (T)std::rand() / RAND_MAX;

    // 4) Reservar x,y en device y copiar
    T *dy=nullptr, *dy_2=nullptr; 
    CHECK_CUDA(cudaMalloc((void**)&dy, H.rows*sizeof(T)));
    CHECK_CUDA(cudaMalloc((void**)&dy_2, H.rows*sizeof(T)));
    CHECK_CUDA(cudaMemcpy(dy, hy.data(), H.rows*sizeof(T), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dy_2, hy.data(), H.rows*sizeof(T), cudaMemcpyHostToDevice));

    // 5) Ejecutar SpMV con Csr
    printf("Ejecutando SpMV...\n");
    const int m = dA.rows, n = dA.cols, nnz = dA.nnz;    

    cusparseSpMatDescr_t A_csr;
    CHECK_CUSPARSE(cusparseCreateCsr(
        &A_csr, m, n, nnz,
        (void*)dA.d_rowptr, (void*)dA.d_colind, (void*)dA.d_vals,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_R_32F
    ));
    
    double time_csr = runSpMV(dA, dA.d_x, dy, A_csr);
    printf("SpMV completado.\n");
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
    
    douuble time_coo = runSpMV(dA, dA.d_x, dy_2, A_coo, true);
    CHECK_CUSPARSE(cusparseDestroySpMat(A_coo));

    // 6) Traer resultado y mostrar algunas entradas
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hy.data(), dy, H.rows*sizeof(T), cudaMemcpyDeviceToHost));
    
    // 7) Cargar Matrix Eigen 
    Eigen::SparseMatrix<T, Eigen::ColMajor, int> A = load_eigen_from_mtx<T>(path);
    A.makeCompressed();
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> x_cpu(hx.data(), A.cols());

    // 8) Medir tiempo de SpMV en CPU (Eigen)
    auto t0 = std::chrono::high_resolution_clock::now();
    Eigen::Matrix<T, Eigen::Dynamic, 1> y_cpu = A * x_cpu;
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_cpu = std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    // 9) Comparar error Cpu y Gpu (Csr y Coo)
    // Eigen::Map<const Eigen::Matrix<T, -1, 1>> y_gpu(hy.data(), (int)hy.size());
    // CHECK_CUDA(cudaMemcpy(hy.data(), dy_2, H.rows*sizeof(T), cudaMemcpyDeviceToHost));
    // Eigen::Map<const Eigen::Matrix<T, -1, 1>> y_gpu_2(hy.data(), (int)hy.size());

    // errorGpuCpu(y_cpu, y_gpu);  
    // errorGpuCpu(y_cpu, y_gpu_2);

    // 10) Limpieza

    CHECK_CUDA(cudaFree(dy));
    CHECK_CUDA(cudaFree(dy_2));
    destroy_device(dA);
    CHECK_CUSPARSE(cusparseDestroy(handle));
    append_row_csv(path, m, n, nnz, time_csr, time_coo, ms_cpu);
    return 0;
}
