#include "helpers/mtxToCuda.h"
#include <cusparse.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <string>
#include <Eigen/Sparse>

#define CHECK_CUDA(x)  do { cudaError_t err = (x); if (err != cudaSuccess) { \
  fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); exit(1);} } while(0)
#define CHECK_CUSPARSE(x) do { cusparseStatus_t st = (x); if (st != CUSPARSE_STATUS_SUCCESS) { \
  fprintf(stderr, "cuSPARSE %s:%d: %d\n", __FILE__, __LINE__, (int)st); exit(1);} } while(0)

template<typename T>
struct descriptorDevice {
    int rows=0, cols=0, nnz=0;
    int *d_rowptr=nullptr, *d_colind=nullptr;
    T   *d_vals=nullptr, *d_x=nullptr;
};

template<typename T>
descriptorDevice<T> to_device(const CSRHost<T>& H, const std::vector<T>& x) {
    descriptorDevice<T> D{H.rows, H.cols, H.nnz};
    CHECK_CUDA(cudaMalloc((void**)&D.d_rowptr, (size_t)(H.rows+1)*sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&D.d_colind, (size_t)H.nnz*sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&D.d_vals,   (size_t)H.nnz*sizeof(T)));
    CHECK_CUDA(cudaMemcpy(D.d_rowptr, H.rowptr.data(), (size_t)(H.rows+1)*sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(D.d_colind, H.colind.data(), (size_t)H.nnz*sizeof(int),      cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(D.d_vals,   H.vals.data(),   (size_t)H.nnz*sizeof(T),        cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc((void**)&D.d_x, (size_t)x.size()*sizeof(T)));
    CHECK_CUDA(cudaMemcpy(D.d_x, x.data(), (size_t)x.size()*sizeof(T), cudaMemcpyHostToDevice));
    return D;
}

template<typename T>
void destroy_device(descriptorDevice<T>& D) {
    if (D.d_rowptr) cudaFree(D.d_rowptr);
    if (D.d_colind) cudaFree(D.d_colind);
    if (D.d_vals)   cudaFree(D.d_vals);
    if (D.d_x)      cudaFree(D.d_x);
    D = {};
}

double runSpMV_once(cusparseHandle_t handle,
                    const cusparseSpMatDescr_t descr_A,
                    int m, int n,
                    const float* dx, float* dy,
                    bool is_coo=false) {
    cusparseDnVecDescr_t x_vec=nullptr, y_vec=nullptr;
    CHECK_CUSPARSE(cusparseCreateDnVec(&x_vec, (int64_t)n, (void*)dx, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&y_vec, (int64_t)m, (void*)dy, CUDA_R_32F));
    float alpha = 1.0f, beta = 0.0f;
    const cusparseSpMVAlg_t alg = is_coo ? CUSPARSE_SPMV_COO_ALG1
                                         : CUSPARSE_SPMV_CSR_ALG1;

    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, descr_A, x_vec, &beta, y_vec, CUDA_R_32F, alg, &bufferSize));

    void* dBuffer = nullptr;
    if (bufferSize > 0) CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));


    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    const int iters = 1;
    const int warm_up = 0;
    for (int i=0; i < warm_up; ++i){
        CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, descr_A, x_vec, &beta, y_vec, CUDA_R_32F, alg, dBuffer));
    }
    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, descr_A, x_vec, &beta, y_vec, CUDA_R_32F, alg, dBuffer));
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float total_ms=0.0f; CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));
    double ms = total_ms / iters;

    if (dBuffer) cudaFree(dBuffer);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cusparseDestroyDnVec(x_vec); cusparseDestroyDnVec(y_vec);
    return ms;
}

void append_row_csv(const std::string& csv_path, const std::string& matrix_name, int n, int k, int nnz, double t_csr, double t_coo, double t_eigen) {
    bool write_header = true;
    {
        std::ifstream test(csv_path, std::ios::binary);
        if (test.good()) { test.seekg(0, std::ios::end); write_header = (test.tellg() == 0); }
    }
    std::ofstream f(csv_path, std::ios::app);
    if (!f) { fprintf(stderr, "No pude abrir %s\n", csv_path.c_str()); return; }
    if (write_header) f << "matrix_name,n,k,nnz,tiempoCSR_ms,tiempoCOO_ms,tiempoEigen_ms\n";
    f << matrix_name << "," << n << "," << k << "," << nnz << ","
      << std::fixed << t_csr << "," << t_coo << "," << t_eigen << "\n";
}

int main(int argc, char** argv) {
    const char* path = (argc > 1 ? argv[1] : "example.mtx");
    Eigen::initParallel();
    using T = float;
    CSRHost<T> H = load_csr_from_mtx<T>(path);
    printf("Matriz: %d x %d  nnz=%d\n", H.rows, H.cols, H.nnz);

    std::vector<T> hx(H.cols), hy(H.rows, 0.0f);
    std::srand(42);
    for (int i = 0; i < (int)hx.size(); ++i) hx[i] = (T)std::rand() / RAND_MAX;
    descriptorDevice<T> dA = to_device(H, hx);

    T *dy=nullptr, *dy_2=nullptr;
    CHECK_CUDA(cudaMalloc((void**)&dy,   (size_t)H.rows*sizeof(T)));
    CHECK_CUDA(cudaMalloc((void**)&dy_2, (size_t)H.rows*sizeof(T)));
    CHECK_CUDA(cudaMemset(dy,   0, (size_t)H.rows*sizeof(T)));
    CHECK_CUDA(cudaMemset(dy_2, 0, (size_t)H.rows*sizeof(T)));

    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // CSR descriptor
    cusparseSpMatDescr_t A_csr;
    CHECK_CUSPARSE(cusparseCreateCsr(
        &A_csr, (int64_t)H.rows, (int64_t)H.cols, (int64_t)H.nnz, 
        (void*)dA.d_rowptr, (void*)dA.d_colind, (void*)dA.d_vals,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    double time_csr = runSpMV_once(handle, A_csr, H.rows, H.cols, dA.d_x, dy, false);
    CHECK_CUSPARSE(cusparseDestroySpMat(A_csr));

    // CSR -> COO rows
    int *A_coo_rows = nullptr;
    CHECK_CUDA(cudaMalloc(&A_coo_rows, (size_t)H.nnz * sizeof(int)));
    CHECK_CUSPARSE(cusparseXcsr2coo(handle, dA.d_rowptr, H.nnz, H.rows, A_coo_rows, CUSPARSE_INDEX_BASE_ZERO));

    // COO descriptor
    cusparseSpMatDescr_t A_coo;
    CHECK_CUSPARSE(cusparseCreateCoo(&A_coo, (int64_t)H.rows, (int64_t)H.cols, (int64_t)H.nnz,
        A_coo_rows, (void*)dA.d_colind, (void*)dA.d_vals,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    double time_coo = runSpMV_once(handle, A_coo, H.rows, H.cols, dA.d_x, dy_2, true);
    CHECK_CUSPARSE(cusparseDestroySpMat(A_coo));
    CHECK_CUDA(cudaFree(A_coo_rows));

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hy.data(), dy, (size_t)H.rows*sizeof(T), cudaMemcpyDeviceToHost));

    // CPU Eigen
    Eigen::setNbThreads(1);
    Eigen::SparseMatrix<T, Eigen::ColMajor, int> A = load_eigen_from_mtx<T>(path);
    A.makeCompressed();
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> x_cpu(hx.data(), A.cols());
    auto t0 = std::chrono::high_resolution_clock::now();
    Eigen::Matrix<T, Eigen::Dynamic, 1> y_cpu = A * x_cpu;
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_cpu = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Limpieza
    CHECK_CUDA(cudaFree(dy));
    CHECK_CUDA(cudaFree(dy_2));
    destroy_device(dA);
    CHECK_CUSPARSE(cusparseDestroy(handle));
    append_row_csv("resultados.csv", path, H.rows, H.cols, H.nnz, time_csr, time_coo, ms_cpu);
    return 0;
}
