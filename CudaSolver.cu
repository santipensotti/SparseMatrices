#include <cstdio>
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>
#include <cusparse.h>


#define CHECK(status) do { _check((status), __FILE__, __LINE__); } while(false)
inline void _check(cudaError_t error_code, const char *file, int line)
{
    if (error_code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error %d %s: %s. In file '%s' on line %d\n", error_code, cudaGetErrorName(error_code), cudaGetErrorString(error_code), file, line);
        fflush(stderr);
        exit(1);
    }
}
inline void _check(cusparseStatus_t status, const char *file, int line)
{
    if (status != CUSPARSE_STATUS_SUCCESS) {
        fprintf(stderr, "CUSPARSE Error %d %s: %s. In file '%s' on line %d\n", status, cusparseGetErrorName(status), cusparseGetErrorString(status), file, line);
        fflush(stderr);
        exit(1);
    }
}



struct trsm_system
{
    int size;
    int nrhs;
    int nnz;
    int ld;
    int * A_rowptrs = nullptr;
    int * A_colidxs = nullptr;
    double * A_vals = nullptr;
    double * B = nullptr; // row-major
    double * X = nullptr; // row-major
};

trsm_system load_system(const char * file)
{
    FILE * f = fopen(file, "r");
    if(f == nullptr) throw std::runtime_error("cannot open matrix file");

    trsm_system sys;
    fscanf(f, "%d%d%d", &sys.size, &sys.nrhs, &sys.nnz);
    std::vector<int> rowptrs(sys.size+1);
    std::vector<int> colidxs(sys.nnz);
    std::vector<double> vals(sys.nnz);
    for(int r = 0; r <= sys.size; r++) fscanf(f, "%d", &rowptrs[r]);
    for(int i = 0; i < sys.nnz; i++) fscanf(f, "%d", &colidxs[i]);
    for(int i = 0; i < sys.nnz; i++) fscanf(f, "%lf", &vals[i]);

    fclose(f);

    std::vector<double> B(sys.size * sys.nrhs);
    srand(42); // Set seed for reproducibility
    for(int i = 0; i < B.size(); i++) B[i] = (double)rand() / RAND_MAX;

    CHECK(cudaMalloc(&sys.A_rowptrs, (sys.size + 1) * sizeof(int)));
    CHECK(cudaMalloc(&sys.A_colidxs, sys.nnz * sizeof(int)));
    CHECK(cudaMalloc(&sys.A_vals, sys.nnz * sizeof(double)));
    size_t pitch_B;
    CHECK(cudaMallocPitch(&sys.B, &pitch_B, sys.nrhs * sizeof(double), sys.size));
    sys.ld = pitch_B / sizeof(double);
    size_t pitch_X;
    CHECK(cudaMallocPitch(&sys.X, &pitch_X, sys.nrhs * sizeof(double), sys.size));
    if(pitch_X != pitch_B) throw std::runtime_error("different pitches");

    CHECK(cudaMemcpy(sys.A_rowptrs, rowptrs.data(), (sys.size + 1) * sizeof(int), cudaMemcpyDefault));
    CHECK(cudaMemcpy(sys.A_colidxs, colidxs.data(), sys.nnz * sizeof(int), cudaMemcpyDefault));
    CHECK(cudaMemcpy(sys.A_vals, vals.data(), sys.nnz * sizeof(double), cudaMemcpyDefault));
    CHECK(cudaMemcpy2D(sys.B, pitch_B, B.data(), sys.nrhs * sizeof(double), sys.nrhs * sizeof(double), sys.size, cudaMemcpyDefault));

    return sys;
}


void run_spm(const trsm_system& sys, cusparseSpMatDescr_t& descr_A, const cusparseHandle_t& handle, bool is_coo = false){
    cusparseDnMatDescr_t descr_B, descr_X;
    CHECK(cusparseCreateDnMat(&descr_B, sys.size, sys.nrhs, sys.ld, sys.B, CUDA_R_64F, CUSPARSE_ORDER_ROW));
    CHECK(cusparseCreateDnMat(&descr_X, sys.size, sys.nrhs, sys.ld, sys.X, CUDA_R_64F, CUSPARSE_ORDER_ROW));

    cusparseSpSMDescr_t descr_spsm;
    CHECK(cusparseSpSM_createDescr(&descr_spsm));

    cudaEvent_t e_start, e_stop;
    CHECK(cudaEventCreate(&e_start));
    CHECK(cudaEventCreate(&e_stop));

    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    double one = 1.0;
    size_t buffersize;
    void * buffer;
    int warmup = 2;
    int repeats = 10;
    float ms_total = 0;

    CHECK(cusparseSpSM_bufferSize(handle, opA, opB, &one, descr_A, descr_B, descr_X, CUDA_R_64F, CUSPARSE_SPSM_ALG_DEFAULT, descr_spsm, &buffersize));
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMalloc(&buffer, buffersize));

    CHECK(cusparseSpSM_analysis(handle, opA, opB, &one, descr_A, descr_B, descr_X, CUDA_R_64F, CUSPARSE_SPSM_ALG_DEFAULT, descr_spsm, buffer));
    CHECK(cudaDeviceSynchronize());

    for(int rep = 0; rep < warmup + repeats; rep++)
    {
        CHECK(cudaEventRecord(e_start));
        CHECK(cusparseSpSM_solve(handle, opA, opB, &one, descr_A, descr_B, descr_X, CUDA_R_64F, CUSPARSE_SPSM_ALG_DEFAULT, descr_spsm));
        CHECK(cudaEventRecord(e_stop));
        CHECK(cudaDeviceSynchronize());
        float ms;
        CHECK(cudaEventElapsedTime(&ms, e_start, e_stop));

        if(rep > warmup) ms_total += ms;
    }

    CHECK(cudaFree(buffer));
    CHECK(cudaEventDestroy(e_start));
    CHECK(cudaEventDestroy(e_stop));
    CHECK(cusparseDestroyDnMat(descr_B));
    CHECK(cusparseDestroyDnMat(descr_X));
    float ms_avg = ms_total / repeats;
    printf("%s,%dx%d,%12.6f\n", is_coo ? "COO" : "CSR", sys.size, sys.size, ms_avg);
}

void trsm_generic(trsm_system & sys)
{
    cusparseHandle_t handle;
    CHECK(cusparseCreate(&handle));

    auto set_tri_attrs = [](cusparseSpMatDescr_t A){
        auto lower   = CUSPARSE_FILL_MODE_LOWER;
        auto nonunit = CUSPARSE_DIAG_TYPE_NON_UNIT;
        CHECK(cusparseSpMatSetAttribute(A, CUSPARSE_SPMAT_FILL_MODE, &lower,   sizeof(lower)));
        CHECK(cusparseSpMatSetAttribute(A, CUSPARSE_SPMAT_DIAG_TYPE, &nonunit, sizeof(nonunit)));
    };
    // CSR
    cusparseSpMatDescr_t A_Csr;
    CHECK(cusparseCreateCsr(&A_Csr, sys.size, sys.size, sys.nnz,
                                    sys.A_rowptrs, sys.A_colidxs, sys.A_vals,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    set_tri_attrs(A_Csr);

    // COO
    int *A_coo_rows = nullptr;
    CHECK(cudaMalloc(&A_coo_rows, sys.nnz * sizeof(int)));
    CHECK(cusparseXcsr2coo(handle, sys.A_rowptrs, sys.nnz, sys.size,
                           A_coo_rows, CUSPARSE_INDEX_BASE_ZERO));

    cusparseSpMatDescr_t A_coo;
    CHECK(cusparseCreateCoo(&A_coo, sys.size, sys.size, sys.nnz,
                            A_coo_rows, sys.A_colidxs, sys.A_vals,
                            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    set_tri_attrs(A_coo);

    // Ejecuto los dos formatos
    run_spm(sys, A_Csr, handle);
    run_spm(sys, A_coo, handle, true);

    CHECK(cusparseDestroySpMat(A_Csr));
    CHECK(cusparseDestroySpMat(A_coo));
    CHECK(cudaFree(A_coo_rows));
    CHECK(cusparseDestroy(handle));
}

void clear_system(trsm_system & sys)
{
    CHECK(cudaFree(sys.A_rowptrs));
    CHECK(cudaFree(sys.A_colidxs));
    CHECK(cudaFree(sys.A_vals));
    CHECK(cudaFree(sys.B));
    CHECK(cudaFree(sys.X));
}


int main(){
    std::vector<const char*> matrix_files = {"matrix13.txt", "matrix16.txt", "matrix20.txt", "matrix25.txt"};
    printf("type,size,time\n");
    for(const char* file : matrix_files){
        trsm_system sys = load_system(file);
        trsm_generic(sys);
        clear_system(sys);
    }

    return 0;
}