#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <chrono>

using T = double;
using SpMat = Eigen::SparseMatrix<T>; 

struct trsm_system {
    int size{};
    int nrhs{};
    int nnz{};
    SpMat            A;  
    Eigen::MatrixXd  B;
};

trsm_system load_system(const char* file)
{
    FILE* f = fopen(file, "r");
    if (!f) throw std::runtime_error("cannot open matrix file");

    trsm_system sys;
    if (fscanf(f, "%d%d%d", &sys.size, &sys.nrhs, &sys.nnz) != 3) {
        fclose(f);
        throw std::runtime_error("bad header");
    }

    std::vector<int>    rowptr(sys.size + 1);
    std::vector<int>    colidx(sys.nnz);
    std::vector<double> vals(sys.nnz);

    for (int r = 0; r <= sys.size; ++r) if (fscanf(f, "%d", &rowptr[r]) != 1) {
        fclose(f); throw std::runtime_error("bad rowptr");
    }
    for (int i = 0; i < sys.nnz; ++i)    if (fscanf(f, "%d", &colidx[i]) != 1) {
        fclose(f); throw std::runtime_error("bad colidx");
    }
    for (int i = 0; i < sys.nnz; ++i)    if (fscanf(f, "%lf", &vals[i]) != 1) {
        fclose(f); throw std::runtime_error("bad vals");
    }
    fclose(f);

    std::vector<Eigen::Triplet<T>> trips;
    trips.reserve(sys.nnz);
    for (int i = 0; i < sys.size; ++i) {
        for (int p = rowptr[i]; p < rowptr[i + 1]; ++p) {
            trips.emplace_back(i, colidx[p], vals[p]);
        }
    }

    sys.A.resize(sys.size, sys.size);
    sys.A.setFromTriplets(trips.begin(), trips.end());
    sys.A.makeCompressed();

    sys.B.resize(sys.size, sys.nrhs);
    srand(42);
    for (int i = 0; i < sys.size; ++i)
        for (int j = 0; j < sys.nrhs; ++j)
            sys.B(i, j) = static_cast<double>(rand()) / RAND_MAX;

    return sys;
}

static double benchmark_sparse_triangular(const SpMat& A, const Eigen::MatrixXd& B,
                                          int warmup = 2, int repeats = 10)
{
    double total_ms = 0.0;
    Eigen::MatrixXd X(B.rows(), B.cols());
    for (int rep = 0; rep < warmup + repeats; ++rep) {
        auto t0 = std::chrono::high_resolution_clock::now();
        X.noalias() = A.template triangularView<Eigen::Lower>().solve(B);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> dt = t1 - t0;
        if (rep >= warmup) total_ms += dt.count();
    }
    return total_ms / repeats;
}

static double benchmark_dense_triangular(const SpMat& A, const Eigen::MatrixXd& B,
                                         int warmup = 2, int repeats = 10)
{
    // Convertir una vez a densa fuera del bucle de timing
    Eigen::MatrixXd Ad = Eigen::MatrixXd(A); // usa solo la parte inferior en el solve
    double total_ms = 0.0;
    Eigen::MatrixXd X(B.rows(), B.cols());
    for (int rep = 0; rep < warmup + repeats; ++rep) {
        auto t0 = std::chrono::high_resolution_clock::now();
        X.noalias() = Ad.template triangularView<Eigen::Lower>().solve(B);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> dt = t1 - t0;
        if (rep >= warmup) total_ms += dt.count();
    }
    return total_ms / repeats;
}

void trsm_generic(trsm_system& sys)
{
    // Sparse (CSR interno de Eigen)
    double ms_sparse = benchmark_sparse_triangular(sys.A, sys.B);

    // Dense (conversión a MatrixXd y solve triangular)
    double ms_dense  = benchmark_dense_triangular(sys.A, sys.B);

    std::printf("SparseTriSolve,%dx%d,%.6f ms\n", sys.size, sys.size, ms_sparse);
    std::printf("DenseTriSolve,%dx%d,%.6f ms\n",  sys.size, sys.size, ms_dense);
}

int main()
{
    std::vector<const char*> matrix_files = {"matrix13.txt", "matrix16.txt", "matrix20.txt"};
    for (const char* file : matrix_files) {
        try {
            trsm_system sys = load_system(file);
            trsm_generic(sys);
        } catch (const std::exception& e) {
            std::cerr << "Error en " << file << ": " << e.what() << "\n";
        }
    }
    return 0;
}
