#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <cstdio>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <chrono>


typedef Eigen::SparseMatrix<double> SpMat;
struct trsm_system
{
    int size;
    int nrhs;
    int nnz;
    int ld;
    Eigen::SparseMatrix<double, Eigen::RowMajor> eigen_matrix;
    Eigen::MatrixXd B ;
};

trsm_system load_system(const char * file)
{
    FILE * f = fopen(file, "r");
    if(f == nullptr) throw std::runtime_error("cannot open matrix file");

    trsm_system sys;
    fscanf(f, "%d%d%d", &sys.size, &sys.nrhs, &sys.nnz);
    int size = sys.size, nrhs = sys.nrhs, nnz = sys.nnz;
    std::vector<int> rowptrs(sys.size+1);
    std::vector<int> colidxs(sys.nnz);
    std::vector<double> vals(sys.nnz);
    for(int r = 0; r <= sys.size; r++) fscanf(f, "%d", &rowptrs[r]);
    for(int i = 0; i < sys.nnz; i++) fscanf(f, "%d", &colidxs[i]);
    for(int i = 0; i < sys.nnz; i++) fscanf(f, "%lf", &vals[i]);
    Eigen::SparseMatrix<double, Eigen::RowMajor> eigen_matrix(size, size);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(nnz);

    for (int i = 0; i < size; ++i) {
        for (int j = rowptrs[i]; j < rowptrs[i+1]; ++j) {
            triplets.emplace_back(i, colidxs[j], vals[j]);
        }
    }

    eigen_matrix.setFromTriplets(triplets.begin(), triplets.end());
    eigen_matrix.makeCompressed();

    fclose(f);

    Eigen::MatrixXd B(size, nrhs);
    srand(42);
    for (int i=0; i < sys.size; ++i){
        for (int j=0; j < sys.nrhs; ++j){
            B(i, j) = (double)rand() / RAND_MAX;
        }
    }
    sys.B  = B;
    sys.eigen_matrix = eigen_matrix;
    return sys;
}

void trsm_generic(trsm_system & sys)
{
    auto start = std::chrono::high_resolution_clock::now();
    Eigen::SparseMatrix<double, Eigen::RowMajor>  A = sys.eigen_matrix;
    Eigen::MatrixXd X = A.template triangularView<Eigen::UnitLower>().solve(sys.B);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("%dx%d,%12ld ms\n", sys.size, sys.size, duration);
}

int main(){
    std::vector<const char*> matrix_files = {"matrix13.txt"};
    for(const char* file : matrix_files){
        trsm_system sys = load_system(file);
        trsm_generic(sys);
    }

    return 0;
}
