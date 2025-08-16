#ifndef MTX_TO_CUDA_H
#define MTX_TO_CUDA_H

#include <vector>
#include <string>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/Sparse>
#include <cstdio>

// Carga matriz en formato CSR desde archivo .mtx
template<typename T>
bool load_csr(const std::string& filename,
              Eigen::SparseMatrix<T, Eigen::RowMajor>& A,
              const int*& rowptr, const int*& colind, const T*& vals) {
    if (!Eigen::loadMarket(A, filename)) {
        fprintf(stderr, "No pude leer el .mtx\n");
        return false;
    }
    A.makeCompressed();
    rowptr = A.outerIndexPtr();
    colind = A.innerIndexPtr();
    vals   = A.valuePtr();
    return true;
}

// Extrae COO de Eigen::SparseMatrix
template<typename T>
void get_coo(const Eigen::SparseMatrix<T>& A,
             std::vector<int>& rows, std::vector<int>& cols, std::vector<T>& vals) {
    rows.clear(); cols.clear(); vals.clear();
    for (int k = 0; k < A.outerSize(); ++k) {
        for (typename Eigen::SparseMatrix<T>::InnerIterator it(A, k); it; ++it) {
            rows.push_back(it.row());
            cols.push_back(it.col());
            vals.push_back(it.value());
        }
    }
}

// Extrae CSR de Eigen::SparseMatrix
template<typename T>
void get_csr(const Eigen::SparseMatrix<T, Eigen::RowMajor>& A,
             const int*& rowptr, const int*& colind, const T*& vals) {
    rowptr = A.outerIndexPtr();
    colind = A.innerIndexPtr();
    vals   = A.valuePtr();
}

#endif // MTX_TO_CUDA_H
