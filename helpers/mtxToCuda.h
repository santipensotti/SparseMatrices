#ifndef MTX_TO_CUDA_H
#define MTX_TO_CUDA_H

#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <tuple>
#include <Eigen/Sparse>
#include <unsupported/Eigen/SparseExtra>

template<typename T>
using SpMat = Eigen::SparseMatrix<T, Eigen::RowMajor, int>;

// ====== Contenedores host ======
template<typename T>
struct CSRHost {
    int rows = 0, cols = 0, nnz = 0;
    std::vector<int> rowptr;  
    std::vector<int> colind;  
    std::vector<T>   vals;    
};

template<typename T>
struct COOHost {
    int rows = 0, cols = 0, nnz = 0;
    std::vector<int> row;   
    std::vector<int> col;   
    std::vector<T>   val;   
};
// ---------- Utilidades internas ----------
namespace detail {

template<typename T>
inline CSRHost<T> coo_to_csr(const COOHost<T>& C) {
    CSRHost<T> R;
    R.rows = C.rows; R.cols = C.cols; R.nnz = C.nnz;
    R.rowptr.assign(R.rows + 1, 0);
    R.colind.resize(R.nnz);
    R.vals.resize(R.nnz);

    // contar por filas
    for (int i = 0; i < C.nnz; ++i) {
        ++R.rowptr[C.row[i] + 1];
    }
    // prefijo
    for (int r = 0; r < R.rows; ++r) R.rowptr[r + 1] += R.rowptr[r];

    // fill
    std::vector<int> write_ptr = R.rowptr;
    for (int i = 0; i < C.nnz; ++i) {
        int r = C.row[i];
        int p = write_ptr[r]++;
        R.colind[p] = C.col[i];
        R.vals  [p] = C.val[i];
    }

    // asegurar orden por columna dentro de cada fila
    for (int r = 0; r < R.rows; ++r) {
        int start = R.rowptr[r], end = R.rowptr[r+1];
        auto first_col = R.colind.begin() + start, last_col = R.colind.begin() + end;
        auto first_val = R.vals.begin()   + start;
        // ordenar pares (col, val) por col
        std::vector<std::pair<int,T>> tmp(end - start);
        for (int k = 0; k < end - start; ++k)
            tmp[k] = { R.colind[start + k], R.vals[start + k] };
        std::sort(tmp.begin(), tmp.end(), [](auto& a, auto& b){ return a.first < b.first; });
        for (int k = 0; k < end - start; ++k) {
            R.colind[start + k] = tmp[k].first;
            R.vals  [start + k] = tmp[k].second;
        }
    }

    return R;
}

}


template<typename T>
SpMat<T> load_eigen_from_mtx(const std::string& filename) {
    SpMat<T> A;
    if (!Eigen::loadMarket(A, filename)) {
        throw std::runtime_error("No pude leer el .mtx: " + filename);
    }
    return A;
}

template<typename T>
COOHost<T> load_coo_from_mtx(const std::string& filename) {
    SpMat<T> A;
    if (!Eigen::loadMarket(A, filename)) {
        throw std::runtime_error("No pude leer el .mtx: " + filename);
    }
    A.makeCompressed(); 

    std::vector<Eigen::Triplet<T, int>> trips;
    trips.reserve((size_t)A.nonZeros());
    for (int k = 0; k < A.outerSize(); ++k) {
        for (typename SpMat<T>::InnerIterator it(A, k); it; ++it) {
            trips.emplace_back(it.row(), it.col(), it.value());
        }
    }

    COOHost<T> H;
    H.rows = A.rows();
    H.cols = A.cols();
    H.nnz  = (int)trips.size();
    H.row.resize(H.nnz);
    H.col.resize(H.nnz);
    H.val.resize(H.nnz);

    for (int i = 0; i < H.nnz; ++i) {
        H.row[i] = trips[i].row();
        H.col[i] = trips[i].col();
        H.val[i] = trips[i].value();
    }

    return H;
}

// Carga CSR directamente desde .mtx (vía COO normalizado -> CSR)
template<typename T>
CSRHost<T> load_csr_from_mtx(const std::string& filename) {
    COOHost<T> C = load_coo_from_mtx<T>(filename);
    return detail::coo_to_csr(C);
}

// Extraer COO desde Eigen::SparseMatrix (por si ya la tenés en memoria)
template<typename T>
COOHost<T> eigen_to_coo(const Eigen::SparseMatrix<T, Eigen::ColMajor, int>& A) {
    Eigen::SparseMatrix<T, Eigen::ColMajor, int> Ac = A;
    Ac.makeCompressed();

    COOHost<T> H;
    H.rows = Ac.rows(); H.cols = Ac.cols();
    H.nnz  = (int)Ac.nonZeros();
    H.row.reserve(H.nnz); H.col.reserve(H.nnz); H.val.reserve(H.nnz);

    for (int k = 0; k < Ac.outerSize(); ++k) {
        for (typename Eigen::SparseMatrix<T, Eigen::ColMajor, int>::InnerIterator it(Ac, k); it; ++it) {
            H.row.push_back(it.row());
            H.col.push_back(it.col());
            H.val.push_back(it.value());
        }
    }
    H.nnz = (int)H.val.size();
    return H;
}

#endif // MTX_TO_CUDA_H
