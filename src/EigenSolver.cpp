#include "helpers/mtxToCuda.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <string>
#include <fstream>
#include <cstdio>

// Función para escribir resultados en un archivo CSV
void create_csv(const char* filename, const char* path, int rows, int cols, long long nnz, double ms, int nthreads) {
    bool write_header = true;
    {
        std::ifstream test(csv_path, std::ios::binary);
        if (test.good()) { test.seekg(0, std::ios::end); write_header = (test.tellg() == 0); }
    }
    std::ofstream f(csv_path, std::ios::app);
    if (!f) { fprintf(stderr, "No pude abrir %s\n", csv_path.c_str()); return; }
    if (write_header) {
        out << "file,rows,cols,nnz,ms,nthreads\n";
    }
    out << '"' << path << '"' << ',' << rows << ',' << cols << ',' << nnz << ',' << ms << ',' << nthreads << '\n';
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <ruta_al_archivo.mtx> [numero_de_hilos]" << std::endl;
        return 1;
    }
    const char* path = argv[1];
    int num_threads = (argc > 2) ? std::atoi(argv[2]) : 0; // 0 para usar el máximo de hilos

    Eigen::initParallel();
    Eigen::setNbThreads(num_threads);

    using T = float;
    
    try {
        Eigen::SparseMatrix<T, Eigen::RowMajor, int> A = load_eigen_from_mtx<T>(path);
        A.makeCompressed();

        std::cout << "Matriz: " << path << std::endl;
        std::cout << "Dimensiones: " << A.rows() << " x " << A.cols() << ", NNZ: " << A.nonZeros() << std::endl;
        std::cout << "Usando " << Eigen::nbThreads() << " hilos para el cálculo." << std::endl;

        // Crear un vector aleatorio para la multiplicación
        std::vector<T> hx(A.cols());
        std::srand(42);
        for (size_t i = 0; i < hx.size(); ++i) {
            hx[i] = (T)std::rand() / RAND_MAX;
        }

        Eigen::Map<const Eigen::Matrix<T, -1, 1>> x_cpu(hx.data(), A.cols());
        
        // Realizar la multiplicación y medir el tiempo
        auto t0 = std::chrono::high_resolution_clock::now();
        Eigen::Matrix<T, -1, 1> y_cpu = A * x_cpu;
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms_cpu = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout << "SpMV completado en " << ms_cpu << " ms." << std::endl;

        // Guardar el resultado en un CSV
        create_csv("eigen_benchmark.csv", path, A.rows(), A.cols(), A.nonZeros(), ms_cpu, Eigen::nbThreads());

    } catch (const std::runtime_error& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}