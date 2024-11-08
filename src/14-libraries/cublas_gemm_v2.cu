#include <stdio.h>
#include"error.cuh"
#include<cublas_v2.h>

void print_matrix(int R, int C, double* A, const char* name) {
    printf("%s:\n", name);
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            printf("%10.6f", A[i * C + j]);
        }
        printf("\n");
    }
}


int main() {
    int M = 2, K = 3, N = 4;

    int MK = M * K;
    int KN = K * N;
    int MN = M * N;

    double* h_A = (double*)malloc(MK * sizeof(double));
    double* h_B = (double*)malloc(KN * sizeof(double));
    double* h_C = (double*)malloc(MN * sizeof(double));

    for (int i = 0; i < MK; i++) {
        h_A[i] = i;
    }
    print_matrix(M, K, h_A, "A");

    for (int i = 0; i < KN; i++) {
        h_B[i] = i;
    }
    print_matrix(K, N, h_B, "B");

    for (int i = 0; i < MN; i++) {
        h_C[i] = 0;
    }

    double* g_A, * g_B, * g_C;
    CHECK(cudaMalloc((void**)&g_A, sizeof(double) * MK));
    CHECK(cudaMalloc((void**)&g_B, sizeof(double) * KN));
    CHECK(cudaMalloc((void**)&g_C, sizeof(double) * MN));

    cublasSetVector(MK, sizeof(double), h_A, 1, g_A, 1);
    cublasSetVector(KN, sizeof(double), h_B, 1, g_B, 1);
    cublasSetVector(MN, sizeof(double), h_C, 1, g_C, 1);

    cublasHandle_t handle;
    cublasCreate(&handle);

    double alpha = 1.0, beta = 0.0;
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, g_A, M, g_B, K, &beta, g_C, M);
    cublasDestroy(handle);

    cublasGetVector(MN, sizeof(double), g_C, 1, h_C, 1);
    print_matrix(M, N, h_C, "C");

    free(h_A);
    free(h_B);
    free(h_C);

    CHECK(cudaFree(g_A));
    CHECK(cudaFree(g_B));
    CHECK(cudaFree(g_C));

    return 0;


}


