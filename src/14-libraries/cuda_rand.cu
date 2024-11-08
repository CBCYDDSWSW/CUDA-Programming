#include<stdio.h>
#include<stdlib.h>
#include<curand.h>

void output_result(double* y) {
    FILE* fid = fopen("x1.txt", "w");
    for (int i = 0; i < 100000; i++) {
        fprintf(fid, "%lf\n", y[i]);
    }
    fclose(fid);
}

int main() {
    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, 1234);
    int N = 100000;
    double* x;
    cudaMalloc((void**)&x, N * sizeof(double));
    curandGenerateUniformDouble(generator, x, N);

    double* y = (double*)malloc(N *sizeof(double));
    cudaMemcpy(y, x, sizeof(double) * N, cudaMemcpyDeviceToHost);

    output_result(y);
    cudaFree(x);
    free(y);
    curandDestroyGenerator(generator);
    return 0;
}