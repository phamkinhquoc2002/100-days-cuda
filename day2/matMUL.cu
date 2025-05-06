#include <iostream>
#include <cuda_runtime.h>

__global__ void matMul(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if(row < M && col < N) {
        float val = 0.0f;
        for (int k = 0; k < K; k++) {
            val += A[row*K + k] * B[k*N + col];
        }
        C[row * N + col] = val;
    }
}

int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    float milliseconds = 0;

    int M = 128;
    int N = 256;
    int K = 512;

    float *A, *B, *C;

    A = (float *)malloc(M*K*sizeof(float));
    B = (float *)malloc(K*N*sizeof(float));
    C = (float *)malloc(M*N*sizeof(float));

    for (int i=0; i < M; i ++) {
        for (int j=0; j < K; j ++) {
            A[i * K + j] = 1.0f;
        }
    }

    for (int i=0; i < K; i ++) {
        for (int j=0; j < N; j ++) {
            B[i * N + j] = 1.0f;
        }
    }

    float *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim(ceil(N/16.0f), ceil(M/16.0f));

    matMul<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU Time: %.4f ms\n", milliseconds);
    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_C);
    cudaFree(d_B);
}


