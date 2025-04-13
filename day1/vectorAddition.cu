#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(float *A, float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x+ threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    float milliseconds = 0;

    int N = 1024;
    const int size = N * sizeof(int);
    float *d_A; 
    float *d_B;
    float *d_C;
    

    float *h_A = new float[N];
    float *h_B = new float[N];
    float *h_C = new float[N];

    for (int i=0; i<N; i++) {
        h_A[i] = 1;
        h_B[i] = 1;
    };

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_A, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = N / threadsPerBlock;

    vectorAdd<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU Time: %.4f ms\n", milliseconds);


    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    for (int i =0; i < 5; i ++) {
        std::cout << "C[" << i <<"]"<<"is" << h_C[i] << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;

}