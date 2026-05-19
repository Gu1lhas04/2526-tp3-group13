
#include <cmath>
#include "print.h"
#include "bunnyIO.h"
#include "device-bunnyMIP.h"

// Kernel 1: Thresholding (Each thread processes one voxel independently)
__global__ void threshold_kernel(uint16_t* volume, int size, uint16_t threshold) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    if (volume[i] < threshold)
        volume[i] = 0;
}


// Kernel 2: 3D Gaussian Blur (Each thread computes one output voxel from its 3x3x3 neighborhood. Boundary voxels are clamped (replicate padding))
__global__ void gaussian_blur_kernel(const uint16_t* input, uint16_t* output, int N, int M, const float* kernel) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= N || y >= N || z >= M) return;

    float value = 0.0f;

    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nz = max(0, min(M - 1, z + dz));
                int ny = max(0, min(N - 1, y + dy));
                int nx = max(0, min(N - 1, x + dx));

                uint16_t voxel = input[(size_t)nz * N * N + ny * N + nx];
                int kidx = (dz + 1) * 9 + (dy + 1) * 3 + (dx + 1);
                value += (float)voxel * kernel[kidx];
            }
        }
    }

    output[(size_t)z * N * N + y * N + x] = (uint16_t)value;
}


// Device entry point
void device_bunny_mip(const uint16_t* input, uint16_t threshold, float sigma, const float* R, uint16_t* output)
{
    print("Running functions on the GPU\n");

    const int N = kBunnySize;
    const int M = kBunnyN;
    const size_t volume_bytes = (size_t)N * N * M * sizeof(uint16_t);

    // Build Gaussian kernel on host
    float h_kernel[27];
    float kernel_sum = 0.0f;
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                float dist_sq = (float)(dx*dx + dy*dy + dz*dz);
                float w = expf(-dist_sq / (2.0f * sigma * sigma));
                h_kernel[(dz+1)*9 + (dy+1)*3 + (dx+1)] = w;
                kernel_sum += w;
            }
        }
    }
    for (int i = 0; i < 27; i++) h_kernel[i] /= kernel_sum;

    // Allocate device memory
    uint16_t *d_volume, *d_blurred;
    float *d_kernel, *d_R;

    cudaMalloc(&d_volume, volume_bytes);
    cudaMalloc(&d_blurred, volume_bytes);
    cudaMalloc(&d_kernel, 27 * sizeof(float));
    cudaMalloc(&d_R, 9 * sizeof(float));

    // Transfer to device
    cudaMemcpy(d_volume, input, volume_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, 27 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_R, R, 9 * sizeof(float), cudaMemcpyHostToDevice);

    // Step 1: Threshold
    print("  gpu: applying threshold\n");
    int total_voxels = N * N * M;
    int threads1D = 256;
    int blocks1D = (total_voxels + threads1D - 1) / threads1D;
    threshold_kernel<<<blocks1D, threads1D>>>(d_volume, total_voxels, threshold);
    cudaDeviceSynchronize();

    // Step 2: Gaussian Blur
    print("  gpu: applying filter\n");
    dim3 blockDim3D(8, 8, 8);
    dim3 gridDim3D(
        (N + blockDim3D.x - 1) / blockDim3D.x,
        (N + blockDim3D.y - 1) / blockDim3D.y,
        (M + blockDim3D.z - 1) / blockDim3D.z
    );
    gaussian_blur_kernel<<<gridDim3D, blockDim3D>>>(d_volume, d_blurred, N, M, d_kernel);
    cudaDeviceSynchronize();

    // Step 3: MIP (to be implemented by Person B)
    print("  gpu: generating MIP\n");

    // Cleanup
    cudaFree(d_volume);
    cudaFree(d_blurred);
    cudaFree(d_kernel);
    cudaFree(d_R);
}
