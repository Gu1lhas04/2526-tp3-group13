
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


// Kernel 3: Maximum Intensity Projection — single image
// Each thread handles one output pixel.
__global__ void rotated_mip_kernel(const uint16_t* volume, uint16_t* image, int N, int M, const float* R, int ray_range) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= N || y >= N) return;

    uint16_t maxIntensity = 0;

    // 1. Center the 2D screen coordinates
    float u = x - N / 2.0f;
    float v = y - N / 2.0f;

    // 2. Step the ray through the volume
    for (int step = -ray_range; step < ray_range; step++) {
        float w = (float)step;

        // 3. Transform point (u, v, w) to volume space
        float rotX = R[0] * u + R[1] * v + R[2] * w + N / 2.0f;
        float rotY = R[3] * u + R[4] * v + R[5] * w + N / 2.0f;
        float rotZ = R[6] * u + R[7] * v + R[8] * w + M / 2.0f;

        // 4. Boundary Check
        if (rotX >= 0.0f && rotX < (float)N &&
            rotY >= 0.0f && rotY < (float)N &&
            rotZ >= 0.0f && rotZ < (float)M) {
            
            size_t idx = (size_t)((int)rotZ * N * N + (int)rotY * N + (int)rotX);
            uint16_t val = volume[idx];
            if (val > maxIntensity) {
                maxIntensity = val;
            }
        }
    }
    image[y * N + x] = maxIntensity;
}


// Kernel 4: Multi-frame MIP — generates num_frames images in one kernel call.
__global__ void multi_mip_kernel(const uint16_t* volume, uint16_t* images, int N, int M, const float* R_all, int num_frames, int ray_range) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int frame = blockIdx.z;

    if (x >= N || y >= N || frame >= num_frames) return;

    const float* R = R_all + frame * 9;

    uint16_t maxIntensity = 0;
    float u = x - N / 2.0f;
    float v = y - N / 2.0f;

    for (int step = -ray_range; step < ray_range; step++) {
        float w = (float)step;
        float rotX = R[0]*u + R[1]*v + R[2]*w + N / 2.0f;
        float rotY = R[3]*u + R[4]*v + R[5]*w + N / 2.0f;
        float rotZ = R[6]*u + R[7]*v + R[8]*w + M / 2.0f;

        if (rotX >= 0.0f && rotX < (float)N && rotY >= 0.0f && rotY < (float)N && rotZ >= 0.0f && rotZ < (float)M) {
            size_t idx = (size_t)((int)rotZ * N * N + (int)rotY * N + (int)rotX);
            uint16_t val = volume[idx];
            if (val > maxIntensity) maxIntensity = val;
        }
    }

    images[(size_t)frame * N * N + y * N + x] = maxIntensity;
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
    dim3 gridDim3D((N + blockDim3D.x - 1) / blockDim3D.x, (N + blockDim3D.y - 1) / blockDim3D.y, (M + blockDim3D.z - 1) / blockDim3D.z);
    gaussian_blur_kernel<<<gridDim3D, blockDim3D>>>(d_volume, d_blurred, N, M, d_kernel);
    cudaDeviceSynchronize();

    // Step 3: MIP
    print("  gpu: generating MIP\n");
    uint16_t* d_raster;
    cudaMalloc(&d_raster, N * N * sizeof(uint16_t));

    int ray_range = (int)(sqrtf(N*N + N*N + M*M) / 2.0f) + 1;
    dim3 blockDim2D(16, 16);
    dim3 gridDim2D((N + blockDim2D.x - 1) / blockDim2D.x, (N + blockDim2D.y - 1) / blockDim2D.y);

    rotated_mip_kernel<<<gridDim2D, blockDim2D>>>(d_blurred, d_raster, N, M, d_R, ray_range);
    cudaDeviceSynchronize();

    // Transfer output raster back to CPU
    cudaMemcpy(output, d_raster, N * N * sizeof(uint16_t), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_volume);
    cudaFree(d_blurred);
    cudaFree(d_kernel);
    cudaFree(d_R);
    cudaFree(d_raster);
}


// Generates num_frames MIP images in one kernel call.
void device_bunny_generate_video(const uint16_t* input, uint16_t threshold, float sigma, const float* R_all, int num_frames, uint16_t* output)
{
    print("Generating %d frames on the GPU\n", num_frames);

    const int N = kBunnySize;
    const int M = kBunnyN;
    const size_t volume_bytes = (size_t)N * N * M * sizeof(uint16_t);
    const size_t frames_bytes = (size_t)num_frames * N * N * sizeof(uint16_t);

    // Build Gaussian kernel on host
    float h_kernel[27];
    float kernel_sum = 0.0f;
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                float dist_sq = (float)(dx*dx + dy*dy + dz*dz);
                float w = expf(-dist_sq / (2.0f * sigma * sigma));
                h_kernel[(dz+1)*9 + (dy+1)*3 + (dx+1)] = w;
                kernel_sum += w;
            }
    for (int i = 0; i < 27; i++) h_kernel[i] /= kernel_sum;

    // Allocate device memory
    uint16_t *d_volume, *d_blurred, *d_frames;
    float *d_kernel, *d_R_all;

    cudaMalloc(&d_volume, volume_bytes);
    cudaMalloc(&d_blurred, volume_bytes);
    cudaMalloc(&d_frames, frames_bytes);
    cudaMalloc(&d_kernel, 27 * sizeof(float));
    cudaMalloc(&d_R_all, num_frames * 9 * sizeof(float));

    cudaMemcpy(d_volume, input, volume_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, 27 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_R_all, R_all, num_frames * 9 * sizeof(float), cudaMemcpyHostToDevice);

    // Step 1: Threshold
    int total_voxels = N * N * M;
    int threads1D = 256;
    int blocks1D = (total_voxels + threads1D - 1) / threads1D;
    threshold_kernel<<<blocks1D, threads1D>>>(d_volume, total_voxels, threshold);
    cudaDeviceSynchronize();

    // Step 2: Gaussian Blur
    dim3 blockDim3D(8, 8, 8);
    dim3 gridDim3D((N + blockDim3D.x - 1) / blockDim3D.x, (N + blockDim3D.y - 1) / blockDim3D.y, (M + blockDim3D.z - 1) / blockDim3D.z);
    gaussian_blur_kernel<<<gridDim3D, blockDim3D>>>(d_volume, d_blurred, N, M, d_kernel);
    cudaDeviceSynchronize();

    // Step 3: Multi-frame MIP — all frames in one kernel call
    int ray_range = (int)(sqrtf(N*N + N*N + M*M) / 2.0f) + 1;
    dim3 blockDim2D(16, 16, 1);
    dim3 gridDim2D((N + blockDim2D.x - 1) / blockDim2D.x, (N + blockDim2D.y - 1) / blockDim2D.y, num_frames );
    multi_mip_kernel<<<gridDim2D, blockDim2D>>>(d_blurred, d_frames, N, M, d_R_all, num_frames, ray_range);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_frames, frames_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_volume);
    cudaFree(d_blurred);
    cudaFree(d_frames);
    cudaFree(d_kernel);
    cudaFree(d_R_all);
}

