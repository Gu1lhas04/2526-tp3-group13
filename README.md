# CLE 2025/26 — Practical Assignment 3
## Volumetric MIP with CUDA

Group 13

---

## Overview

The Stanford Bunny CT Scan is a 3D volume of 512×512×361 voxels. This program takes that raw data and outputs a 2D image using Maximum Intensity Projection (MIP). Three stages run on the GPU: thresholding to remove air noise, 3D Gaussian blur to smooth the volume, and MIP projection to flatten it into an image.

---

## Requirements

- NVIDIA GPU with CUDA support
- CUDA Toolkit + `nvcc`
- `make`

---

## Compilation

```bash
make
```

Full rebuild from scratch:

```bash
make clean && make
```

---

## Usage

```bash
./bunnyMIP [OPTIONS]
```

| Parameter | Description | Default |
|---|---|---|
| `--threshold <value>` | Voxels below this intensity are zeroed out | 32768 |
| `--sigma <value>` | Standard deviation for the Gaussian blur | 1.0 |
| `--roll <degrees>` | Rotation around Z axis | 0.0 |
| `--pitch <degrees>` | Rotation around X axis | 0.0 |
| `--yaw <degrees>` | Rotation around Y axis | 0.0 |
| `--frames <N>` | Generate N MIP frames rotating 360° in yaw and encode a video | 0 (disabled) |

```bash
# defaults
./bunnyMIP

# custom threshold and blur
./bunnyMIP --threshold 10000 --sigma 1.5

# rotated view
./bunnyMIP --roll 45 --pitch 30 --yaw 90

# generate a 36-frame rotating video (one kernel call for all frames)
./bunnyMIP --threshold 10000 --frames 36
```

Output goes to `output/`:
- `bunnyMIP_cpu.pgm` — CPU reference image
- `bunnyMIP_gpu.pgm` — GPU result
- `frame_000.pgm` … `frame_NNN.pgm` — individual video frames (when `--frames` is used)
- `bunny.mp4` — encoded video (requires ffmpeg)

---

## CUDA kernels

### `threshold_kernel`

Zeroes out any voxel below the threshold. One thread per voxel, ~95 million total, 1D grid with 256 threads per block. The operation runs in-place on the device volume.

### `gaussian_blur_kernel`

Each thread reads the 3×3×3 neighborhood of its voxel and computes a weighted average. The 27 Gaussian weights are computed on the host from `sigma` and copied to the GPU once before the kernel launches. Voxels at the volume boundary clamp to the nearest edge. 3D grid, 8×8×8 threads per block.

### `rotated_mip_kernel`

Each thread handles one output pixel. It casts a ray through the volume and keeps the highest intensity it finds. The rotation matrix R (built from roll/pitch/yaw) maps screen coordinates into volume space for the ray traversal. Ray length comes from the volume diagonal so nothing gets clipped at any rotation angle. 2D grid, 16×16 threads per block.

### `multi_mip_kernel` (innovation)

Extends `rotated_mip_kernel` to generate N frames in a single kernel call. The grid adds a third dimension: `blockIdx.z` selects the frame, and each frame uses its own rotation matrix from the array `R_all`. All frames are computed fully in parallel on the GPU — no loop on the host side. Output is a contiguous `N × 512 × 512` buffer. Used by `--frames N` to produce the rotating video.

---

## Validation

After each run, GPU and CPU outputs are compared pixel by pixel. A pixel is flagged if the difference exceeds 2. All tested parameter combinations produced 0.00% difference.

---

## Speedup analysis

Benchmarked on **banana.ua.pt** — GPU: NVIDIA GeForce GTX 1660 (6 GB VRAM, CUDA 10.2), CPU: Intel Xeon (Ubuntu 18.04).

### Per-kernel breakdown (threshold=10000, sigma=1.0, yaw=0°)

| Stage | CPU time | GPU kernel time | Speedup |
|---|---|---|---|
| Threshold | ~66 ms | 2.40 ms | ~27x |
| Gaussian Blur | ~4200 ms | 24.05 ms | ~175x |
| MIP Projection | ~2430 ms | 1.80 ms | ~1350x |

### End-to-end (includes Host↔Device transfers + memory allocation)

| Configuration | CPU time | GPU time | Speedup |
|---|---|---|---|
| threshold=10000, sigma=1.0, yaw=0° | 6702.4 ms | 45.2 ms | **148.2x** |

The GPU end-to-end time covers the full pipeline: Host→Device transfer (180 MiB), the three kernels, and Device→Host transfer. CUDA context initialization is excluded via a warmup call before timing, following standard GPU benchmarking practice.

Per-kernel times are measured with `cudaEvent` pairs around each kernel launch, providing precise GPU-side timing independent of PCIe transfer overhead.

The three stages parallelised on the GPU:
- **Threshold** — trivially parallel (one thread per voxel, ~94.5 M threads)
- **Gaussian blur** — 3D stencil, fully parallel over output voxels (8×8×8 blocks)
- **MIP** — embarrassingly parallel over output pixels (16×16 blocks)

---

## Project structure

```
bunnyMIP/
├── data/           # 361 CT scan slices (512x512 each)
├── include/        # Headers
├── output/         # Output images
├── src/
│   ├── bunnyMIP.cpp        # Sequential CPU reference
│   ├── device-bunnyMIP.cu  # CUDA kernels
│   ├── main.cpp            # Entry point, CLI parsing, validation
│   ├── bunnyIO.cpp         # Volume loader
│   ├── imageIO.cpp         # PGM writer
│   └── print.cpp
└── makefile
```
