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

---

## Validation

After each run, GPU and CPU outputs are compared pixel by pixel. A pixel is flagged if the difference exceeds 2. All tested parameter combinations produced 0.00% difference.

---

## Speedup analysis

Benchmarked on **banana.ua.pt** — GPU: NVIDIA GeForce GTX 1660 (6 GB VRAM, CUDA 10.2), CPU: Intel Xeon (Ubuntu 18.04).

| Configuration | CPU time | GPU time | Speedup |
|---|---|---|---|
| threshold=10000, sigma=1.0, yaw=0° | 6730.7 ms | 837.0 ms | **8.0x** |
| threshold=10000, sigma=1.0, yaw=45° | 6321.1 ms | 844.4 ms | **7.5x** |
| threshold=10000, sigma=1.0, yaw=90° | 4669.5 ms | 842.3 ms | **5.5x** |

The GPU time is roughly constant (~840 ms) across all rotations because all 512×512 output threads always run, regardless of angle. The CPU time drops at yaw=90° because rays exit the thinner axis of the volume sooner, shortening the inner loop.

The measured speedup (~8x) represents the full wall-clock pipeline including PCIe memory transfers (180 MiB Host→Device + raster Device→Host). The pure compute speedup of the kernels is higher; the PCIe bottleneck limits the overall gain observed in wall-clock time.

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
