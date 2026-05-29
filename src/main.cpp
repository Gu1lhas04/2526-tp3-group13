
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "print.h"
#include "bunnyIO.h"
#include "imageIO.h"

#include "bunnyMIP.h"
#include "device-bunnyMIP.h"

inline float d2r(float angle) { return angle * M_PI / 180.0; }

void generate_rotation_matrix(float pitch, float yaw, float roll, float* R, bool inverse = false) {
    float cp = std::cos(pitch); float sp = std::sin(pitch);
    float cy = std::cos(yaw);   float sy = std::sin(yaw);
    float cr = std::cos(roll);  float sr = std::sin(roll);

    // Combined rotation R = Rz * Ry * Rx
    float m[3][3];
    m[0][0] = cy * cr;
    m[0][1] = cy * sr;
    m[0][2] = -sy;

    m[1][0] = sp * sy * cr - cp * sr;
    m[1][1] = sp * sy * sr + cp * cr;
    m[1][2] = sp * cy;

    m[2][0] = cp * sy * cr + sp * sr;
    m[2][1] = cp * sy * sr - sp * cr;
    m[2][2] = cp * cy;

    // Fill the flattened array
    if (!inverse) {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                R[i * 3 + j] = m[i][j];
    } else {
        // Transpose for backward mapping (Inverse rotation)
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                R[j * 3 + i] = m[i][j];
    }
}



int main(int argc, char* argv[]) {
    print("CLE2026 - BunnyMIP\n");

    // Default parameters
    uint16_t threshold = 1 << 15;
    float sigma = 1.0f;
    float roll = 0.0f;
    float pitch = 0.0f;
    float yaw = 0.0f;
    int num_frames = 0;

    // CLI argument parsing
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], "--threshold") == 0) threshold = (uint16_t)atoi(argv[++i]);
        else if (strcmp(argv[i], "--sigma") == 0) sigma = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--roll") == 0) roll = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--pitch") == 0) pitch = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--yaw") == 0) yaw = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--frames") == 0) num_frames = atoi(argv[++i]);
    }

    print("Parameters: threshold=%d sigma=%.2f roll=%.1f pitch=%.1f yaw=%.1f frames=%d\n",
          threshold, sigma, roll, pitch, yaw, num_frames);

    uint16_t* volume = loadBunnyCT("data");

    // Raster output when running on the host
    uint16_t* h_raster = new uint16_t[kBunnySize*kBunnySize];
    // Raster output when running on the GPU
    uint16_t* d_raster = new uint16_t[kBunnySize*kBunnySize];

    float R[3*3];
    generate_rotation_matrix(d2r(pitch), d2r(yaw), d2r(roll), R);

    // CPU
    host_bunny_mip(volume, threshold, sigma, R, h_raster);
    // GPU (CUDA)
    device_bunny_mip(volume, threshold, sigma , R, d_raster);

    int raster_size = kBunnySize * kBunnySize;
    int diff = 0;
    for (int i = 0; i < raster_size; i++) {
        int local_diff = abs((int)h_raster[i]) - ((int)d_raster[i]);
        if (local_diff > 2)
            diff = diff + 1;
    }

    print("\n>> Output difference: %.2f%%\n", (diff / (float)raster_size) * 100.0);

    savePGM16("output/bunnyMIP_cpu.pgm", h_raster, kBunnySize, kBunnySize);
    savePGM16("output/bunnyMIP_gpu.pgm", d_raster, kBunnySize, kBunnySize);

    // Video generation
    if (num_frames > 0) {
        print("\nGenerating video with %d frames...\n", num_frames);

        // Build one rotation matrix per frame, rotating yaw evenly over 360 degrees
        float* R_all = new float[num_frames * 9];
        float step = 360.0f / num_frames;
        for (int i = 0; i < num_frames; i++)
            generate_rotation_matrix(d2r(pitch), d2r(yaw + i * step), d2r(roll), R_all + i * 9);

        uint16_t* frames = new uint16_t[(size_t)num_frames * kBunnySize * kBunnySize];
        device_bunny_generate_video(volume, threshold, sigma, R_all, num_frames, frames);

        // Save each frame as a PGM file
        for (int i = 0; i < num_frames; i++) {
            char filename[64];
            snprintf(filename, sizeof(filename), "output/frame_%03d.pgm", i);
            savePGM16(filename, frames + (size_t)i * kBunnySize * kBunnySize, kBunnySize, kBunnySize);
        }

        // Combine frames into a video using ffmpeg
        print("Running ffmpeg...\n");
        system("ffmpeg -y -framerate 24 -i output/frame_%03d.pgm "
               "-vf scale=512:512 -c:v libx264 -pix_fmt yuv420p output/bunny.mp4");
        print("Video saved to output/bunny.mp4\n");

        delete[] R_all;
        delete[] frames;
    }

    delete [] volume;
    return 0;
}
