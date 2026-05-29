
#pragma once

#include <cstdint>

void device_bunny_mip(const uint16_t* input, uint16_t threshold,float sigma, const float* R, uint16_t* output);

// Generates num_frames MIP images in one kernel call.
void device_bunny_generate_video(const uint16_t* input, uint16_t threshold, float sigma, const float* R_all, int num_frames, uint16_t* output);
