#ifndef UTILS_H
#define UTILS_H

#pragma once

#ifdef KERNEL_EXPORTS
#define KERNEL_API __declspec(dllexport)
#else
#define KERNEL_API __declspec(dllimport)
#endif


#include <curand_kernel.h>

extern __global__ void generate_random_numbers(curandState* state, double* random_numbers, int num_elements);
extern __global__ void setup_rng(curandState* state, unsigned long seed);
extern __host__ double mean_of_array(double* array, int size);

#endif // UTILS_H
