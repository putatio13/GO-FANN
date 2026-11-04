#pragma once

#include <cstdint>


//#define __ENABLE_HASH

typedef float data_value_t;

#ifdef __ENABLE_HASH
typedef unsigned int value_t;
typedef int dist_t;
#else
typedef uint8_t value_t;
//typedef int8_t value_t;
//typedef float value_t;
typedef float dist_t;
#endif
typedef uint64_t idx_t;
typedef int UINT;


//#define ACC_BATCH_SIZE 4096
#define ACC_BATCH_SIZE 1000000

//for GPU
#ifndef FIXED_DEGREE
#define FIXED_DEGREE 64
#endif

#ifndef FIXED_DEGREE_SHIFT
#define FIXED_DEGREE_SHIFT 5
#endif

//for CPU construction
#define SEARCH_DEGREE 15
#define CONSTRUCT_SEARCH_BUDGET 150
