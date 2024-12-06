#ifndef MORTON_H
#define MORTON_H
#include "../include/vec_math.hpp"
#include "../include/cuda_buffer.hpp"
#include <corecrt_math.h>
#include "cuda_runtime.h"
#include "../include/cuda_types.hpp"
// 声明 CUDA 函数接口
#ifdef __cplusplus
extern "C" {
#endif
	namespace bvh_radSearch {
		void sortmorton(cudaBuffer<query_t> &queries, int points_num);
	}
#ifdef __cplusplus
}
#endif

#endif // KERNEL_H