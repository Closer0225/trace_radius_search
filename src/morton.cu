#pragma once
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/gather.h>
#include <thrust/device_vector.h>
#include "../include/morton.h"
namespace bvh_radSearch {
	struct MapByIndex {
		query_t* data;
		int* indices;

		__device__ query_t operator()(int i) const {
			return data[indices[i]];
		}
	};
	struct Morton
	{
		const static int levels = 10;
		const static int bits_per_level = 3;
		const static int nbits = levels * bits_per_level;

		using code_t = int;

		__device__ __host__ __forceinline__
			static int spreadBits(int x, int offset)
		{
			//......................9876543210
			x = (x | (x << 10)) & 0x000f801f; //............98765..........43210
			x = (x | (x << 4)) & 0x00e181c3; //........987....56......432....10
			x = (x | (x << 2)) & 0x03248649; //......98..7..5..6....43..2..1..0
			x = (x | (x << 2)) & 0x09249249; //....9..8..7..5..6..4..3..2..1..0

			return x << offset;
		}

		__device__ __host__ __forceinline__
			static int compactBits(int x, int offset)
		{
			x = (x >> offset) & 0x09249249;  //....9..8..7..5..6..4..3..2..1..0
			x = (x | (x >> 2)) & 0x03248649;  //......98..7..5..6....43..2..1..0                                          
			x = (x | (x >> 2)) & 0x00e181c3;  //........987....56......432....10                                       
			x = (x | (x >> 4)) & 0x000f801f;  //............98765..........43210                                          
			x = (x | (x >> 10)) & 0x000003FF;  //......................9876543210        

			return x;
		}

		__device__ __host__ __forceinline__
			static code_t createCode(int cell_x, int cell_y, int cell_z)
		{
			return spreadBits(cell_x, 0) | spreadBits(cell_y, 1) | spreadBits(cell_z, 2);
		}

		__device__ __host__ __forceinline__
			static void decomposeCode(code_t code, int& cell_x, int& cell_y, int& cell_z)
		{
			cell_x = compactBits(code, 0);
			cell_y = compactBits(code, 1);
			cell_z = compactBits(code, 2);
		}

		__device__ __host__ __forceinline__
			static uint3 decomposeCode(code_t code)
		{
			return make_uint3(compactBits(code, 0), compactBits(code, 1), compactBits(code, 2));
		}
	};

	struct CalcMorton
	{
		const static int depth_mult = 1 << Morton::levels;

		float3 minp_;
		float3 dims_;

		__device__ __host__ __forceinline__ CalcMorton(query_t minp, query_t maxp) : minp_(minp.position)
		{
			dims_.x = maxp.position.x - minp.position.x;
			dims_.y = maxp.position.y - minp.position.y;
			dims_.z = maxp.position.z - minp.position.z;
		}

		__device__ __host__ __forceinline__ Morton::code_t operator()(const query_t& p) const
		{
			const int cellx = static_cast<int>(
				fminf(
					floorf(depth_mult * fminf(1.0f, fmaxf(0.0f, (p.position.x - minp_.x) / dims_.x))),
					static_cast<float>(depth_mult - 1)));
			const int celly = static_cast<int>(
				fminf(
					floorf(depth_mult * fminf(1.0f, fmaxf(0.0f, (p.position.y - minp_.y) / dims_.y))),
					static_cast<float>(depth_mult - 1)));
			const int cellz = static_cast<int>(
				fminf(
					floorf(depth_mult * fminf(1.0f, fmaxf(0.0f, (p.position.z - minp_.z) / dims_.z))),
					static_cast<float>(depth_mult - 1)));

			return Morton::createCode(cellx, celly, cellz);
		}
	};
	struct SelectMaxPoint
	{
		__host__ __device__ __forceinline__ query_t operator()(const query_t& e1, const query_t& e2) const
		{
			query_t result;
			result.position.x = fmax(e1.position.x, e2.position.x);
			result.position.y = fmax(e1.position.y, e2.position.y);
			result.position.z = fmax(e1.position.z, e2.position.z);
			result.radius = e1.radius;
			result.count = e1.count;
			return result;
		}
	};
	struct SelectMinPoint
	{
		__host__ __device__ __forceinline__ query_t operator()(const query_t& e1, const query_t& e2) const
		{
			query_t result;
			result.position.x = fmin(e1.position.x, e2.position.x);
			result.position.y = fmin(e1.position.y, e2.position.y);
			result.position.z = fmin(e1.position.z, e2.position.z);
			result.radius = e1.radius;
			result.count = e1.count;
			return result;
		}
	};

	void sortmorton(cudaBuffer<query_t>&queries, int points_num) {
		//找到该三维点云中的最大和最小x,y,z的值
		thrust::device_ptr<query_t> beg(queries.getPtr<query_t>());
		thrust::device_ptr<query_t> end = beg + points_num;

		//ScopeTimer timer("reduce"); 
		query_t atmax, atmin;
		atmax.position.x = atmax.position.y = atmax.position.z = std::numeric_limits<float>::max();
		atmin.position.x = atmin.position.y = atmin.position.z = std::numeric_limits<float>::lowest();
		atmax.radius = 0; atmin.radius = 0; atmax.count = 0; atmin.count = 0;
		query_t minp = thrust::reduce(beg, end, atmax, SelectMinPoint());
		query_t maxp = thrust::reduce(beg, end, atmin, SelectMaxPoint());


		cudaBuffer<int> codes;
		codes.alloc(points_num);
		cudaBuffer<int> indices;
		indices.alloc(points_num);

		thrust::device_ptr<int> codes_beg(codes.getPtr<int>());
		thrust::device_ptr<int> codes_end = codes_beg + points_num;

		thrust::device_ptr<int> indices_beg(indices.getPtr<int>());
		thrust::device_ptr<int> indices_end = indices_beg + points_num;

		//ScopeTimer timer("morton"); 
		thrust::transform(beg, end, codes_beg, CalcMorton(minp, maxp));

		//ScopeTimer timer("sort"); 
		//0,1,2,3,4,5,……
		thrust::sequence(indices_beg, indices_end);
		thrust::sort_by_key(codes_beg, codes_end, indices_beg);

		cudaBuffer<query_t> reordered_queries;
		reordered_queries.alloc(points_num);
		thrust::device_ptr<query_t> reordered_beg(reordered_queries.getPtr<query_t>());
		thrust::device_ptr<query_t> reordered_end = reordered_beg + points_num;
		thrust::gather(indices_beg, indices_end, beg, reordered_beg);
		/*thrust::transform(
			thrust::make_permutation_iterator(beg, indices_beg),
			thrust::make_permutation_iterator(end, indices_end),
			reordered_beg,
			thrust::identity<query_t>() 
		);*/
		cudaMemcpy(queries.getPtr<query_t>(), reordered_queries.getPtr<query_t>(),
			points_num * sizeof(query_t), cudaMemcpyDeviceToDevice);
	}
}