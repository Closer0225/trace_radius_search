

#pragma once

#include <optix.h>
#include <stdint.h>
#include <cstdio>

#include "../include/vec_math.hpp"
#include "../include/cuda_types.hpp"
const int order = 2;
const int maxresult = 150;
namespace bvh_radSearch
{
	template <class T>
	__device__ __host__ __forceinline__ void dswap(T& a, T& b)
	{
		T c(a); a = b; b = c;
	}

	__device__ __forceinline__ void computeRoots2(const float& b, const float& c, float3& roots)
	{
		roots.x = 0.f;
		float d = b * b - 4.f * c;
		if (d < 0.f)
			d = 0.f;
		float sd = sqrtf(d);
		roots.z = 0.5f * (-b + sd);
		roots.y = 0.5f * (-b - sd);
	}

	__device__ __forceinline__ void computeRoots3(float c0, float c1, float c2, float3& roots)
	{
		if (c0 == 0)
		{
			computeRoots2(c2, c1, roots);
		}
		else
		{
			const float s_inv3 = 1.f / 3.f;
			const float s_sqrt3 = sqrtf(3.f);
			float c2_over_3 = c2 * s_inv3;
			float a_over_3 = (c1 - c2 * c2_over_3) * s_inv3;
			if (a_over_3 > 0.f)
				a_over_3 = 0.f;

			float half_b = 0.5f * (c0 + c2_over_3 * (2.f * c2_over_3 * c2_over_3 - c1));

			float q = half_b * half_b + a_over_3 * a_over_3 * a_over_3;
			if (q > 0.f)
				q = 0.f;
			float rho = sqrtf(-a_over_3);
			float theta = atan2(sqrtf(-q), half_b) * s_inv3;
			float cos_theta = __cosf(theta);
			float sin_theta = __sinf(theta);
			roots.x = c2_over_3 + 2.f * rho * cos_theta;
			roots.y = c2_over_3 - rho * (cos_theta + s_sqrt3 * sin_theta);
			roots.z = c2_over_3 - rho * (cos_theta - s_sqrt3 * sin_theta);
			//将得到的根按大小排列，x最小，y最大
			if (roots.x >= roots.y)
				dswap(roots.x, roots.y);

			if (roots.y >= roots.z)
			{
				dswap(roots.y, roots.z);

				if (roots.x >= roots.y) {
					dswap(roots.x, roots.y);
				}
			}
			if (roots.x <= 0) // 对称正半定矩阵的本征值不能是负的！将其设置为0
				computeRoots2(c2, c1, roots);
		}

	}

	__device__ __forceinline__ double3
		operator+(const double3& v1, const double3& v2)
	{
		return make_double3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
	}

	__device__ __forceinline__ float3
		operator/(const double3& v1, const int& v)
	{
		return make_float3(v1.x / v, v1.y / v, v1.z / v);
	}

	__device__ __forceinline__ float weight_func(const float sq_dist, const float search_radius) {
		return std::exp(-sq_dist / search_radius / search_radius);
	};

	__device__  __forceinline__ void asDiagonal(const float* a, const float* b, float* result, int n, int nr) {
		for (int i = 0; i < nr; i++)
			for (int j = 0; j < n; j++)
				result[i * maxresult + j] = a[i * maxresult + j] * b[j];
	}

	__device__  __forceinline__ void MatirxCross(const float* a, const float* b, float* result, int n, int nr) {
		for (int i = 0; i < nr; i++)
			for (int j = 0; j < nr; j++)
				for (int k = 0; k < n; k++)
					result[i * nr + j] += a[i * maxresult + k] * b[j * maxresult + k];
	}

	__device__  __forceinline__ void MatrixVectorCross(const float* a, const float* b, float* result, int n, int nr) {
		for (int i = 0; i < nr; i++)
			for (int j = 0; j < n; j++)
				result[i] += a[i * maxresult + j] * b[j];
	}

	const int nr_coeff = (order + 1) * (order + 2) / 2;

	const int Matirxn = nr_coeff * (nr_coeff + 1) / 2;

	class SymmetricMatrix {
	public:
		__device__  __forceinline__ SymmetricMatrix(float* P_weight_Pt, int n) {
			n_ = n;
			for (int i = 0; i < n_; i++)
				for (int j = i; j < n_; j++)
					data_[i * n - i * (i - 1) / 2 + j - i] = P_weight_Pt[i * n + j];
		}

		__device__  __forceinline__ float& operator()(int i, int j) {
			if (i > j) dswap(i, j);
			return data_[i * n_ - i * (i - 1) / 2 + j - i];
		}

		__device__  __forceinline__ const float& operator()(int i, int j) const {
			if (i > j) dswap(i, j);
			return data_[i * n_ - i * (i - 1) / 2 + j - i];
		}

		__device__  __forceinline__ int size() const { return n_; }

		int n_;
		float data_[Matirxn];
	};
	__device__  __forceinline__ bool llt(SymmetricMatrix& A) {
		int n = A.size();
		for (int k = 0; k < n; ++k) {
			if (A(k, k) <= 0) {
				//printf("Matrix is not positive definite.\n" );
				return false;
			}
			A(k, k) = sqrt(A(k, k));
			for (int i = k + 1; i < n; ++i) {
				A(i, k) /= A(k, k);
			}

			for (int i = k + 1; i < n; ++i) {
				for (int j = k + 1; j <= i; ++j) {
					A(i, j) -= A(i, k) * A(j, k);
				}
			}
		}
		return true;
	}

	__device__  __forceinline__ void solveInPlace(const SymmetricMatrix& A, float* b) {
		int n = A.size();
		for (int i = 0; i < n; ++i) {
			for (int j = 0; j < i; ++j) {
				b[i] -= A(i, j) * b[j];
			}
			b[i] /= A(i, i);
		}
		for (int i = n - 1; i >= 0; --i) {
			for (int j = i + 1; j < n; ++j) {
				b[i] -= A(j, i) * b[j];  // 注意A(j, i)等同于A(i, j)
			}
			b[i] /= A(i, i);
		}
	}

	//计算三维向量
	struct Eigen33
	{
	public:
		template<int Rows>
		struct MiniMat
		{
			float3 data[Rows];
			__device__ __host__ __forceinline__ float3& operator[](int i) { return data[i]; }
			__device__ __host__ __forceinline__ const float3& operator[](int i) const { return data[i]; }
		};
		using Mat33 = MiniMat<3>;
		using Mat43 = MiniMat<4>;

		//用于计算给定向量垂直的单位向量
		static __forceinline__ __device__ float3 unitOrthogonal(const float3& src)
		{
			float3 perp;
			//x!=0 || y!=0
			if (!(src.x == 0) || !(src.y == 0))
			{
				float invnm = rsqrtf(src.x * src.x + src.y * src.y);
				perp.x = -src.y * invnm;
				perp.y = src.x * invnm;
				perp.z = 0.0f;
			}
			// x==0&&y==0
			else
			{
				float invnm = rsqrtf(src.z * src.z + src.y * src.y);
				perp.x = 0.0f;
				perp.y = -src.z * invnm;
				perp.z = src.y * invnm;
			}

			return perp;
		}

		__device__ __forceinline__ Eigen33(volatile float* mat_pkg_arg) : mat_pkg(mat_pkg_arg) {}

		__device__ __forceinline__ void compute(Mat33& tmp, Mat33& vec_tmp, Mat33& evecs, float3& evals)
		{

			float max01 = fmaxf(std::abs(mat_pkg[0]), std::abs(mat_pkg[1]));
			float max23 = fmaxf(std::abs(mat_pkg[2]), std::abs(mat_pkg[3]));
			float max45 = fmaxf(std::abs(mat_pkg[4]), std::abs(mat_pkg[5]));
			float m0123 = fmaxf(max01, max23);
			float scale = fmaxf(max45, m0123);

			if (scale <= 0)
				scale = 1.f;

			mat_pkg[0] /= scale;
			mat_pkg[1] /= scale;
			mat_pkg[2] /= scale;
			mat_pkg[3] /= scale;
			mat_pkg[4] /= scale;
			mat_pkg[5] /= scale;

			float c0 = m00() * m11() * m22()
				+ 2.f * m01() * m02() * m12()
				- m00() * m12() * m12()
				- m11() * m02() * m02()
				- m22() * m01() * m01();
			float c1 = m00() * m11() -
				m01() * m01() +
				m00() * m22() -
				m02() * m02() +
				m11() * m22() -
				m12() * m12();
			float c2 = m00() + m11() + m22();
			// x^3 - c2*x^2 + c1*x - c0 = 0
			computeRoots3(c0, c1, c2, evals);

			//最大值和最小值相等  以下部分已被演算过  正确
			if (evals.x == evals.z)
			{
				evecs[0] = make_float3(1.f, 0.f, 0.f);
				evecs[1] = make_float3(0.f, 1.f, 0.f);
				evecs[2] = make_float3(0.f, 0.f, 1.f);
			}
			//两个最小值相等
			else if (evals.x == evals.y)
			{
				// first and second equal                
				tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
				tmp[0].x -= evals.z; tmp[1].y -= evals.z; tmp[2].z -= evals.z;

				vec_tmp[0] = cross(tmp[0], tmp[1]);
				vec_tmp[1] = cross(tmp[0], tmp[2]);
				vec_tmp[2] = cross(tmp[1], tmp[2]);

				float len1 = dot(vec_tmp[0], vec_tmp[0]);
				float len2 = dot(vec_tmp[1], vec_tmp[1]);
				float len3 = dot(vec_tmp[2], vec_tmp[2]);

				if (len1 >= len2 && len1 >= len3)
				{
					evecs[2] = vec_tmp[0] * rsqrtf(len1);
				}
				else if (len2 >= len1 && len2 >= len3)
				{
					evecs[2] = vec_tmp[1] * rsqrtf(len2);
				}
				else
				{
					evecs[2] = vec_tmp[2] * rsqrtf(len3);
				}

				evecs[1] = unitOrthogonal(evecs[2]);
				evecs[0] = cross(evecs[1], evecs[2]);

			}
			//两个最大值相等
			else if (evals.z == evals.y)
			{
				// second and third equal                                    
				tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();

				tmp[0].x -= evals.x; tmp[1].y -= evals.x; tmp[2].z -= evals.x;

				vec_tmp[0] = cross(tmp[0], tmp[1]);
				vec_tmp[1] = cross(tmp[0], tmp[2]);
				vec_tmp[2] = cross(tmp[1], tmp[2]);

				float len1 = dot(vec_tmp[0], vec_tmp[0]);
				float len2 = dot(vec_tmp[1], vec_tmp[1]);
				float len3 = dot(vec_tmp[2], vec_tmp[2]);

				if (len1 >= len2 && len1 >= len3)
				{
					evecs[0] = vec_tmp[0] * rsqrtf(len1);
				}
				else if (len2 >= len1 && len2 >= len3)
				{
					evecs[0] = vec_tmp[1] * rsqrtf(len2);
				}
				else
				{
					evecs[0] = vec_tmp[2] * rsqrtf(len3);
				}

				evecs[1] = unitOrthogonal(evecs[0]);
				evecs[2] = cross(evecs[0], evecs[1]);
			}
			//三个不同的特征值
			else
			{

				tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
				tmp[0].x -= evals.z; tmp[1].y -= evals.z; tmp[2].z -= evals.z;

				vec_tmp[0] = cross(tmp[0], tmp[1]);
				vec_tmp[1] = cross(tmp[0], tmp[2]);
				vec_tmp[2] = cross(tmp[1], tmp[2]);

				float len1 = dot(vec_tmp[0], vec_tmp[0]);
				float len2 = dot(vec_tmp[1], vec_tmp[1]);
				float len3 = dot(vec_tmp[2], vec_tmp[2]);

				float mmax[3];

				unsigned int min_el = 2;
				unsigned int max_el = 2;
				if (len1 >= len2 && len1 >= len3)
				{
					mmax[2] = len1;
					evecs[2] = vec_tmp[0] * rsqrtf(len1);
				}
				else if (len2 >= len1 && len2 >= len3)
				{
					mmax[2] = len2;
					evecs[2] = vec_tmp[1] * rsqrtf(len2);
				}
				else
				{
					mmax[2] = len3;
					evecs[2] = vec_tmp[2] * rsqrtf(len3);
				}

				tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
				tmp[0].x -= evals.y; tmp[1].y -= evals.y; tmp[2].z -= evals.y;

				vec_tmp[0] = cross(tmp[0], tmp[1]);
				vec_tmp[1] = cross(tmp[0], tmp[2]);
				vec_tmp[2] = cross(tmp[1], tmp[2]);

				len1 = dot(vec_tmp[0], vec_tmp[0]);
				len2 = dot(vec_tmp[1], vec_tmp[1]);
				len3 = dot(vec_tmp[2], vec_tmp[2]);

				if (len1 >= len2 && len1 >= len3)
				{
					mmax[1] = len1;
					evecs[1] = vec_tmp[0] * rsqrtf(len1);
					min_el = len1 <= mmax[min_el] ? 1 : min_el;
					max_el = len1 > mmax[max_el] ? 1 : max_el;
				}
				else if (len2 >= len1 && len2 >= len3)
				{
					mmax[1] = len2;
					evecs[1] = vec_tmp[1] * rsqrtf(len2);
					min_el = len2 <= mmax[min_el] ? 1 : min_el;
					max_el = len2 > mmax[max_el] ? 1 : max_el;
				}
				else
				{
					mmax[1] = len3;
					evecs[1] = vec_tmp[2] * rsqrtf(len3);
					min_el = len3 <= mmax[min_el] ? 1 : min_el;
					max_el = len3 > mmax[max_el] ? 1 : max_el;
				}

				tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
				tmp[0].x -= evals.x; tmp[1].y -= evals.x; tmp[2].z -= evals.x;

				vec_tmp[0] = cross(tmp[0], tmp[1]);
				vec_tmp[1] = cross(tmp[0], tmp[2]);
				vec_tmp[2] = cross(tmp[1], tmp[2]);

				len1 = dot(vec_tmp[0], vec_tmp[0]);
				len2 = dot(vec_tmp[1], vec_tmp[1]);
				len3 = dot(vec_tmp[2], vec_tmp[2]);


				if (len1 >= len2 && len1 >= len3)
				{
					mmax[0] = len1;
					evecs[0] = vec_tmp[0] * rsqrtf(len1);
					min_el = len3 <= mmax[min_el] ? 0 : min_el;
					max_el = len3 > mmax[max_el] ? 0 : max_el;
				}
				else if (len2 >= len1 && len2 >= len3)
				{
					mmax[0] = len2;
					evecs[0] = vec_tmp[1] * rsqrtf(len2);
					min_el = len3 <= mmax[min_el] ? 0 : min_el;
					max_el = len3 > mmax[max_el] ? 0 : max_el;
				}
				else
				{
					mmax[0] = len3;
					evecs[0] = vec_tmp[2] * rsqrtf(len3);
					min_el = len3 <= mmax[min_el] ? 0 : min_el;
					max_el = len3 > mmax[max_el] ? 0 : max_el;
				}

				unsigned mid_el = 3 - min_el - max_el;
				evecs[min_el] = normalize(cross(evecs[(min_el + 1) % 3], evecs[(min_el + 2) % 3]));
				evecs[mid_el] = normalize(cross(evecs[(mid_el + 1) % 3], evecs[(mid_el + 2) % 3]));

			}
			evals *= scale;

		}
	private:
		volatile float* mat_pkg;

		__device__  __forceinline__ float m00() const { return mat_pkg[0]; }
		__device__  __forceinline__ float m01() const { return mat_pkg[1]; }
		__device__  __forceinline__ float m02() const { return mat_pkg[2]; }
		__device__  __forceinline__ float m10() const { return mat_pkg[1]; }
		__device__  __forceinline__ float m11() const { return mat_pkg[3]; }
		__device__  __forceinline__ float m12() const { return mat_pkg[4]; }
		__device__  __forceinline__ float m20() const { return mat_pkg[2]; }
		__device__  __forceinline__ float m21() const { return mat_pkg[4]; }
		__device__  __forceinline__ float m22() const { return mat_pkg[5]; }

		__device__  __forceinline__ float3 row0() const { return make_float3(m00(), m01(), m02()); }
		__device__  __forceinline__ float3 row1() const { return make_float3(m10(), m11(), m12()); }
		__device__  __forceinline__ float3 row2() const { return make_float3(m20(), m21(), m22()); }

	};
	struct payload_t
	{
		query_t query;
		uint32_t count;

		uint32_t offset;
		int32_t maxDistElemi;
		int32_t foundNeighbors;
		float_t maxDistElemf;
		int optixIndices[maxresult];
		float optixDists[maxresult];
	};

	__device__ __forceinline__ void* unpackPointer(
		uint32_t i0,
		uint32_t i1) noexcept
	{
		const uint64_t uptr = static_cast<uint64_t>(i0) << 32 | i1;
		void* ptr = reinterpret_cast<void*>(uptr);
		return ptr;
	}

	__device__ __forceinline__ void packPointer(
		const void* ptr,
		uint32_t& i0,
		uint32_t& i1) noexcept
	{
		const uint64_t uptr = reinterpret_cast<uint64_t>(ptr);
		i0 = uptr >> 32;
		i1 = uptr & 0x00000000ffffffff;
	}

	template<typename T>
	__device__ __forceinline__ T* getPayload(void) noexcept
	{
		const uint32_t u0 = optixGetPayload_0();
		const uint32_t u1 = optixGetPayload_1();
		return reinterpret_cast<T*>(unpackPointer(u0, u1));
	}

	extern "C" { __constant__ Params params; }

	__device__ void findLargestDist(payload_t& payload) noexcept
	{
		payload.maxDistElemi = 0;
		payload.maxDistElemf = payload.optixDists[payload.maxDistElemi];

		for (int32_t k = 1; k < params.knn; ++k)
		{
			float_t tmpDist = payload.optixDists[k];
			if (tmpDist > payload.maxDistElemf)
			{
				payload.maxDistElemi = k;
				payload.maxDistElemf = tmpDist;
			}
		}
	}


	extern "C" __global__ void __raygen__radSearch_count(void)
	{
		const uint3& idx = optixGetLaunchIndex();
		query_t& query = params.queries[idx.x];
		payload_t payload;
		payload.query = query;
		payload.count = 0;

		uint32_t u0, u1;
		packPointer(&payload, u0, u1);

		optixTrace(params.gasHandle,
			query.position, make_float3(1.e-16f),
			0.f, 1.e-16f, 0.f,
			OptixVisibilityMask(1),
			OPTIX_RAY_FLAG_DISABLE_ANYHIT |
			OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
			0, 4, 0,
			u0, u1);

		query.count = payload.count;
		atomicAdd(&params.totalCount[0], payload.count);
		atomicMax(&params.maxCount[0], payload.count);
		atomicMin(&params.minCount[0], payload.count);
	}

	extern "C" __global__ void __intersection__radSearch_count(void)
	{
		payload_t& payload = *getPayload<payload_t>();

		float3& sample = params.samplePos[optixGetPrimitiveIndex()];

		const float3 diff = sample - optixGetWorldRayOrigin();
		const float_t t = dot(diff, diff);

		if (t < payload.query.radius * payload.query.radius)
		{
			++payload.count;
		}
	}

	extern "C" __global__ void __raygen__radSearch(void)
	{
		const uint3& idx = optixGetLaunchIndex();
		query_t& query = params.queries[idx.x];
		payload_t payload;
		payload.query = query;
		payload.count = 0;
		payload.offset = idx.x * params.knn;
		payload.maxDistElemi = idx.x * params.knn;
		payload.maxDistElemf = query.radius + 1.f;
		payload.foundNeighbors = 0;
		query.count = params.knn;

		uint32_t u0, u1;
		packPointer(&payload, u0, u1);

		optixTrace(params.gasHandle,
			query.position, make_float3(1.e-16f),
			0.f, 1.e-16f, 0.f,
			OptixVisibilityMask(1),
			OPTIX_RAY_FLAG_DISABLE_ANYHIT |
			OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
			0, 4, 0,
			u0, u1);
	}

	extern "C" __global__ void __intersection__radSearch(void)
	{
		payload_t& payload = *getPayload<payload_t>();

		float3& sample = params.samplePos[optixGetPrimitiveIndex()];

		const float3 diff = sample - optixGetWorldRayOrigin();
		const float_t t = dot(diff, diff);

		if (t < payload.query.radius * payload.query.radius)
		{

			const uint32_t idxToSave = payload.offset + payload.foundNeighbors;
			params.optixIndices[idxToSave] = optixGetPrimitiveIndex();
			params.optixDists[idxToSave] = t;
			++payload.foundNeighbors;
		}

	}

	extern "C" __global__ void __raygen__radSearch_smooth(void)
	{
		const uint3& idx = optixGetLaunchIndex();
		query_t& query = params.queries[idx.x];
		payload_t payload;
		payload.query = query;
		payload.foundNeighbors = 0;

		uint32_t u0, u1;
		packPointer(&payload, u0, u1);

		optixTrace(params.gasHandle,
			query.position, make_float3(1.e-16f),
			0.f, 1.e-16f, 0.f,
			OptixVisibilityMask(1),
			OPTIX_RAY_FLAG_DISABLE_ANYHIT |
			OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
			0, 4, 0,
			u0, u1);

		if (payload.foundNeighbors >= 3)
		{
			float3 center = make_float3(0.0, 0.0, 0.0);
			float3 dpoint = make_float3(0.f, 0.f, 0.f);
			for (int i = 0; i < payload.foundNeighbors; ++i)
				dpoint = dpoint + make_float3(params.samplePos[payload.optixIndices[i]].x, params.samplePos[payload.optixIndices[i]].y, params.samplePos[payload.optixIndices[i]].z);
			center = dpoint / payload.foundNeighbors;

			float dx2 = 0, dxy = 0, dxz = 0, dy2 = 0, dyz = 0, dz2 = 0;
			float3  d;
			for (int i = 0; i < payload.foundNeighbors; i++) {
				dpoint = params.samplePos[payload.optixIndices[i]];
				d = dpoint - center;
				dx2 += d.x * d.x;  dxy += d.x * d.y;
				dxz += d.x * d.z;  dy2 += d.y * d.y;
				dyz += d.y * d.z;  dz2 += d.z * d.z;
			}
			float cov[6];
			cov[0] = dx2;
			cov[1] = dxy;
			cov[2] = dxz;
			cov[3] = dy2;
			cov[4] = dyz;
			cov[5] = dz2;

			Eigen33 eigen33(&cov[0]);

			float cov_buffer[3][9];

			Eigen33::Mat33& tmp = (Eigen33::Mat33&)cov_buffer[0][0];
			Eigen33::Mat33& vec_tmp = (Eigen33::Mat33&)cov_buffer[1][0];
			Eigen33::Mat33& evecs = (Eigen33::Mat33&)cov_buffer[2][0];
			float3 evals;

			eigen33.compute(tmp, vec_tmp, evecs, evals);

			params.normals[idx.x].x = evecs[0].x;
			params.normals[idx.x].y = evecs[0].y;
			params.normals[idx.x].z = evecs[0].z;
			float3 normal = params.normals[idx.x];
			float3 query = params.queries[idx.x].position;
			if (length(normal) > 0.5f)
			{
				float3 q;
				q = make_float3(params.queries[idx.x].position.x, params.queries[idx.x].position.y, params.queries[idx.x].position.z);
				query = query + normal * dot(normal, center - q);
			}
			if (order > 1)
			{
				float3 plane_normal;
				plane_normal.x = normal.x;
				plane_normal.y = normal.y;
				plane_normal.z = normal.z;
				auto v_axis = Eigen33::unitOrthogonal(plane_normal);
				auto u_axis = cross(plane_normal, v_axis);
				float search_radius = params.queries[idx.x].radius;

				auto num_neighbors = payload.foundNeighbors;
				if (order > 1)
				{

					if (num_neighbors >= nr_coeff)
					{
						float weight_vec[maxresult];
						float P[nr_coeff * maxresult];
						float f_vec[maxresult];
						float P_weight_Pt[nr_coeff * nr_coeff] = { 0 };
						float3 de_meaned[maxresult];
						for (std::size_t ni = 0; ni < num_neighbors; ++ni)
						{
							de_meaned[ni] = params.samplePos[payload.optixIndices[ni]] - query;
							weight_vec[ni] = weight_func(dot(de_meaned[ni], de_meaned[ni]), search_radius);//移植一个weight_func()函数
						}
						//遍历邻居，在局部坐标系中转换它们，保存高度和多项式项的值
						for (std::size_t ni = 0; ni < num_neighbors; ++ni)
						{
							//转换坐标
							const float u_coord = dot(de_meaned[ni], u_axis);
							const float v_coord = dot(de_meaned[ni], v_axis);
							f_vec[ni] = dot(de_meaned[ni], plane_normal);
							//计算多项式在当前点的项
							int j = 0;
							float u_pow = 1;
							for (int ui = 0; ui <= order; ++ui)
							{
								float v_pow = 1;
								for (int vi = 0; vi <= order - ui; ++vi, j++)
								{
									P[j * maxresult + ni] = u_pow * v_pow;
									v_pow *= v_coord;
								}
								u_pow *= u_coord;
							}
						}

						//计算系数
						float P_weight[nr_coeff * maxresult];
						asDiagonal(P, weight_vec, P_weight, num_neighbors, nr_coeff);
						MatirxCross(P_weight, P, P_weight_Pt, num_neighbors, nr_coeff);
						float c_vec[nr_coeff] = { 0 };
						MatrixVectorCross(P_weight, f_vec, c_vec, num_neighbors, nr_coeff);
						//使用LLT分解（Cholesky分解）来求解多项式系数 c_vec
						SymmetricMatrix A(P_weight_Pt, nr_coeff);
						bool isllt = llt(A);
						float3 proj_normal = make_float3(0.0, 0.0, 0.0);
						if (isllt) {
							solveInPlace(A, c_vec);
							//simple方法
							query = query + (normal * c_vec[0]);
							proj_normal = plane_normal - u_axis * c_vec[order + 1] - v_axis * c_vec[1];
							proj_normal = normalize(proj_normal);
							normal.x = proj_normal.x;
							normal.y = proj_normal.y;
							normal.z = proj_normal.z;
						}

					}
				}
			}
			params.normals[idx.x] = normal;
			params.queries[idx.x].position = make_float3(query.x, query.y, query.z);
		}
		else
			params.normals[idx.x] = make_float3(0.0f);
	}

	extern "C" __global__ void __intersection__radSearch_smooth(void)
	{
		payload_t& payload = *getPayload<payload_t>();

		float3& sample = params.samplePos[optixGetPrimitiveIndex()];

		const float3 diff = sample - optixGetWorldRayOrigin();
		const float_t t = dot(diff, diff);
		if (t < payload.query.radius * payload.query.radius && payload.foundNeighbors < maxresult)
		{
			payload.optixIndices[payload.foundNeighbors] = optixGetPrimitiveIndex();
			++payload.foundNeighbors;
		}
	}

	extern "C" __global__ void __raygen__knnSearch_smooth(void)
	{
		const uint3& idx = optixGetLaunchIndex();
		query_t& query = params.queries[idx.x];
		payload_t payload;
		payload.query = query;
		payload.maxDistElemi = 0;
		payload.maxDistElemf = query.radius + 1.f;
		payload.foundNeighbors = 0;

		uint32_t u0, u1;
		packPointer(&payload, u0, u1);

		optixTrace(params.gasHandle,
			query.position, make_float3(1.e-16f),
			0.f, 1.e-16f, 0.f,
			OptixVisibilityMask(1),
			OPTIX_RAY_FLAG_DISABLE_ANYHIT |
			OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
			0, 4, 0,
			u0, u1);

		if (payload.foundNeighbors >= 3)
		{
			float3 center = make_float3(0.0, 0.0, 0.0);
			float3 dpoint = make_float3(0.f, 0.f, 0.f);
			for (int i = 0; i < payload.foundNeighbors; ++i)
				dpoint = dpoint + make_float3(params.samplePos[payload.optixIndices[i]].x, params.samplePos[payload.optixIndices[i]].y, params.samplePos[payload.optixIndices[i]].z);
			center = dpoint / payload.foundNeighbors;

			float dx2 = 0, dxy = 0, dxz = 0, dy2 = 0, dyz = 0, dz2 = 0;
			float3  d;
			for (int i = 0; i < payload.foundNeighbors; i++) {
				dpoint = params.samplePos[payload.optixIndices[i]];
				d = dpoint - center;
				dx2 += d.x * d.x;  dxy += d.x * d.y;
				dxz += d.x * d.z;  dy2 += d.y * d.y;
				dyz += d.y * d.z;  dz2 += d.z * d.z;
			}
			float cov[6];
			cov[0] = dx2;
			cov[1] = dxy;
			cov[2] = dxz;
			cov[3] = dy2;
			cov[4] = dyz;
			cov[5] = dz2;

			Eigen33 eigen33(&cov[0]);

			float cov_buffer[3][9];

			Eigen33::Mat33& tmp = (Eigen33::Mat33&)cov_buffer[0][0];
			Eigen33::Mat33& vec_tmp = (Eigen33::Mat33&)cov_buffer[1][0];
			Eigen33::Mat33& evecs = (Eigen33::Mat33&)cov_buffer[2][0];
			float3 evals;

			eigen33.compute(tmp, vec_tmp, evecs, evals);

			params.normals[idx.x].x = evecs[0].x;
			params.normals[idx.x].y = evecs[0].y;
			params.normals[idx.x].z = evecs[0].z;
			float3 normal = params.normals[idx.x];
			float3 query = params.queries[idx.x].position;
			if (length(normal) > 0.5f)
			{
				float3 q;
				q = make_float3(params.queries[idx.x].position.x, params.queries[idx.x].position.y, params.queries[idx.x].position.z);
				query = query + normal * dot(normal, center - q);
			}
			if (order > 1)
			{
				float3 plane_normal;
				plane_normal.x = normal.x;
				plane_normal.y = normal.y;
				plane_normal.z = normal.z;
				auto v_axis = Eigen33::unitOrthogonal(plane_normal);
				auto u_axis = cross(plane_normal, v_axis);
				float search_radius = params.queries[idx.x].radius;

				auto num_neighbors = payload.foundNeighbors;
				if (order > 1)
				{

					if (num_neighbors >= nr_coeff)
					{
						float weight_vec[maxresult];
						float P[nr_coeff * maxresult];
						float f_vec[maxresult];
						float P_weight_Pt[nr_coeff * nr_coeff] = { 0 };
						float3 de_meaned[maxresult];
						for (std::size_t ni = 0; ni < num_neighbors; ++ni)
						{
							de_meaned[ni] = params.samplePos[payload.optixIndices[ni]] - query;
							weight_vec[ni] = weight_func(dot(de_meaned[ni], de_meaned[ni]), search_radius);//移植一个weight_func()函数
						}
						//遍历邻居，在局部坐标系中转换它们，保存高度和多项式项的值
						for (std::size_t ni = 0; ni < num_neighbors; ++ni)
						{
							//转换坐标
							const float u_coord = dot(de_meaned[ni], u_axis);
							const float v_coord = dot(de_meaned[ni], v_axis);
							f_vec[ni] = dot(de_meaned[ni], plane_normal);
							//计算多项式在当前点的项
							int j = 0;
							float u_pow = 1;
							for (int ui = 0; ui <= order; ++ui)
							{
								float v_pow = 1;
								for (int vi = 0; vi <= order - ui; ++vi, j++)
								{
									P[j * maxresult + ni] = u_pow * v_pow;
									v_pow *= v_coord;
								}
								u_pow *= u_coord;
							}
						}
						//计算系数
						float P_weight[nr_coeff * maxresult];
						asDiagonal(P, weight_vec, P_weight, num_neighbors, nr_coeff);
						MatirxCross(P_weight, P, P_weight_Pt, num_neighbors, nr_coeff);
						float c_vec[nr_coeff] = { 0 };
						MatrixVectorCross(P_weight, f_vec, c_vec, num_neighbors, nr_coeff);
						//使用LLT分解（Cholesky分解）来求解多项式系数 c_vec
						SymmetricMatrix A(P_weight_Pt, nr_coeff);
						bool isllt = llt(A);
						if (isllt) {
							solveInPlace(A, c_vec);
							//simple方法
							query = query + (normal * c_vec[0]);
							float3 proj_normal;
							proj_normal = plane_normal - u_axis * c_vec[order + 1] - v_axis * c_vec[1];
							proj_normal = normalize(proj_normal);
							normal.x = proj_normal.x;
							normal.y = proj_normal.y;
							normal.z = proj_normal.z;
						}
					}
				}
			}
			params.normals[idx.x] = normal;
			params.queries[idx.x].position = make_float3(query.x, query.y, query.z);
		}
		else
			params.normals[idx.x] = make_float3(0.0f);
	}

	extern "C" __global__ void __intersection__knnSearch_smooth(void)
	{
		payload_t& payload = *getPayload<payload_t>();

		float3& sample = params.samplePos[optixGetPrimitiveIndex()];

		const float3 diff = sample - optixGetWorldRayOrigin();
		const float_t t = dot(diff, diff);

		if (t < payload.query.radius * payload.query.radius)
		{
			if (t < payload.maxDistElemf)
			{
				if (payload.foundNeighbors < params.knn)
				{
					const uint32_t idxToSave = payload.foundNeighbors;
					payload.optixIndices[idxToSave] = optixGetPrimitiveIndex();
					payload.optixDists[idxToSave] = t;

					if (payload.foundNeighbors == params.knn - 1)
					{
						findLargestDist(payload);
					}

					++payload.foundNeighbors;
				}
				else
				{
					payload.optixIndices[payload.maxDistElemi] = optixGetPrimitiveIndex();
					payload.optixDists[payload.maxDistElemi] = t;
					findLargestDist(payload);
				}
			}
		}
	}

	extern "C" __global__ void __raygen__radSearch_normal(void)
	{
		const uint3& idx = optixGetLaunchIndex();
		query_t& query = params.queries[idx.x];
		payload_t payload;
		payload.query = query;
		payload.foundNeighbors = 0;

		uint32_t u0, u1;
		packPointer(&payload, u0, u1);

		optixTrace(params.gasHandle,
			query.position, make_float3(1.e-16f),
			0.f, 1.e-16f, 0.f,
			OptixVisibilityMask(1),
			OPTIX_RAY_FLAG_DISABLE_ANYHIT |
			OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
			0, 4, 0,
			u0, u1);

		if (payload.foundNeighbors >= 3)
		{
			float3 center = make_float3(0.0, 0.0, 0.0);
			float3 dpoint = make_float3(0.f, 0.f, 0.f);
			for (int i = 0; i < payload.foundNeighbors; ++i)
				dpoint = dpoint + make_float3(params.samplePos[payload.optixIndices[i]].x, params.samplePos[payload.optixIndices[i]].y, params.samplePos[payload.optixIndices[i]].z);
			center = dpoint / payload.foundNeighbors;

			float dx2 = 0, dxy = 0, dxz = 0, dy2 = 0, dyz = 0, dz2 = 0;
			float3  d;
			for (int i = 0; i < payload.foundNeighbors; i++) {
				dpoint = params.samplePos[payload.optixIndices[i]];
				d = dpoint - center;
				dx2 += d.x * d.x;  dxy += d.x * d.y;
				dxz += d.x * d.z;  dy2 += d.y * d.y;
				dyz += d.y * d.z;  dz2 += d.z * d.z;
			}
			float cov[6];
			cov[0] = dx2;
			cov[1] = dxy;
			cov[2] = dxz;
			cov[3] = dy2;
			cov[4] = dyz;
			cov[5] = dz2;

			Eigen33 eigen33(&cov[0]);

			float cov_buffer[3][9];

			Eigen33::Mat33& tmp = (Eigen33::Mat33&)cov_buffer[0][0];
			Eigen33::Mat33& vec_tmp = (Eigen33::Mat33&)cov_buffer[1][0];
			Eigen33::Mat33& evecs = (Eigen33::Mat33&)cov_buffer[2][0];
			float3 evals;

			eigen33.compute(tmp, vec_tmp, evecs, evals);

			params.normals[idx.x].x = evecs[0].x;
			params.normals[idx.x].y = evecs[0].y;
			params.normals[idx.x].z = evecs[0].z;
		}
		else
			params.normals[idx.x] = make_float3(0.0f);
	}

	extern "C" __global__ void __intersection__radSearch_normal(void)
	{
		payload_t& payload = *getPayload<payload_t>();

		float3& sample = params.samplePos[optixGetPrimitiveIndex()];

		const float3 diff = sample - optixGetWorldRayOrigin();
		const float_t t = dot(diff, diff);
		if (t < payload.query.radius * payload.query.radius && payload.foundNeighbors < maxresult)
		{
			payload.optixIndices[payload.foundNeighbors] = optixGetPrimitiveIndex();
			++payload.foundNeighbors;
		}
	}

	extern "C" __global__ void __raygen__knnSearch_normal(void)
	{
		const uint3& idx = optixGetLaunchIndex();
		query_t& query = params.queries[idx.x];
		payload_t payload;
		payload.query = query;
		payload.maxDistElemi = 0;
		payload.maxDistElemf = query.radius + 1.f;
		payload.foundNeighbors = 0;

		uint32_t u0, u1;
		packPointer(&payload, u0, u1);

		optixTrace(params.gasHandle,
			query.position, make_float3(1.e-16f),
			0.f, 1.e-16f, 0.f,
			OptixVisibilityMask(1),
			OPTIX_RAY_FLAG_DISABLE_ANYHIT |
			OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
			0, 4, 0,
			u0, u1);

		if (payload.foundNeighbors >= 3)
		{
			float3 center = make_float3(0.0, 0.0, 0.0);
			float3 dpoint = make_float3(0.f, 0.f, 0.f);
			for (int i = 0; i < payload.foundNeighbors; ++i)
				dpoint = dpoint + make_float3(params.samplePos[payload.optixIndices[i]].x, params.samplePos[payload.optixIndices[i]].y, params.samplePos[payload.optixIndices[i]].z);
			center = dpoint / payload.foundNeighbors;

			float dx2 = 0, dxy = 0, dxz = 0, dy2 = 0, dyz = 0, dz2 = 0;
			float3  d;
			for (int i = 0; i < payload.foundNeighbors; i++) {
				dpoint = params.samplePos[payload.optixIndices[i]];
				d = dpoint - center;
				dx2 += d.x * d.x;  dxy += d.x * d.y;
				dxz += d.x * d.z;  dy2 += d.y * d.y;
				dyz += d.y * d.z;  dz2 += d.z * d.z;
			}
			float cov[6];
			cov[0] = dx2;
			cov[1] = dxy;
			cov[2] = dxz;
			cov[3] = dy2;
			cov[4] = dyz;
			cov[5] = dz2;

			Eigen33 eigen33(&cov[0]);

			float cov_buffer[3][9];

			Eigen33::Mat33& tmp = (Eigen33::Mat33&)cov_buffer[0][0];
			Eigen33::Mat33& vec_tmp = (Eigen33::Mat33&)cov_buffer[1][0];
			Eigen33::Mat33& evecs = (Eigen33::Mat33&)cov_buffer[2][0];
			float3 evals;

			eigen33.compute(tmp, vec_tmp, evecs, evals);

			params.normals[idx.x].x = evecs[0].x;
			params.normals[idx.x].y = evecs[0].y;
			params.normals[idx.x].z = evecs[0].z;
		}
		else
			params.normals[idx.x] = make_float3(0.0f);
	}

	extern "C" __global__ void __intersection__knnSearch_normal(void)
	{
		payload_t& payload = *getPayload<payload_t>();

		float3& sample = params.samplePos[optixGetPrimitiveIndex()];

		const float3 diff = sample - optixGetWorldRayOrigin();
		const float_t t = dot(diff, diff);

		if (t < payload.query.radius * payload.query.radius)
		{
			if (t < payload.maxDistElemf)
			{
				if (payload.foundNeighbors < params.knn)
				{
					const uint32_t idxToSave = payload.foundNeighbors;
					payload.optixIndices[idxToSave] = optixGetPrimitiveIndex();
					payload.optixDists[idxToSave] = t;

					if (payload.foundNeighbors == params.knn - 1)
					{
						findLargestDist(payload);
					}

					++payload.foundNeighbors;
				}
				else
				{
					payload.optixIndices[payload.maxDistElemi] = optixGetPrimitiveIndex();
					payload.optixDists[payload.maxDistElemi] = t;
					findLargestDist(payload);
				}
			}
		}
	}
}