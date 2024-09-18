#pragma once

#include <vector>
#include "cuda_buffer.hpp"
#include "cuda_types.hpp"

namespace bvh_radSearch
{
    class bvh_index final
    {
    public:

        bvh_index(const OptixDeviceContext&);
        ~bvh_index(void);

        bvh_index(const bvh_index&) = default;
        bvh_index(bvh_index&&) = default;

        bvh_index& operator=(const bvh_index&) = default;
        bvh_index& operator=(bvh_index&&) = default;

        void destroy(void);
        void init(void);

        float_t build(std::vector<OptixAabb>& pSamples, float_t* pGaS_size = nullptr);

        float_t radius_search(std::vector<query_t>& pQueries, std::vector<int32_t>& pIndices, std::vector<float_t>& pDists, uint32_t* pMaxCapacity);

        float_t radius_search_count(std::vector<query_t>& pQueries, statistics_t& pStats);

        float_t truncated_knn(std::vector<query_t>& pQueries, uint32_t knn, std::vector<int32_t>& pIndices, std::vector<float_t>& pDists);

        float_t radius_search_normals(std::vector<query_t>& pQueries, std::vector<float3>& pNormals);

        float_t radius_search_smooth(std::vector<query_t>& pQueries, std::vector<float3>& pNormals);

        float_t knn_search_smooth(std::vector<query_t>& pQueries, uint32_t knn, std::vector<float3>& pNormals);

        float_t knn_search_normals(std::vector<query_t>& pQueries, uint32_t knn, std::vector<float3>& pNormals);


    protected:

        // Empty

    private:

        enum MODULE_E : uint32_t
        {
            COUNT,
            RAD_SEARCH,
            RAD_SEARCH_SMOOTH,
            KNN_SEARCH_SMOOTH,
            RAD_SEARCH_NORMAL,
            KNN_SEARCH_NORMAL,
            SIZE
        };

        struct shader_t
        {
            OptixProgramGroup raygen;
            OptixProgramGroup hit;
            OptixProgramGroup miss;

            cudaBuffer<RayGenSbtRecord> gen_rec;
            cudaBuffer<HitGroupSbtRecord> hit_rec;
            cudaBuffer<MissSbtRecord> miss_rec;

            shader_t(void) noexcept :
                raygen(nullptr),
                hit(nullptr),
                miss(nullptr)
            { /* Empty */
            }

            ~shader_t(void)
            {
                OPTIX_CHECK(optixProgramGroupDestroy(raygen));
                OPTIX_CHECK(optixProgramGroupDestroy(hit));
                OPTIX_CHECK(optixProgramGroupDestroy(miss));

                gen_rec.destroy();
                hit_rec.destroy();
                miss_rec.destroy();
            }
        };

        const OptixDeviceContext& mOptixContext;

        CUdeviceptr mGasBuffer;

        cudaBuffer<Params> mDeviceParams;

        cudaBuffer<uint32_t> mTotalCount;
        cudaBuffer<uint32_t> mMinCount;
        cudaBuffer<uint32_t> mMaxCount;

        cudaBuffer<float3> mSamples;
        cudaBuffer<float3> mNormals;

        Params mParams;

        CUstream mCudaStream;

        OptixModule mModule;

        std::vector<shader_t> mShaders;

        OptixPipeline mPipeline;

        OptixShaderBindingTable mSBT;
    };
}