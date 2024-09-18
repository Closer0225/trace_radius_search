// Fast Radius Search Exploiting Ray Tracing Frameworks
// Authors: I. Evangelou, G. Papaioannou, K. Vardis, A. A. Vasilakis

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#include <iomanip>
#include <iostream>
#include <string>

#include <optix.h>
#include <optix_types.h>
#include <cuda_runtime.h>
#include <optix_stubs.h>
#include <algorithm>
#include "radius_search.hpp"
#include "optix_types.h"
#include "cuda_buffer.hpp"
#include "utils.hpp"

#include <curand.h>

using namespace bvh_radSearch;

void openPointCloud(const std::string& fileName, std::vector<float3> &points)
{
    int i = 0;
    std::string errorInfo;
    FILE* fp = fopen(fileName.c_str(), "r");
    if (fp)
    {
        try
        {
            float3 point;
            float3 normal;
            char line[1024];

            while (fgets(line, 1023, fp))
            {
                if (sscanf(line, "%f%f%f%f%f%f", &point.x, &point.y, &point.z, &normal.x, &normal.y, &normal.z) == 6)
                {
                    points.push_back(point);
                }
                else if (sscanf(line, "%f%f%f", &point.x, &point.y, &point.z) == 3)
                    points.push_back(make_float3(point.x, point.y, point.z));
                i++;
            }
        }
        catch (std::exception& error)
        {
            errorInfo = error.what();
        }
        fclose(fp);
    }
    else
        errorInfo = "打开点云文件失败！\n";
}

void test_point_cloud(const OptixDeviceContext& pCntx,
    const std::string & pFileName,
    float_t pRadii, int32_t knn = -1)
{
    std::vector<float3> points{};
    openPointCloud(pFileName, points);
    std::vector<float3> queryPoints{};
    queryPoints.reserve(points.size());
    for (const auto& point : points)
        queryPoints.push_back(make_float3(point.x, point.y, point.z));
    std::vector<query_t> queries{};
    std::vector<OptixAabb> aabbs{};
    std::vector<int32_t> indices{};
    std::vector<float_t> dists{};
    std::vector<float3> outNormals{};
    buildAabbs(points, pRadii, aabbs);

    queries.reserve(queryPoints.size());
    std::for_each(queryPoints.begin(), queryPoints.end(),
        [&](float3& pVal) {
            query_t q{};
            q.position = pVal;
            q.radius = pRadii;
            q.count = 0;
            queries.push_back(q); });

    auto index = bvh_radSearch::bvh_index(pCntx);
    index.init();

    float_t gas_size = 0.f;
    float_t ms_build_timer = index.build(aabbs, &gas_size);
    float_t ms_search_timer = 0.f;

    uint32_t maxCapacity = 0;


#ifdef _DEBUG

    std::vector<query_t> dqueries = queries;
    std::vector<int32_t> dindices{};
    std::vector<float_t> ddists{};
    uint32_t dmaxCapacity = 0;

    if (knn == -1)
    {

        ms_search_timer = index.radius_search(queries, indices, dists, &maxCapacity);
    }
    else
    {
        ms_search_timer = index.truncated_knn(queries, knn, indices, dists);
    }

    assert(dmaxCapacity == maxCapacity);
    assert_rad_search(dqueries, dindices, queries, indices, maxCapacity);
#else

    if (knn == -1)
    {
        ms_search_timer = index.radius_search_normals(queries, outNormals);
    }
    else
    {
        ms_search_timer = index.truncated_knn(queries, knn, indices, dists);
    }
#endif

    index.destroy();
    std::cout << "Log: " << pFileName << std::endl;
    log_timer(gas_size, ms_build_timer, ms_search_timer);
    std::cout << std::endl;
}

int main(int argc, char* argv[])
{
    OptixDeviceContext context = nullptr;

    try
    {
        CUDA_CHECK(cudaFree(0));

        CUcontext cuCtx = 0;
        OPTIX_CHECK(optixInit());
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &context_log_cb;
        options.logCallbackLevel = 4;
        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
        test_point_cloud(context, "../data/happy_vrip.asc", 1.0);

        system("PAUSE");
        OPTIX_CHECK(optixDeviceContextDestroy(context));
    }
    catch (std::exception& e)
    {
        std::cerr << "Caught exception: " << e.what() << "\n";
    }

    return 0;
}
