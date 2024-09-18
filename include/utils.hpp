// Fast Radius Search Exploiting Ray Tracing Frameworks
// Authors: I. Evangelou, G. Papaioannou, K. Vardis, A. A. Vasilakis
#pragma once

#include <curand.h>

#include <iostream>
#include <filesystem>
#include <unordered_map>
#include <set>

#include "cuda_buffer.hpp"

using namespace bvh_radSearch;

static void context_log_cb(
    unsigned int level, const char* tag,
    const char* message, void* /*cbdata */)
{
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: "
        << message << "\n";
}

static void log_timer(float_t pIndex_size, float_t pBuild, float_t pSearch)
{
    std::cout << "ADS (" << pIndex_size << "MB" << ")" << " build time: " << pBuild << "ms" << std::endl;
    std::cout << "Search time: " << pSearch << "ms" << std::endl;
}

static void log_timer(float_t pIndex_size, float_t pBuild, float_t pSearch, statistics_t pStats)
{
    std::cout << "ADS (" << pIndex_size << "MB" << ")" << " build time: " << pBuild << "ms" << std::endl;
    std::cout << "Search time: " << pSearch << "ms" << std::endl;

    std::cout << "Total gather: " << pStats.totalGather << std::endl;
    std::cout << "Max gather: " << pStats.maxGather << std::endl;
    std::cout << "Min gather: " << pStats.minGather << std::endl;
    std::cout << "Avg gather: " << pStats.avgGather << std::endl;
}

static void splitPointCloud(
    std::vector<float3>& pSamples,
    std::vector<float3>& pQueries,
    float_t pFraction) noexcept
{
    if (pFraction == 1.f)
    {
        pSamples = pSamples;
        pQueries = pSamples;
        return;
    }

    uint32_t numQueries = pFraction > 1.f ?
        uint32_t(pFraction) : pSamples.size() * pFraction;

    cudaBuffer<float_t> randIndices;
    randIndices.alloc(numQueries);

    curandGenerator_t rng;
    curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(rng, 91);

    curandGenerateUniform(rng, randIndices.getPtr(), randIndices.mArraySize);

    std::vector<float3> queries;
    std::vector<float3> newSamples;
    std::vector<float_t> hostIndices = randIndices.getData();
    std::vector<bool> mask(pSamples.size(), false);

    std::set<uint32_t> cache;

    for (uint32_t i = 0; i < numQueries; ++i)
    {
        const size_t index = size_t(hostIndices[i] * pSamples.size());

        if (cache.find(index) == cache.cend())
        {
            queries.push_back(pSamples[index]);
            mask[index] = true;
            cache.insert(index);
        }
    }

    for (uint32_t i = 0; i < mask.size(); ++i)
    {
        if (!mask[i]) newSamples.push_back(pSamples[i]);
    }

    pSamples = std::move(newSamples);
    pQueries = std::move(queries);
}

static void buildAabbs(
    const std::vector<float3>& pSamples,
    float_t pRadii,
    std::vector<OptixAabb>& pAabbs)
{
    pAabbs.resize(pSamples.size());

    for (size_t i = 0; i < pSamples.size(); ++i)
    {
        float3 tmp = pSamples[i];

        OptixAabb& aabb = pAabbs[i];
        aabb.minX = tmp.x - pRadii;
        aabb.minY = tmp.y - pRadii;
        aabb.minZ = tmp.z - pRadii;

        aabb.maxX = tmp.x + pRadii;
        aabb.maxY = tmp.y + pRadii;
        aabb.maxZ = tmp.z + pRadii;
    }
}

static void assert_rad_search(
    const std::vector<query_t>& pRefQueries,
    const std::vector<query_t>& pQueries)
{
    for (size_t i = 0; i < pRefQueries.size(); ++i)
    {
        const query_t& refq = pRefQueries[i];
        const query_t& q = pQueries[i];
        assert(refq.count == q.count);
    }
}

static void assert_rad_search(
    const std::vector<query_t>& pRefQueries,
    const std::vector<int32_t>& pRefIndices,
    const std::vector<query_t>& pQueries,
    const std::vector<int32_t>& pIndices,
    const uint32_t pMaxCapacity)
{
    for (size_t i = 0; i < pRefQueries.size(); ++i)
    {
        const query_t& refq = pRefQueries[i];
        const query_t& q = pQueries[i];
        assert(refq.count == q.count);

        for (uint32_t k = 0; k < pMaxCapacity; ++k)
        {
            assert(
                pRefIndices[(pMaxCapacity * i) + k] ==
                pIndices[(pMaxCapacity * i) + k]);
        }
    }
}