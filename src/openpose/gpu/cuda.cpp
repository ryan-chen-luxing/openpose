#ifdef USE_CUDA
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <openpose/utilities/fastMath.hpp>
#endif
#include <openpose/gpu/cuda.hpp>

#include <sstream>

namespace op
{
    #ifdef USE_CUDA
        const dim3 THREADS_PER_BLOCK_TINY{32, 32, 1};
        const dim3 THREADS_PER_BLOCK_SMALL{64, 64, 1};
        const dim3 THREADS_PER_BLOCK_MEDIUM{128, 128, 1};
        const dim3 THREADS_PER_BLOCK_BIG{256, 256, 1};
        const dim3 THREADS_PER_BLOCK_HUGE{512, 512, 1};
    #endif

    void cudaCheck(const int line, const std::string& function, const std::string& file)
    {
        try
        {
            #ifdef USE_CUDA
                const auto errorCode = cudaPeekAtLastError();
                if(errorCode != cudaSuccess)
                    error("Cuda check failed (" + std::to_string(errorCode) + " vs. " + std::to_string(cudaSuccess) + "): "
                          + cudaGetErrorString(errorCode), line, function, file);
            #else
                UNUSED(line);
                UNUSED(function);
                UNUSED(file);
                error("OpenPose must be compiled with the `USE_CUDA` macro definition in order to use this"
                      " functionality.", __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    int getCudaGpuNumber()
    {
        try
        {
            #ifdef USE_CUDA
                int gpuNumber;
                cudaGetDeviceCount(&gpuNumber);
                cudaCheck(__LINE__, __FUNCTION__, __FILE__);

                if (true)
                {
                    for (auto i = 0; i < gpuNumber; ++i)
                    {
                        cudaDeviceProp prop;
                        cudaGetDeviceProperties(&prop, i);

                        std::stringstream ss;

                        ss << "cuda device " << i << ", name: " << prop.name \
                            << ", totalGlobalMem: " << prop.totalGlobalMem \
                            << ", major: " << prop.major \
                            << ", minor: " << prop.minor << std::endl;

                        log(ss.str(), Priority::High);
                    }
                }

                return gpuNumber;
            #else
                error("OpenPose must be compiled with the `USE_CUDA` macro definition in order to use this"
                      " functionality.", __LINE__, __FUNCTION__, __FILE__);
                return -1;
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return -1;
        }
    }

    void getNumberCudaThreadsAndBlocks(dim3& numberCudaThreads, dim3& numberCudaBlocks, const Point<int>& frameSize)
    {
        try
        {
            #ifdef USE_CUDA
                // numberCudaThreads
                // Image <= 480p    --> THREADS_PER_BLOCK_TINY
                // Image <= 720p    --> THREADS_PER_BLOCK_SMALL
                // Image <= 1080p   --> THREADS_PER_BLOCK_MEDIUM
                // Image <= 4k      --> THREADS_PER_BLOCK_BIG
                // Image >  4K      --> THREADS_PER_BLOCK_HUGE
                const auto maxValue = fastMax(frameSize.x, frameSize.y);
                // > 4K
                if (maxValue > 3840)
                    numberCudaThreads = THREADS_PER_BLOCK_HUGE;
                // 4K
                else if (maxValue > 1980)
                    numberCudaThreads = THREADS_PER_BLOCK_BIG;
                // FullHD
                else if (maxValue > 1280)
                    numberCudaThreads = THREADS_PER_BLOCK_MEDIUM;
                // HD
                else if (maxValue > 640)
                    numberCudaThreads = THREADS_PER_BLOCK_SMALL;
                // VGA
                else
                    numberCudaThreads = THREADS_PER_BLOCK_TINY;
                // numberCudaBlocks
                numberCudaBlocks = dim3{getNumberCudaBlocks((unsigned int)frameSize.x, numberCudaThreads.x),
                                        getNumberCudaBlocks((unsigned int)frameSize.y, numberCudaThreads.y),
                                        numberCudaThreads.z};
            #else
                UNUSED(numberCudaThreads);
                UNUSED(numberCudaBlocks);
                UNUSED(frameSize);
                error("OpenPose must be compiled with the `USE_CUDA` macro definition in order to use this"
                      " functionality.", __LINE__, __FUNCTION__, __FILE__);
            #endif
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }
}
