#include "EventKernels.cuh"

class moduleRangesKernel
{
public:
    ALPAKA_NO_HOST_ACC_WARNING
    template<
        typename TAcc,
        typename TIdx>
    ALPAKA_FN_ACC auto operator()(
        TAcc const & acc,
        struct SDL::modules *modulesInGPU,
        struct SDL::hits *hitsInGPU,
        TIdx const & nLowerModules) const
    -> void
    {

        TIdx const gridThreadIdx(alpaka::idx::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u]);
        TIdx const threadElemExtent(alpaka::workdiv::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u]);
        TIdx const threadFirstElemIdx(gridThreadIdx * threadElemExtent);

        if(threadFirstElemIdx < nLowerModules)
        {
            // Calculate the number of elements to compute in this thread.
            // The result is uniform for all but the last thread.
            TIdx const threadLastElemIdx(threadFirstElemIdx+threadElemExtent);
            TIdx const threadLastElemIdxClipped((nLowerModules > threadLastElemIdx) ? threadLastElemIdx : nLowerModules);

            for(TIdx i(threadFirstElemIdx); i<threadLastElemIdxClipped; ++i)
            {
                uint16_t upperIndex = modulesInGPU->partnerModuleIndices[i];
                if (hitsInGPU->hitRanges[i * 2] != -1 && hitsInGPU->hitRanges[upperIndex * 2] != -1)
                {
                    hitsInGPU->hitRangesLower[i] =  hitsInGPU->hitRanges[i * 2]; 
                    hitsInGPU->hitRangesUpper[i] =  hitsInGPU->hitRanges[upperIndex * 2];
                    hitsInGPU->hitRangesnLower[i] = hitsInGPU->hitRanges[i * 2 + 1] - hitsInGPU->hitRanges[i * 2] + 1;
                    hitsInGPU->hitRangesnUpper[i] = hitsInGPU->hitRanges[upperIndex * 2 + 1] - hitsInGPU->hitRanges[upperIndex * 2] + 1;
                }
            }
        }
    }
};

void SDL::EventKernels::addHitToEvent(struct SDL::modules *modulesInGPU, struct SDL::hits *hitsInGPU, uint16_t const & nLowerModules)
{
    // Define the work division
    Idx const elementsPerThread(3u);
    Idx const numElements(nLowerModules);
    alpaka::vec::Vec<Dim, Idx> const extent(numElements);

    // Let alpaka calculate good block and grid sizes given our full problem extent
    alpaka::workdiv::WorkDivMembers<Dim, Idx> const workDiv(
        alpaka::workdiv::getValidWorkDiv<Acc>(
            devAcc,
            extent,
            elementsPerThread,
            false,
            alpaka::workdiv::GridBlockExtentSubDivRestrictions::Unrestricted));

    // Instantiate the kernel function object
    moduleRangesKernel kernel;

    // Create the kernel execution task.
    auto const taskKernel(alpaka::kernel::createTaskKernel<Acc>(
        workDiv,
        kernel,
        modulesInGPU,
        hitsInGPU,
        numElements));

    // Enqueue the kernel execution task
    alpaka::queue::enqueue(queue, taskKernel);
}
