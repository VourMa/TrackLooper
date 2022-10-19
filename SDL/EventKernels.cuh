#ifndef EventKernels_cuh
#define EventKernels_cuh

#include "Module.cuh"
#include "Hit.cuh"

#include <alpaka/alpaka.hpp>

using Dim = alpaka::dim::DimInt<1u>;
using Idx = std::size_t;
using Acc = alpaka::acc::AccGpuCudaRt<Dim, Idx>;
using DevAcc = alpaka::dev::Dev<Acc>;
using QueueProperty = alpaka::queue::NonBlocking;
using QueueAcc = alpaka::queue::Queue<Acc, QueueProperty>;

namespace SDL
{
    class EventKernels
    {
    private:
        DevAcc devAcc;
        QueueAcc queue;

    public:
        EventKernels()
            : devAcc(alpaka::pltf::getDevByIdx<Acc>(0u)),
              queue(devAcc) {}
        //~EventKernels();

        void addHitToEvent(struct SDL::modules* modulesInGPU, struct SDL::hits* hitsInGPU, uint16_t const& nLowerModules);
    };
}

#endif
