#include "TrackCandidate.cuh"

#include "allocate.h"


void SDL::createTrackCandidatesInUnifiedMemory(struct trackCandidates& trackCandidatesInGPU, unsigned int maxTrackCandidates)
{
#ifdef CACHE_ALLOC
    cudaStream_t stream=0;
    trackCandidatesInGPU.trackCandidateType = (short*)cms::cuda::allocate_managed(maxTrackCandidates * sizeof(short),stream);
    cudaMallocManaged(&trackCandidatesInGPU.objectIndices, 2 * maxTrackCandidates * sizeof(unsigned int));
    trackCandidatesInGPU.nTrackCandidates= (unsigned int*)cms::cuda::allocate_managed( sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatesT4T4= (unsigned int*)cms::cuda::allocate_managed( sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatesT4T3= (unsigned int*)cms::cuda::allocate_managed( sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatesT3T4= (unsigned int*)cms::cuda::allocate_managed( sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatespT2= (unsigned int*)cms::cuda::allocate_managed(sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatespT3= (unsigned int*)cms::cuda::allocate_managed(sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatesT5= (unsigned int*)cms::cuda::allocate_managed( sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatespT5= (unsigned int*)cms::cuda::allocate_managed( sizeof(unsigned int),stream);
    trackCandidatesInGPU.logicalLayers = (unsigned int*)cms::cuda::allocate_managed(7 * maxTrackCandidates * sizeof(unsigned int), stream);
    trackCandidatesInGPU.lowerModuleIndices = (unsigned int*)cms::cuda::allocate_managed(7 * maxTrackCandidates * sizeof(unsigned int), stream);
    trackCandidatesInGPU.hitIndices = (unsigned int*)cms::cuda::allocate_managed(14 * maxTrackCandidates * sizeof(unsigned int), stream);

    trackCandidatesInGPU.centerX = (float*)cms::cuda::allocate_managed(maxTrackCandidates * sizeof(float), stream);
    trackCandidatesInGPU.centerY = (float*)cms::cuda::allocate_managed(maxTrackCandidates * sizeof(float), stream);
    trackCandidatesInGPU.radius = (float*)cms::cuda::allocate_managed(maxTrackCandidates * sizeof(float), stream);

    trackCandidatesInGPU.partOfExtension = (bool*)cms::cuda::allocate_managed(maxTrackCandidates * sizeof(bool), stream);
    trackCandidatesInGPU.nTrackCandidatespLS= (unsigned int*)cms::cuda::allocate_managed( sizeof(unsigned int),stream);

#else
    cudaMallocManaged(&trackCandidatesInGPU.trackCandidateType, maxTrackCandidates * sizeof(short));
    cudaMallocManaged(&trackCandidatesInGPU.objectIndices, 2 * maxTrackCandidates * sizeof(unsigned int));
    cudaMallocManaged(&trackCandidatesInGPU.nTrackCandidates, sizeof(unsigned int));
    cudaMallocManaged(&trackCandidatesInGPU.nTrackCandidatesT4T4, sizeof(unsigned int));
    cudaMallocManaged(&trackCandidatesInGPU.nTrackCandidatesT4T3, sizeof(unsigned int));
    cudaMallocManaged(&trackCandidatesInGPU.nTrackCandidatesT3T4, sizeof(unsigned int));
    cudaMallocManaged(&trackCandidatesInGPU.nTrackCandidatespT2, sizeof(unsigned int));
    cudaMallocManaged(&trackCandidatesInGPU.nTrackCandidatespT3, sizeof(unsigned int));
    cudaMallocManaged(&trackCandidatesInGPU.nTrackCandidatesT5, sizeof(unsigned int));
    cudaMallocManaged(&trackCandidatesInGPU.nTrackCandidatespT5, sizeof(unsigned int));

    cudaMallocManaged(&trackCandidatesInGPU.logicalLayers, maxTrackCandidates * 7 * sizeof(unsigned int));
    cudaMallocManaged(&trackCandidatesInGPU.lowerModuleIndices, maxTrackCandidates * 7 * sizeof(unsigned int));
    cudaMallocManaged(&trackCandidatesInGPU.hitIndices, maxTrackCandidates * 14 * sizeof(unsigned int));

    cudaMallocManaged(&trackCandidatesInGPU.centerX, maxTrackCandidates * sizeof(float));
    cudaMallocManaged(&trackCandidatesInGPU.centerY, maxTrackCandidates * sizeof(float));
    cudaMallocManaged(&trackCandidatesInGPU.radius, maxTrackCandidates * sizeof(float));

    cudaMallocManaged(&trackCandidatesInGPU.nTrackCandidatespLS, sizeof(unsigned int));
    cudaMallocManaged(&trackCandidatesInGPU.partOfExtension, maxTrackCandidates*sizeof(bool));
#endif

    cudaMemset(trackCandidatesInGPU.nTrackCandidates,0, sizeof(unsigned int));
    cudaMemset(trackCandidatesInGPU.nTrackCandidatesT4T4,0, sizeof(unsigned int));
    cudaMemset(trackCandidatesInGPU.nTrackCandidatesT4T3,0, sizeof(unsigned int));
    cudaMemset(trackCandidatesInGPU.nTrackCandidatesT3T4,0, sizeof(unsigned int));
    cudaMemset(trackCandidatesInGPU.nTrackCandidatesT5,0, sizeof(unsigned int));
    cudaMemset(trackCandidatesInGPU.nTrackCandidatespT2,0, sizeof(unsigned int));
    cudaMemset(trackCandidatesInGPU.nTrackCandidatespT3,0, sizeof(unsigned int));
    cudaMemset(trackCandidatesInGPU.nTrackCandidatespT5,0, sizeof(unsigned int));
    cudaMemset(trackCandidatesInGPU.nTrackCandidatespLS,0, sizeof(unsigned int));
    cudaMemset(trackCandidatesInGPU.partOfExtension, false, maxTrackCandidates * sizeof(bool));
}
void SDL::createTrackCandidatesInExplicitMemory(struct trackCandidates& trackCandidatesInGPU, unsigned int maxTrackCandidates)
{
#ifdef CACHE_ALLOC
    cudaStream_t stream=0;
    int dev;
    cudaGetDevice(&dev);
    //TODO 
    cudaMalloc(&trackCandidatesInGPU.trackCandidateType, maxTrackCandidates * sizeof(short));
    cudaMalloc(&trackCandidatesInGPU.objectIndices, 2 * maxTrackCandidates * sizeof(unsigned int));
    trackCandidatesInGPU.nTrackCandidates= (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatesT4T4= (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatesT4T3= (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatesT3T4= (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatespT2= (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatespT3= (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatesT5= (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatespT5= (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int),stream);
    trackCandidatesInGPU.nTrackCandidatespLS= (unsigned int*)cms::cuda::allocate_device(dev, sizeof(unsigned int),stream);
    trackCandidatesInGPU.partOfExtension = (bool*)cms::cuda::allocate_device(dev, maxTrackCandidates * sizeof(bool), stream);

    trackCandidatesInGPU.logicalLayers = (unsigned int*)cms::cuda::allocate_device(dev, 7 * maxTrackCandidates * sizeof(unsigned int), stream);
    trackCandidatesInGPU.lowerModuleIndices = (unsigned int*)cms::cuda::allocate_device(dev, 7 * maxTrackCandidates * sizeof(unsigned int), stream);
    trackCandidatesInGPU.hitIndices = (unsigned int*)cms::cuda::allocate_device(dev, 14 * maxTrackCandidates * sizeof(unsigned int), stream);

    trackCandidatesInGPU.centerX = (float*)cms::cuda::allocate_device(dev, maxTrackCandidates * sizeof(float), stream);
    trackCandidatesInGPU.centerY = (float*)cms::cuda::allocate_device(dev, maxTrackCandidates * sizeof(float), stream);
    trackCandidatesInGPU.radius = (float*)cms::cuda::allocate_device(dev, maxTrackCandidates * sizeof(float), stream);

#else
    cudaMalloc(&trackCandidatesInGPU.trackCandidateType, maxTrackCandidates * sizeof(short));
    cudaMalloc(&trackCandidatesInGPU.objectIndices, 2 * maxTrackCandidates * sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.nTrackCandidates, sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.nTrackCandidatesT4T4, sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.nTrackCandidatesT4T3, sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.nTrackCandidatesT3T4, sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.nTrackCandidatespT2, sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.nTrackCandidatespT3, sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.nTrackCandidatesT5, sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.nTrackCandidatespT5, sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.nTrackCandidatespLS, sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.partOfExtension, maxTrackCandidates * sizeof(bool));

    cudaMalloc(&trackCandidatesInGPU.logicalLayers, 7 * maxTrackCandidates * sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.lowerModuleIndices, 7 * maxTrackCandidates * sizeof(unsigned int));
    cudaMalloc(&trackCandidatesInGPU.hitIndices, 14 * maxTrackCandidates * sizeof(unsigned int));

    cudaMalloc(&trackCandidatesInGPU.centerX, maxTrackCandidates * sizeof(float));
    cudaMalloc(&trackCandidatesInGPU.centerY, maxTrackCandidates * sizeof(float));
    cudaMalloc(&trackCandidatesInGPU.radius, maxTrackCandidates * sizeof(float));
#endif
    cudaMemset(trackCandidatesInGPU.nTrackCandidates,0, sizeof(unsigned int));
    cudaMemset(trackCandidatesInGPU.nTrackCandidatesT4T4,0, sizeof(unsigned int));
    cudaMemset(trackCandidatesInGPU.nTrackCandidatesT4T3,0, sizeof(unsigned int));
    cudaMemset(trackCandidatesInGPU.nTrackCandidatesT3T4,0, sizeof(unsigned int));
    cudaMemset(trackCandidatesInGPU.nTrackCandidatesT5,0, sizeof(unsigned int));
    cudaMemset(trackCandidatesInGPU.nTrackCandidatespT2,0, sizeof(unsigned int));
    cudaMemset(trackCandidatesInGPU.nTrackCandidatespT3,0, sizeof(unsigned int));
    cudaMemset(trackCandidatesInGPU.nTrackCandidatespT5,0, sizeof(unsigned int));
    cudaMemset(trackCandidatesInGPU.nTrackCandidatespLS,0, sizeof(unsigned int));
    cudaMemset(trackCandidatesInGPU.partOfExtension, false, maxTrackCandidates * sizeof(bool));
}

__device__ void SDL::addTrackCandidateToMemory(struct trackCandidates& trackCandidatesInGPU, short trackCandidateType, unsigned int innerTrackletIndex, unsigned int outerTrackletIndex, unsigned int trackCandidateIndex)
{
    trackCandidatesInGPU.trackCandidateType[trackCandidateIndex] = trackCandidateType;
    trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex] = innerTrackletIndex;
    trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex + 1] = outerTrackletIndex;
}

__device__ void SDL::addTrackCandidateToMemory(struct trackCandidates& trackCandidatesInGPU, short trackCandidateType, unsigned int innerTrackletIndex, unsigned int outerTrackletIndex, unsigned int* logicalLayerIndices, unsigned int* lowerModuleIndices, unsigned int* hitIndices, float& centerX, float& centerY, float& radius, unsigned int trackCandidateIndex)
{
    trackCandidatesInGPU.trackCandidateType[trackCandidateIndex] = trackCandidateType;
    trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex] = innerTrackletIndex;
    trackCandidatesInGPU.objectIndices[2 * trackCandidateIndex + 1] = outerTrackletIndex;
    
    size_t limits = trackCandidateType == 7 ? 7 : 5;

    //send the starting pointer to the logicalLayer and hitIndices
    for(size_t i = 0; i < limits; i++)
    {
        trackCandidatesInGPU.logicalLayers[7 * trackCandidateIndex + i] = logicalLayerIndices[i];
        trackCandidatesInGPU.lowerModuleIndices[7 * trackCandidateIndex + i] = lowerModuleIndices[i];
    }
    for(size_t i = 0; i < 2 * limits; i++)
    {
        trackCandidatesInGPU.hitIndices[14 * trackCandidateIndex + i] = hitIndices[i];
    }
    trackCandidatesInGPU.centerX[trackCandidateIndex] = centerX;
    trackCandidatesInGPU.centerY[trackCandidateIndex] = centerY;
    trackCandidatesInGPU.radius[trackCandidateIndex] = radius;
}

__device__ bool SDL::runTrackCandidateDefaultAlgoTwoTracklets(struct tracklets& trackletsInGPU, struct triplets& tripletsInGPU, unsigned int innerTrackletIndex, unsigned int outerTrackletIndex, short& trackCandidateType)
{
    bool pass = true;
    trackCandidateType = 0;
    if(not hasCommonSegment(trackletsInGPU, tripletsInGPU, innerTrackletIndex, outerTrackletIndex, trackCandidateType))
    {
        pass = false;
    }
    return pass;
}

__device__ bool SDL::runTrackCandidateDefaultAlgoTwoTracklets(struct pixelTracklets& pixelTrackletsInGPU, struct tracklets& trackletsInGPU, struct triplets& tripletsInGPU, unsigned int innerTrackletIndex, unsigned int outerTrackletIndex, short& trackCandidateType)
{
    bool pass = true;
    trackCandidateType = 0;
    if(not hasCommonSegment(pixelTrackletsInGPU, trackletsInGPU, tripletsInGPU, innerTrackletIndex, outerTrackletIndex, trackCandidateType))
    {
        pass = false;
    }
    return pass;
}


SDL::trackCandidates::trackCandidates()
{
    trackCandidateType = nullptr;
    objectIndices = nullptr;
    nTrackCandidates = nullptr;
    nTrackCandidatesT4T4 = nullptr;
    nTrackCandidatesT4T3 = nullptr;
    nTrackCandidatesT3T4 = nullptr;
    nTrackCandidatespT2 = nullptr;
    nTrackCandidatesT5 = nullptr;
    nTrackCandidatespT3 = nullptr;
    logicalLayers = nullptr;
    hitIndices = nullptr;
    lowerModuleIndices = nullptr;
    partOfExtension = nullptr;

    centerX = nullptr;
    centerY = nullptr;
    radius = nullptr;
    nTrackCandidatespT5 = nullptr;
    nTrackCandidatespLS = nullptr;
}

SDL::trackCandidates::~trackCandidates()
{
}

void SDL::trackCandidates::freeMemoryCache()
{
#ifdef Explicit_Track
    int dev;
    cudaGetDevice(&dev);
    //FIXME
    cudaFree(trackCandidateType);
    cms::cuda::free_device(dev,nTrackCandidates);
    cms::cuda::free_device(dev,nTrackCandidatesT4T4);
    cms::cuda::free_device(dev,nTrackCandidatesT4T3);
    cms::cuda::free_device(dev,nTrackCandidatesT3T4);
    cms::cuda::free_device(dev,nTrackCandidatespT2);
    cms::cuda::free_device(dev,nTrackCandidatespT3);
    cms::cuda::free_device(dev,nTrackCandidatesT5);
    cms::cuda::free_device(dev,nTrackCandidatespT5);

    cms::cuda::free_device(dev, logicalLayers);
    cms::cuda::free_device(dev, hitIndices);
    cms::cuda::free_device(dev, lowerModuleIndices);

    cms::cuda::free_device(dev,nTrackCandidatespLS);
    cms::cuda::free_device(dev, centerX);
    cms::cuda::free_device(dev, centerY);
    cms::cuda::free_device(dev, radius);
    cms::cuda::free_device(dev, partOfExtension);
#else
    cms::cuda::free_managed(trackCandidateType);
    cms::cuda::free_managed(nTrackCandidates);
    cms::cuda::free_managed(nTrackCandidatesT4T4);
    cms::cuda::free_managed(nTrackCandidatesT4T3);
    cms::cuda::free_managed(nTrackCandidatesT3T4);
    cms::cuda::free_managed(nTrackCandidatespT2);
    cms::cuda::free_managed(nTrackCandidatespT3);
    cms::cuda::free_managed(nTrackCandidatesT5);
    cms::cuda::free_managed(nTrackCandidatespT5);
    cms::cuda::free_managed(logicalLayers);
    cms::cuda::free_managed(hitIndices);
    cms::cuda::free_managed(lowerModuleIndices);
    cms::cuda::free_managed(centerX);
    cms::cuda::free_managed(centerY);
    cms::cuda::free_managed(radius);
    cms::cuda::free_managed(partOfExtension);
    cms::cuda::free_managed(nTrackCandidatespLS);
#endif
    cudaFree(objectIndices);

}
void SDL::trackCandidates::freeMemory()
{
    cudaFree(trackCandidateType);
    cudaFree(objectIndices);
    cudaFree(nTrackCandidates);
    cudaFree(nTrackCandidatesT4T4);
    cudaFree(nTrackCandidatesT4T3);
    cudaFree(nTrackCandidatesT3T4);
    cudaFree(nTrackCandidatespT2);
    cudaFree(nTrackCandidatespT3);
    cudaFree(nTrackCandidatesT5);
    cudaFree(nTrackCandidatespT5);

    cudaFree(logicalLayers);
    cudaFree(hitIndices);
    cudaFree(partOfExtension);

    cudaFree(centerX);
    cudaFree(centerY);
    cudaFree(radius);
    
    cudaFree(lowerModuleIndices);
    cudaFree(nTrackCandidatespLS);
}

__device__ bool SDL::runTrackCandidateDefaultAlgoTrackletToTriplet(struct tracklets& trackletsInGPU, struct triplets& tripletsInGPU, unsigned int innerTrackletIndex, unsigned int outerTripletIndex, short& trackCandidateType)
{
    bool pass = true;
    trackCandidateType = 1;
    if(not hasCommonSegment(trackletsInGPU, tripletsInGPU, innerTrackletIndex, outerTripletIndex, trackCandidateType))
    {
        pass = false;
    }

    return pass;
}

__device__ bool SDL::runTrackCandidateDefaultAlgoTrackletToTriplet(struct pixelTracklets& pixelTrackletsInGPU, struct tracklets& trackletsInGPU, struct triplets& tripletsInGPU, unsigned int innerTrackletIndex, unsigned int outerTripletIndex, short& trackCandidateType)
{
    bool pass = true;
    trackCandidateType = 1;
    if(not hasCommonSegment(pixelTrackletsInGPU, trackletsInGPU, tripletsInGPU, innerTrackletIndex, outerTripletIndex, trackCandidateType))
    {
        pass = false;
    }

    return pass;
}


__device__ bool SDL::runTrackCandidateDefaultAlgoTripletToTracklet(struct tracklets& trackletsInGPU, struct triplets& tripletsInGPU, unsigned int innerTripletIndex, unsigned int outerTrackletIndex, short& trackCandidateType)
{
    bool pass = true;
    trackCandidateType = 2;
    if(not hasCommonSegment(trackletsInGPU, tripletsInGPU, innerTripletIndex, outerTrackletIndex, trackCandidateType))
    {
        pass = false;
    }
    return pass;
}

__device__ bool SDL::hasCommonSegment(struct tracklets& trackletsInGPU, struct triplets& tripletsInGPU, unsigned int innerObjectIndex, unsigned int outerObjectIndex, short trackCandidateType)
{
    unsigned int innerObjectOuterSegmentIndex, outerObjectInnerSegmentIndex;

    if(trackCandidateType == 0)
    {
        //2 tracklets
        innerObjectOuterSegmentIndex = trackletsInGPU.segmentIndices[2 * innerObjectIndex + 1];
        outerObjectInnerSegmentIndex = trackletsInGPU.segmentIndices[2 * outerObjectIndex];
    }
    else if(trackCandidateType == 1)
    {
        //T4T3
        innerObjectOuterSegmentIndex = trackletsInGPU.segmentIndices[2 * innerObjectIndex + 1];
        outerObjectInnerSegmentIndex = tripletsInGPU.segmentIndices[2 * outerObjectIndex];
    }
    else if(trackCandidateType == 2)
    {
        //T3T4
        innerObjectOuterSegmentIndex = tripletsInGPU.segmentIndices[2 * innerObjectIndex + 1];
        outerObjectInnerSegmentIndex = trackletsInGPU.segmentIndices[2 * outerObjectIndex];
    }

    return (innerObjectOuterSegmentIndex == outerObjectInnerSegmentIndex);
}

__device__ bool SDL::hasCommonSegment(struct pixelTracklets& pixelTrackletsInGPU, struct tracklets& trackletsInGPU, struct triplets& tripletsInGPU, unsigned int innerObjectIndex, unsigned int outerObjectIndex, short trackCandidateType)
{
    unsigned int innerObjectOuterSegmentIndex, outerObjectInnerSegmentIndex;

    if(trackCandidateType == 0)
    {
        //2 tracklets
        innerObjectOuterSegmentIndex = pixelTrackletsInGPU.segmentIndices[2 * innerObjectIndex + 1];
        outerObjectInnerSegmentIndex = trackletsInGPU.segmentIndices[2 * outerObjectIndex];
    }
    else if(trackCandidateType == 1)
    {
        //T4T3
        innerObjectOuterSegmentIndex = pixelTrackletsInGPU.segmentIndices[2 * innerObjectIndex + 1];
        outerObjectInnerSegmentIndex = tripletsInGPU.segmentIndices[2 * outerObjectIndex];
    }

    return (innerObjectOuterSegmentIndex == outerObjectInnerSegmentIndex);
}
