#ifndef __MEMORY_POOL__
#define __MEMORY_POOL__

#include <cstddef>
#include <queue>
using namespace std;

struct Chunk {
    /** 
    * When free, 'next' stores the address of the
    * next chunk in the list.
    * When allocated the 'next' space is overwritten by user data. 
    */
  Chunk *next;
};

class PoolAllocator {
 public:
    PoolAllocator(size_t chunksPerBlock)
        : mChunksPerBlock(chunksPerBlock) {}

    ~PoolAllocator();
 
    void *allocate(size_t size);
    void deallocate(void *ptr, size_t size);
 
    private:
        size_t mChunksPerBlock;
        Chunk *mAlloc = nullptr;
        Chunk *allocateBlock(size_t size);
};
#endif