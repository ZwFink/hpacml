#include <stdlib.h>

#include "memory_pool.h"

Chunk *PoolAllocator::allocateBlock(size_t chunkSize) {
  size_t blockSize = mChunksPerBlock * chunkSize;
  Chunk *blockBegin = reinterpret_cast<Chunk *>(malloc(blockSize));
  Chunk *chunk = blockBegin;
  for (int i = 0; i < mChunksPerBlock - 1; ++i) {
    chunk->next =
        reinterpret_cast<Chunk *>(reinterpret_cast<char *>(chunk) + chunkSize);
    chunk = chunk->next;
  }
  chunk->next = nullptr;
  return blockBegin;
}

void *PoolAllocator::allocate(size_t size) {
  if (mAlloc == nullptr) 
    mAlloc = allocateBlock(size);
 
  Chunk *freeChunk = mAlloc;
  mAlloc = mAlloc->next;
 
  return freeChunk;
}

/**
 * Puts the chunk into the front of the chunks list.
 */
void PoolAllocator::deallocate(void *chunk, size_t size) {
  reinterpret_cast<Chunk *>(chunk)->next = mAlloc;
  mAlloc = reinterpret_cast<Chunk *>(chunk);
}

PoolAllocator::~PoolAllocator(){
    //TODO: Keep track of allocated memory and garbase collect it.
}