#ifndef __PACK__
#define __PACK__
#include <queue>
#include <iostream>

#include "approx_internal.h"
#include "memory_pool/memory_pool.h"
#include "database/database.h"

#define BLOCK_SIZE (64*1024*1024) 

using namespace std;
class HPACRegion{
    public:
        size_t IElem;
        size_t OElem;
        size_t NumBytes;
        uintptr_t accurate;
        PoolAllocator allocator;
        BaseDB *db; 
        void *dbRegionId;
        size_t BlkSize;
        int CurrentBlk;
        void *MemPtr;
        int CurrentReleases;
        int ElementsInBlock;
        std::string Name;

        HPACRegion(uintptr_t Addr, size_t IElem, 
            size_t OElem, size_t chunks, const char *name) :
            accurate(Addr), IElem(IElem), OElem(OElem), 
            NumBytes((IElem + OElem)*sizeof(double)),
            allocator(chunks), db(nullptr), dbRegionId(nullptr),
            BlkSize(BLOCK_SIZE), CurrentBlk(0), MemPtr(nullptr), CurrentReleases(0),
            Name(name){
                ElementsInBlock = ((BlkSize) / NumBytes) + ((BlkSize % NumBytes) != 0);
                BlkSize = ElementsInBlock * NumBytes;
                std::cout<< "BlkSize is :" << BlkSize << "\n";
            };

        HPACRegion(uintptr_t Addr, size_t IElem, 
            size_t OElem, size_t chunks, BaseDB *db, void *dRId,
            const char *name) :
            accurate(Addr), IElem(IElem), OElem(OElem), 
            NumBytes((IElem + OElem)*sizeof(double)),
            allocator(chunks), db(db), dbRegionId(dRId),
            BlkSize(BLOCK_SIZE), CurrentBlk(0), MemPtr(nullptr), CurrentReleases(0),
            Name(name){
                ElementsInBlock = ((BlkSize) / NumBytes) + ((BlkSize % NumBytes) != 0);
                BlkSize = ElementsInBlock * NumBytes;
            };

        void release(void * ptr){
            CurrentReleases++;
            if (CurrentReleases == ElementsInBlock && CurrentReleases == CurrentBlk){
                if (db)
                    db->DataToDB(dbRegionId, (double *) MemPtr, CurrentBlk, IElem + OElem);
                allocator.deallocate(MemPtr, sizeof(double) * (IElem + OElem) * BlkSize);
                CurrentReleases = 0;
                CurrentBlk = 0;
                MemPtr = nullptr;
            }
        }

        ~HPACRegion() {
            if (MemPtr != nullptr){
                if (db)
                    db->DataToDB(dbRegionId, (double *) MemPtr, CurrentBlk, IElem + OElem);
                allocator.deallocate(MemPtr, BlkSize);
            }
        }

        void* allocate(){
            if ( MemPtr !=nullptr ){
                if (CurrentBlk < ElementsInBlock){
                    int Index = CurrentBlk;
                    CurrentBlk += 1;
                    return (void *) &(static_cast<double*> (MemPtr))[Index*(IElem + OElem)];
                }
                else{
                    std::cerr<< "This should never happen, memory has not been released\n";
                    exit(-1);
                }
            }
            else{
                MemPtr = allocator.allocate(BlkSize);
                CurrentBlk = 1;
                return MemPtr;
            }
        }

        size_t getBlockSize() { return BlkSize; }
        size_t getNumRows() { return ElementsInBlock ; }
        void setDBRegionId(void *RID) { dbRegionId = RID; }
        void setDB( BaseDB *DB) {db = DB;}
        std::string getName() { return Name; }
};

struct HPACPacket{
    double *inputs; 
    double *outputs;
    HPACRegion *feature;
    ~HPACPacket() {
        feature->release((void *) inputs);
        inputs = nullptr;
        outputs = nullptr;
        feature = nullptr;
    }
};

#endif