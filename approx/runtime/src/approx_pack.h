#ifndef __PACK__
#define __PACK__
#include <queue>
#include <iostream>

#include "approx_internal.h"
#include "memory_pool/memory_pool.h"
#include "database/database.h"

#define BLOCK_SIZE (4096*16) 

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
        int BlkSize;
        int CurrentBlk;
        void *MemPtr;
        int CurrentReleases;
        std::string Name;

        HPACRegion(uintptr_t Addr, size_t IElem, 
            size_t OElem, size_t chunks, const char *name) :
            accurate(Addr), IElem(IElem), OElem(OElem), 
            NumBytes((IElem + OElem)*sizeof(double)),
            allocator(chunks), db(nullptr), dbRegionId(nullptr),
            BlkSize(BLOCK_SIZE), CurrentBlk(0), MemPtr(nullptr), CurrentReleases(0),
            Name(name){};

        HPACRegion(uintptr_t Addr, size_t IElem, 
            size_t OElem, size_t chunks, BaseDB *db, void *dRId,
            const char *name) :
            accurate(Addr), IElem(IElem), OElem(OElem), 
            NumBytes((IElem + OElem)*sizeof(double)),
            allocator(chunks), db(db), dbRegionId(dRId),
            BlkSize(BLOCK_SIZE), CurrentBlk(0), MemPtr(nullptr), CurrentReleases(0),
            Name(name){};

        void release(void * ptr){
            CurrentReleases++;
            if (CurrentReleases == BlkSize && CurrentReleases == CurrentBlk){
                if (db)
                    db->DataToDB(dbRegionId, (double *) MemPtr, BlkSize, IElem + OElem);
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
                allocator.deallocate(MemPtr, sizeof(double) * (IElem + OElem) * CurrentBlk);
            }
        }

        void* allocate(){
            if ( MemPtr !=nullptr ){
                if (CurrentBlk < BlkSize){
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
                MemPtr = allocator.allocate((IElem + OElem)*sizeof(double)*BlkSize);
                CurrentBlk = 1;
                return MemPtr;
            }
        }


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