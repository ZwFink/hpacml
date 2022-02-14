#ifndef __PACK__
#define __PACK__
#include <queue>

#include "approx_internal.h"
#include "memory_pool/memory_pool.h"
#include "database/database.h"

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
        HPACRegion(uintptr_t Addr, size_t IElem, 
            size_t OElem, size_t chunks) :
            accurate(Addr), IElem(IElem), OElem(OElem), 
            NumBytes((IElem + OElem)*sizeof(double)),
            allocator(chunks), db(nullptr), dbRegionId(nullptr) {};

        HPACRegion(uintptr_t Addr, size_t IElem, 
            size_t OElem, size_t chunks, BaseDB *db, void *dRId) :
            accurate(Addr), IElem(IElem), OElem(OElem), 
            NumBytes((IElem + OElem)*sizeof(double)),
            allocator(chunks), db(db), dbRegionId(dRId) {};

        void deallocate(void * ptr){
            if (db){
                db->DataToDB(dbRegionId, (double *) ptr, 1, IElem + OElem);
            }
            allocator.deallocate(ptr, sizeof(double) * (IElem + OElem));
        }

        void* allocate(){
            return allocator.allocate(NumBytes);
        }
};

struct HPACPacket{
    double *inputs; 
    double *outputs;
    HPACRegion *feature;
    ~HPACPacket() {
        feature->deallocate((void *) inputs);
        inputs = nullptr;
        outputs = nullptr;
        feature = nullptr;
    }
};

#endif