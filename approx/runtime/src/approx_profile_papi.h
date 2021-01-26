#ifndef __APPROX_PROFILE_PAPI__
#define __APPROX_PROFILE_PAPI__

#include <unordered_map>
#include <hdf5.h>

#include <approx_internal.h>
#include <approx_profile.h>


class RegionProfiler{
    long long *Accum;
    char *RName;
    unsigned int NStats;
    unsigned int NInvocations;
    public:
    RegionProfiler(const char *RName, int NStats);
    ~RegionProfiler();
    void increaseStats(long long *CStats);
    long long* getStats() {return Accum;}
    char* getName(){return RName;}
};

class PapiProfiler : public BasePerfProfiler{
    hid_t FileId;
    long long *TStats;
    int ProfileEvents;
    int BIndex;
    int TEvents;
    int LEvents;
    int HCounters;
    char *PapiNames;
    std::unordered_map<uintptr_t, RegionProfiler*> AddrToStats;
    PapiProfiler(const char *FName);
    ~PapiProfiler();
    void startProfile(const char *RName, uintptr_t FnAddr);
    void stopProfile(const char *RName, uintptr_t FnAddr);
};

#endif