#ifndef __APPROX_PROFILE__
#define __APPROX_PROFILE__

#include <approx_internal.h>
#include <unordered_map>


class BasePerfProfiler{
    public:
    BasePerfProfiler() {};
    BasePerfProfiler(const char *FName) {};
    virtual ~BasePerfProfiler() {};
    virtual void startProfile(const char *RName, uintptr_t FnAddr) {};
    virtual void stopProfile(const char *RName, uintptr_t FnAddr) {};
};

#endif