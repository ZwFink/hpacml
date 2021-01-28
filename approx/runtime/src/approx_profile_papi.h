#ifndef __APPROX_PROFILE_PAPI__
#define __APPROX_PROFILE_PAPI__

#include <hdf5.h>
#include <unordered_map>
#include <vector>

#include <approx_internal.h>
#include <approx_profile.h>

enum STATS{
    AVGI, STDI, MINI, MAXI, ESTAT};

class RegionProfiler {
  std::vector<long long *> Accum;
  char *RName;
  unsigned int NStats;
  double *Stats;

public:
  RegionProfiler(const char *RName, int NStats);
  ~RegionProfiler();
  void increaseStats(long long *CStats);
  double *getStats();
  char *getName() { return RName; }
};

class PapiProfiler : public BasePerfProfiler {
  hid_t FileId;
  long long *TStats;
  char **EventNames;
  int ProfileEvents;
  int BIndex;
  int TEvents;
  int LEvents;
  int HCounters;
  char *PapiNames;
  std::unordered_map<uintptr_t, RegionProfiler *> AddrToStats;

public:
  PapiProfiler(const char *FName);
  ~PapiProfiler();
  void startProfile(const char *RName, uintptr_t FnAddr);
  void stopProfile(const char *RName, uintptr_t FnAddr);
};

#endif