#include <approx_profile.h>
#include <approx_profile_papi.h>

BasePerfProfiler* getProfiler(const char *Env){
    return new PapiProfiler(Env);
}
