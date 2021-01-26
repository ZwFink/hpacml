#include <approx_profile.h>

BasePerfProfiler* getProfiler(const char *Env){
    return new PapiProfiler(Env);
}
