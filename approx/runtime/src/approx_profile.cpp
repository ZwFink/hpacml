#include <approx_profile.h>

BaseDataWriter* getProfiler(const char *Env){
    return new PapiProfiler(Env);
}
