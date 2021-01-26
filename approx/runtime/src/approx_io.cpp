#include <approx_io.h>

BaseDataWriter* getDataWriter(const char *Env){
#ifdef ENABLE_HDF5
    return new HDF5DataWriter(Env);
#else
    return new BaseDataWriter(Env);
#endif
}
