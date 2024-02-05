#include "optional_omp.h"

#ifndef USE_OPENMP
int omp_get_thread_num() { return 0; }
int omp_get_num_threads() { return 1; }
int omp_get_max_threads() { return 1; }
int omp_get_num_procs() { return 1; }
int omp_get_thread_limit() { return 1; }
int omp_get_level() { return 0; }
int omp_in_parallel() { return 0; }
#endif