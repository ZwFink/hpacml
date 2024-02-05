#ifndef OMP_OPTIONAL_HH
#define OMP_OPTIONAL_HH
#ifdef USE_OPENMP
#include <omp.h>
#else
int omp_get_thread_num();
int omp_get_num_threads();
int omp_get_max_threads();
int omp_get_num_procs();
int omp_get_thread_limit();
int omp_get_level();
int omp_in_parallel();
#endif
#endif