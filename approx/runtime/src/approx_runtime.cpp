#include "approx.h"
#include <stdint.h>
#include <stdio.h>
#include <random>

typedef struct approx_perfo_info_t {
  int type;
  int region;
  int step;
  float rate;
} approx_perfo_info_t;


//VarInfoTy is a struct containing info about the in/out/inout variables
//of this region.
typedef struct approx_var_info_t{
    void* ptr;         // Ptr to data
    size_t sz_bytes;   // size of data in bytes
    size_t num_elem;   // Number of elements
    size_t sz_elem;    // Size of elements in bytes
    int16_t data_type; // Type of data float/double/int etc.
    uint8_t dir;       // Direction of data: in/out/inout
} approx_var_info_t;

void _printdeps(approx_var_info_t *vars, int num_deps) {
  for (int i = 0; i < num_deps; i++) {
    printf("%p, SB:%ld, NE:%ld, SE:%ld, DT:%d, DIR:%d\n", vars[i].ptr,
           vars[i].sz_bytes, vars[i].num_elem, vars[i].sz_elem,
           vars[i].data_type, vars[i].dir);
  }
}

bool __approx_skip_iteration(unsigned int i, float pr) {
    // TODO: random seed? reproducible?
    static std::default_random_engine generator;
    static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    float n = distribution(generator);
    if (n <= pr) {
        printf("SKIP n %f __approx_skip_iteration i %d pr %f\n", n, i, pr);
        return true;
    }

    printf("DO n %f __approx_skip_iteration i %d pr %f\n", n, i, pr);
    return false;
}

void __approx_exec_call(void (*accFn)(void *), void (*perfFn)(void *),
                        void *arg, bool cond, void *perfoArgs,
                        void *deps, int num_deps, int memo_type) {
  approx_perfo_info_t *perfo = (approx_perfo_info_t *)perfoArgs;
  approx_var_info_t *var_info = (approx_var_info_t *)deps;
  _printdeps(var_info, num_deps);

  if (cond) {
    if (perfFn) {
      printf("CALLING cond perforated function\n");
      perfFn(arg);
    } else {
      printf("CALLING cond accurate function\n");
      accFn(arg);
    }
  } else {
    printf("CALLING nocond accurate function\n");
    accFn(arg);
  }
}
