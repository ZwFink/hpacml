#include "approx.h"
#include <stdio.h>

typedef struct approx_perfo_info_t{
    int type;
    int region;
    int step;
    float rate;
} approx_perfo_info_t;

typedef struct approx_var_info_t{
    void* ptr;
    int direction;
    int data_type;
    long num_elements;
    int opaque;
} approx_var_info_t;

void __approx_exec_call(void (*accFn)(void *), void (*perfFn)(void *),
                        void *arg, unsigned char cond, void *perfoArgs, void *deps, int num_deps) {
 printf("I received condition %d \n", cond);
 printf("Num Dependencies: %d \n", num_deps);
 approx_perfo_info_t *perfo = (approx_perfo_info_t *) perfoArgs;
 approx_var_info_t  *var_info = (approx_var_info_t *) deps;
 for ( int i = 0; i < num_deps; i++)
   printf("dep[%d]: PTR:%p direction %d elements:%ld \n",i, var_info[i].ptr, var_info[i].direction, var_info[i].num_elements);


  printf("Debug Perfo Args : (%d,%d,%d,%f)\n", perfo->type, perfo->region, perfo->step, perfo->rate);
  if (cond) {
    if (perfFn) {
      perfFn(arg);
    } else {
      printf("Perfo Is not implemented\n");
      accFn(arg);
    }
  } else {
    accFn(arg);
  }
}
