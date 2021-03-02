#include <stdbool.h>

extern "C" {
bool __approx_skip_iteration(unsigned int i, float pr);
void __approx_exec_call(void (*accurate)(void *), void (*perforate)(void *),
                        void *arg, bool cond, void *perfoArgs, void *deps,
                        int num_deps, int memo_type);
}
