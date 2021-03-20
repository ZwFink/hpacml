#include "approx_internal.h"
#include "approx_memoize_iact_in.h"

ThreadMemoryPool<MemoizeInput> inputMemo;
void memoize_in(void (*accurate)(void *), void *arg, approx_var_info_t *inputs,
                int num_inputs, approx_var_info_t *outputs, int num_outputs, bool ExecBoth, int tSize, real_t threshold) {
    static thread_local int threadId = -1;
  MemoizeInput *curr;

    if (threadId == -1){
        if (omp_in_parallel())
            threadId = omp_get_thread_num();
        else
            threadId = 0;
    }

  curr = inputMemo.findMemo(threadId, (unsigned long)accurate);
  if (!curr){ 
    curr = inputMemo.addNew(threadId, new MemoizeInput(accurate, num_inputs, num_outputs, inputs, outputs,getTableSize(), getThreshold()));
  }

    curr->execute_both(arg, inputs, outputs);
    /*
  if (ExecBoth)
    curr->execute_both(arg, inputs, outputs);
  else
    curr->execute(arg, inputs, outputs);
    */
}
