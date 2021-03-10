#include <cfloat>
#include <iostream>
#include <memory>
#include <string>
#include <cstring>
#include <unordered_map>
#include <cmath>
#include <omp.h>

#include "approx_data_util.h"
#include "approx_internal.h"
#include "thread_storage.h"

#define MAX_REGIONS 20
#define NOTFOUND -1

using namespace std;

//!  An Output Memoization class.
/*!
    This is an implementation of the output memoization described in :
    G. Tziantzioulis, N. Hardavellas and S. Campanoni, "Temporal
    Approximate Function Memoization," in IEEE Micro, vol. 38, no. 4,
    pp. 60-70, Jul./Aug. 2018, doi: 10.1109/MM.2018.043191126.

    There are certain assumption of the current implementation

    ASSUMPTION 1) : Each region is always going to be executed
    with the same shape. We do not support regions that change
    the size of their outputs during execution time. If a region
    is invoked the first time with a single output of 10 double
    elements. In all upcoming calls this region should work using
    the same "shape" single output with 10 double elements.

    ASSUMPTION 2) : When Computing the average value of multiple
    outputs we assume all outputs have the same weight. This might
    not be fair in case of outputs that are vectors. Assuming a case
    in which we have 2 outputs a vector of N elements (vec) and a
    scalar variable x. The average is compute as:
    avg = (avg(x) + avg(vec))/2. This implicitly reduces the
    importance of each element vec[i] in the computation of
    the average
*/
class MemoizeOutput {
public:
  /// The State Machine of this technique.
  typedef enum : bool { ACCURATE, APPROXIMATE } executionState;

  /// The accurate function.
  void (*accurate)(void *);
  /// The number of outputs this function produces.
  int num_outputs;
  /// The size of the prediction
  int prediction_size;
  /// The size of the history
  int history_size;
  /// The last computed output values.
  uint8_t **last_values;
  /// The window contais the average value of
  /// the last history_size outputs.
  real_t *window;
  /// The average value of the window
  real_t avg;
  /// The stdev of the window.
  real_t stdev;
  /// RSD = fabs(stdev/average)
  real_t rsd;
  /// User provided threshold
  real_t threshold;
  /// Index of storing in the window.
  int cur_index;
  /// Counter, contains number of invocations done
  /// after the last approximation.
  int active_values;
  /// Counter, contains how many approximate invocations
  /// prior changing state.
  int predicted_values;
  /// Current state of the execution.
  executionState state;
  /// Stat. Counts how many invocations where performed accurately.
  int accurately;
  /// Stat. Counts how many invocations where performed approximately.
  int approximately;
  double outAverage[2];
  int totalInvocations;

private:
  /// Stores last computed output to private memory.
  void register_output(approx_var_info_t *outputs) {
    for (int i = 0; i < num_outputs; i++) {
      memcpy(last_values[i], outputs[i].ptr,
             outputs[i].num_elem * outputs[i].sz_elem);
    }
  }
  /// Copies last computed outputs to current output.
  void approximate(approx_var_info_t *outputs) {
    for (int i = 0; i < num_outputs; i++) {
      memcpy(outputs[i].ptr, last_values[i],
             outputs[i].num_elem * outputs[i].sz_elem);
    }
  }

public:
  MemoizeOutput(void(acc)(void *), approx_var_info_t *outputs, int num_outputs, int pSize, int hSize, float threshold)
      : accurate(acc), num_outputs(num_outputs), history_size(hSize), prediction_size(pSize), 
        avg(0.0), stdev(0.0), rsd(FLT_MAX), threshold(threshold),
        cur_index(0), active_values(0), predicted_values(0), state(ACCURATE),
        accurately(0), approximately(0), totalInvocations(0) {
    // The memory allocation of this approximation technique is far from
    // optimal. In essence we can exploit make allocation to depend on the type
    // of the outputs. and allocate large chunks of memory of the same type.
    // Then we can apply better simd operations on these large chunks of memory.

    last_values = new uint8_t *[num_outputs];
    for (int i = 0; i < num_outputs; i++) {
      last_values[i] = new uint8_t[outputs[i].num_elem * outputs[i].sz_elem];
    }
    window = new real_t[history_size]();

    for (int i = 0; i < 2; i++){
      outAverage[i] = 0.0;
    }
  }

  ~MemoizeOutput() {
    if (totalInvocations != 0 ){
      double MAPE = fabs(outAverage[0] - outAverage[1])/fabs(outAverage[0]);
      cout << "REGION_ERROR:"<<  MAPE << endl;
    }

    cout << "APPROX:"
         << (double)approximately / (double)(accurately + approximately)
         << ":" << approximately<< ":" << accurately << endl;
    for (int i = 0; i < num_outputs; i++) {
      delete[] last_values[i];
      last_values[i] = nullptr;
    }
    delete last_values;
    delete window;
  }

  /// The actual implementation of the approximation technique.
  void execute(void *arg, approx_var_info_t *outputs) {
    if (state == APPROXIMATE) {
      approximate(outputs);
      approximately++;
    } else if (state == ACCURATE) {
      accurate(arg);
      register_output(outputs);
      accurately++;
    }

    // The following code is the actual overhead of.
    // the approximation technique. In the case of
    // output memoization this part of code could also be
    // executed by another "worker thread" to hide the latencies
    // of the approximation technique itself. In serial applications
    // this is kind of straight forward. In parallel application versions
    // this needs to be studied.

    if (state == ACCURATE) {
      // When executed in accurate state we need to
      // compute the new avg, stdev, rsd, and change
      // state if necessary for the next invocation.

      real_t new_value = 0.0;
      // Compute new incoming value. Here we
      // actually utilize assumption 1.
      // This should be further investigated.
      for (int i = 0; i < num_outputs; i++) {
        new_value += average(last_values[i], outputs[i].num_elem,
                             (ApproxType)outputs[i].data_type);
      }
      new_value /= (double)num_outputs;
      real_t old_value = window[cur_index];
      window[cur_index] = new_value;

      cur_index = (cur_index + 1) % history_size;
      avg = avg + (new_value - old_value) / (real_t)history_size;
      active_values++;
      real_t variance = 0.0;
      if (active_values >= history_size) {
        for (int i = 0; i < history_size; i++) {
          real_t tmp = (window[i] - avg);
          variance += tmp * tmp;
        }
        variance /= (real_t)history_size;
        /// We might consider using approximable
        // versions of math functions which are considerably
        // faster but ofcource the results will not be that accurate.
        stdev = sqrt(variance);
        rsd = fabs(stdev / avg);
        if (rsd < threshold) {
          state = APPROXIMATE;
          predicted_values = prediction_size;
        }
      }
    } else if (state == APPROXIMATE) {
      if (--predicted_values == 0) {
        state = ACCURATE;
        active_values = history_size - predicted_values;
        if (active_values < 0) {
          active_values = 0;
        }
      }
    }
  }

  void execute_both(void *arg, approx_var_info_t *outputs) {
    totalInvocations++;
    accurate(arg);
    float correctOut = 0.0f;
    for (int i = 0; i < num_outputs; i++){
      correctOut += aggregate(outputs[i].ptr, outputs[i].num_elem,
                 (ApproxType)outputs[i].data_type);
    }
    outAverage[0] += correctOut;

    if (state == APPROXIMATE) {
      approximate(outputs);
      for (int i = 0; i < num_outputs; i++){
        outAverage[1] += aggregate(outputs[i].ptr, outputs[i].num_elem,
                 (ApproxType)outputs[i].data_type);
      }
      approximately++;
    } else if (state == ACCURATE) {
      outAverage[1] += correctOut;
      register_output(outputs);
      accurately++;
    }

    // The following code is the actual overhead of.
    // the approximation technique. In the case of
    // output memoization this part of code could also be
    // executed by another "worker thread" to hide the latencies
    // of the approximation technique itself. In serial applications
    // this is kind of straight forward. In parallel application versions
    // this needs to be studied.

    if (state == ACCURATE) {
      // When executed in accurate state we need to
      // compute the new avg, stdev, rsd, and change
      // state if necessary for the next invocation.

      real_t new_value = 0.0;
      // Compute new incoming value. Here we
      // actually utilize assumption 1.
      // This should be further investigated.
      for (int i = 0; i < num_outputs; i++) {
        new_value += average(last_values[i], outputs[i].num_elem,
                             (ApproxType)outputs[i].data_type);
      }
      new_value /= (double)num_outputs;
      real_t old_value = window[cur_index];
      window[cur_index] = new_value;

      cur_index = (cur_index + 1) % history_size;
      avg = avg + (new_value - old_value) / (real_t)history_size;
      active_values++;
      real_t variance = 0.0;
      if (active_values >= history_size) {
        for (int i = 0; i < history_size; i++) {
          real_t tmp = (window[i] - avg);
          variance += tmp * tmp;
        }
        variance /= (real_t)history_size;
        /// We might consider using approximable
        // versions of math functions which are considerably
        // faster but ofcource the results will not be that accurate.
        stdev = sqrt(variance);
        rsd = fabs(stdev / avg);
        if (rsd < threshold) {
          state = APPROXIMATE;
          predicted_values = prediction_size;
        }
      }
    } else if (state == APPROXIMATE) {
      if (--predicted_values == 0) {
        state = ACCURATE;
        active_values = history_size - predicted_values;
        if (active_values < 0) {
          active_values = 0;
        }
      }
    }
  }
};


MemoPool<MemoizeOutput> outMemo;

/**
 * Memoize code region using output memoize technique
 *
 * @param  accurate function pointer of the code region to be approximated.
 * @param  arg Arguments of accurate function
 * @param  outputs outputs produced by this code region.
 * @param  num_outputs number of outputs
 * @return void.
 *
 * @brief This function will try to approximate the code region specified
 * by the function pointer. It uses the implementation of the MemoizeOutput
 * class to perform all the required approximation steps.
 */
void memoize_out(void (*accurate)(void *), void *arg,
                 approx_var_info_t *outputs, int num_outputs, bool ExecBoth, int pSize, int hSize, float threshold){
  int threadId = 0;
  MemoizeOutput *curr;
  if (omp_in_parallel()){
    threadId = omp_get_thread_num();
  }

  curr = outMemo.findMemo(threadId, (unsigned long)accurate);
  if (!curr){ 
    curr = outMemo.addNew(threadId, new MemoizeOutput(accurate, outputs, num_outputs, pSize, hSize, threshold));
  }

  if (ExecBoth)
    curr->execute_both(arg, outputs);
  else
    curr->execute(arg, outputs);
}
