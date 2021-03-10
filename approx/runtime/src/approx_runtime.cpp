#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <random>

#include "approx.h"
#include "approx_data_util.h"
#include "approx_internal.h"
#include "approx_io.h"
#include "approx_profile.h"

using namespace std;
#define ENABLE_HDF5 1

#define MEMO_IN 1
#define MEMO_OUT 2

#define NUM_ROWS 1024
#define NUM_DIMS 2

enum ExecuteMode : uint8_t { EXECUTE, PROFILE_TIME, PROFILE_DATA };

void _printdeps(approx_var_info_t *vars, int num_deps) {
  for (int i = 0; i < num_deps; i++) {
    printf("%p, NE:%ld, SE:%ld, DT:%s, DIR:%d\n", vars[i].ptr, vars[i].num_elem,
           vars[i].sz_elem, getTypeName((ApproxType)vars[i].data_type),
           vars[i].dir);
  }
}

class ApproxRuntimeConfiguration {
  ExecuteMode Mode;
  bool ExecuteBoth;
public:
  BasePerfProfiler *TimeProfiler;
  BaseDataWriter *DataProfiler;
  int approxStep;
  float approxRate;
  int tableSize;
  float threshold;
  int historySize;
  int predictionSize;

  ApproxRuntimeConfiguration() {
      TimeProfiler = nullptr;
      DataProfiler = nullptr;
      ExecuteBoth = false;

    const char *env_p = std::getenv("EXECUTE_BOTH");
    if (env_p){
      ExecuteBoth = true;
    }

    env_p = std::getenv("EXECUTE_MODE");
    if (!env_p) {
      Mode = EXECUTE;
    } else if (strcmp(env_p, "TIME_PROFILE") == 0) {
      Mode = PROFILE_TIME;
      env_p = std::getenv("DATA_FILE");
      if (!env_p) {
        env_p = "test.h5";
      }
      TimeProfiler = getProfiler(env_p);
    } else if (strcmp(env_p, "DATA_PROFILE") == 0) {
      Mode = PROFILE_DATA;
      env_p = std::getenv("DATA_FILE");
      if (!env_p) {
        env_p = "test.h5";
      }
      DataProfiler = getDataWriter(env_p);
    } else {
      Mode = EXECUTE;
    }

    env_p = std::getenv("THRESHOLD");
    if (env_p) {
      threshold = atof(env_p);
    }

    tableSize = 0;
    env_p = std::getenv("TABLE_SIZE");
    if (env_p){
      tableSize = atoi(env_p);
    }

    env_p = std::getenv("PREDICTION_SIZE");
    if (env_p) {
      predictionSize = atoi(env_p);
    }

    env_p = std::getenv("HISTORY_SIZE");
    if (env_p) {
      historySize = atoi(env_p);
    }

    env_p = std::getenv("THRESHOLD");
    if (env_p) {
      threshold = atof(env_p);
    }


  }

  bool getExecuteBoth(){ return ExecuteBoth; }
  ExecuteMode getMode() {return Mode; }
};

ApproxRuntimeConfiguration RTEnv;

const float approx_rt_get_percentage(){
  return RTEnv.approxRate;
}

const int approx_rt_get_Step(){
  return RTEnv.approxStep;
}

bool __approx_skip_iteration(unsigned int i, float pr) {
    // TODO: random seed? reproducible?
    static std::default_random_engine generator;
    static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
    float n = distribution(generator);
    if (n <= pr) {
        return true;
    }
    return false;
}


void __approx_exec_call(void (*accurate)(void *), void (*perforate)(void *),
                        void *arg, bool cond, const char *region_name,
                        void *perfoArgs, int memo_type, void *inputs,
                        int num_inputs, void *outputs, int num_outputs) {
  approx_perfo_info_t *perfo = (approx_perfo_info_t *)perfoArgs;
  approx_var_info_t *input_vars = (approx_var_info_t *)inputs;
  approx_var_info_t *output_vars = (approx_var_info_t *)outputs;

  if (RTEnv.getMode() == PROFILE_TIME) {
    RTEnv.TimeProfiler->startProfile(region_name, (uintptr_t) accurate);
    accurate(arg);
    RTEnv.TimeProfiler->stopProfile(region_name, (uintptr_t) accurate);
  } else if (RTEnv.getMode() == PROFILE_DATA) {
    RTEnv.DataProfiler->record_start(region_name, input_vars, num_inputs,
                                     output_vars, num_outputs);
    accurate(arg);
    RTEnv.DataProfiler->record_end(region_name, output_vars, num_outputs);
  } else {
    if (cond) {
      if ( perforate ){
        perforate(arg);
      } else if (memo_type == MEMO_IN) {
        memoize_in(accurate, arg, input_vars, num_inputs, output_vars,
                   num_outputs, RTEnv.getExecuteBoth(), RTEnv.tableSize, RTEnv.threshold );
      } else if (memo_type == MEMO_OUT) {
        memoize_out(accurate, arg, output_vars, num_outputs, RTEnv.getExecuteBoth(), RTEnv.predictionSize, RTEnv.historySize, RTEnv.threshold);
      } else {
        accurate(arg);
      }
    } else {
      accurate(arg);
    }
  }
  return;
}
