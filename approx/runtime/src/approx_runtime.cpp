#include <stdint.h>
#include <string>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <unordered_map>

#include <approx.h>
#include <approx_data_util.h>
#include <approx_internal.h>
#include <approx_io.h>

using namespace std;
#define ENABLE_HDF5 1

#define MEMO_IN 1
#define MEMO_OUT 2

#define NUM_ROWS 1024
#define NUM_DIMS 2

enum ExecuteMode: uint8_t{
  EXECUTE,
  PROFILE_TIME,
  PROFILE_DATA
};
void _printdeps(approx_var_info_t *vars, int num_deps) {
  for (int i = 0; i < num_deps; i++) {
    printf("%p, NE:%ld, SE:%ld, DT:%s, DIR:%d\n", vars[i].ptr, vars[i].num_elem,
           vars[i].sz_elem, getTypeName((ApproxType)vars[i].data_type),
           vars[i].dir);
  }
}

class Profiler{
  std::unordered_map<const char *, std::pair<long, double>> profileData;
  std::unordered_map<const char *, std::chrono::time_point<std::chrono::high_resolution_clock>> startTime;
  public:
    Profiler(){}
    ~Profiler() {
     for (auto p : profileData) {
       std::string region_name = p.first;
       double avg_time = p.second.second/ p.second.first;
       std::cout<< region_name << ":" << avg_time << ":" << p.second.second << ":" << p.second.first <<"\n";
     }
    }
    void start_time(const char *val){
      startTime[val] = std::chrono::high_resolution_clock::now();
    }
    void stop_time (const char *val){
      std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> elapsed_time = (end - startTime[val]);
      std::unordered_map<const char *, std::pair<long, double>>::iterator iter = profileData.find(val);
      if (iter == profileData.end() )
        profileData.insert(std::make_pair(val,std::make_pair(1,elapsed_time.count())));
      else{
        iter->second.second += elapsed_time.count();
        iter->second.first++;
      }
    }
};

class ApproxRuntimeConfiguration{
  ExecuteMode Mode;
  public:
  Profiler profiler;
  BaseDataWriter *data_profiler;

    ApproxRuntimeConfiguration(){
      const char *env_p = std::getenv("EXECUTE_MODE");
      if (!env_p){
        Mode = EXECUTE;
        return;
      }

      if (strcmp(env_p, "TIME_PROFILE") == 0){
        Mode = PROFILE_TIME;
      }
      else if ( strcmp(env_p, "DATA_PROFILE") == 0 ){
        Mode = PROFILE_DATA;
        data_profiler = new HDF5DataWriter();
      }
      else{
        Mode = EXECUTE;
      }
    }

    ~ApproxRuntimeConfiguration(){
      delete data_profiler;
    }

    ExecuteMode getMode(){return Mode;}
};

ApproxRuntimeConfiguration RTEnv;



void __approx_exec_call(void (*accurate)(void *), void (*perforate)(void *),
                        void *arg, bool cond,  const char *region_name, void *perfoArgs, int memo_type,
                        void *inputs, int num_inputs, void *outputs,
                        int num_outputs) {
  approx_perfo_info_t *perfo = (approx_perfo_info_t *)perfoArgs;
  approx_var_info_t *input_vars = (approx_var_info_t *)inputs;
  approx_var_info_t *output_vars = (approx_var_info_t *)outputs;
//  std::cout << "[" << region_name << "] Inputs are:" << std::endl;
//  _printdeps(input_vars, num_inputs);
//  std::cout << "[" << region_name << "] Outputs are:" << std::endl;
//  _printdeps(output_vars, num_outputs);

  if (RTEnv.getMode() == PROFILE_TIME){
    RTEnv.profiler.start_time(region_name);
    accurate(arg);
    RTEnv.profiler.stop_time(region_name);
    return;
  }
#ifdef ENABLE_HDF5
  else if (RTEnv.getMode() == PROFILE_DATA){
    RTEnv.data_profiler->record_start(region_name, input_vars, num_inputs, output_vars, num_outputs);
    accurate(arg);
    RTEnv.data_profiler->record_end(region_name, output_vars, num_outputs);
    return;
  }
#endif
  else{
    if (cond) {
      if (memo_type == MEMO_IN) {
        memoize_in(accurate, arg, input_vars, num_inputs, output_vars, num_outputs);
      } else if (memo_type == MEMO_OUT) {
        memoize_out(accurate, arg, output_vars, num_outputs);
      }
    } else {
      accurate(arg);
    }
  }
}
