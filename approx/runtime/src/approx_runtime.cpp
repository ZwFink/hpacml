//===--- approx_runtime.cpp - driver of approximate runtime system----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This files is the driver of the approximate runtime 
///
//===----------------------------------------------------------------------===//
//


#include <stdint.h>
#include <string>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <unordered_map>
#include <random>
#include "optional_omp.h"

#include "approx.h"
#include "approx_data_util.h"
#include "approx_pack.h"
#include "approx_internal.h"
#include "thread_storage.h"
#include "database/database.h"
#include "approx_surrogate.h"
#include "approx_tensor.h"


using namespace std;

#define MEMO_IN 1
#define MEMO_OUT 2

#define PETRUBATE_IN 1
#define PETRUBATE_OUT 2


enum MLType: uint {
  ML_ONLINETRAIN = 1,
  ML_OFFLINETRAIN,
  ML_INFER,
  ML_END
};


#define RAND_SIZE  10000

float __approx_perfo_rate__;
int __approx_perfo_step__;

enum ExecuteMode: uint8_t{
  EXECUTE
};

class ApproxRuntimeConfiguration{
  ExecuteMode Mode;
public:
  bool ExecuteBoth;
  int tableSize;
  float threshold;
  int historySize;
  int predictionSize;
  int perfoStep;
  float perfoRate;
  float *randomNumbers;
  int count;
  BaseDB *db;
  SurrogateModel<GPUExecutionPolicy, CatTensorTranslator<double>, double> Model{
      "/scratch/mzu/zanef2/surrogates/SurrogateBenchmarks/models/lulesh/model.pt", false};


  ApproxRuntimeConfiguration() {
      ExecuteBoth = false;
      count = 0;

    const char *env_p = std::getenv("EXECUTE_BOTH");
    if (env_p){
      ExecuteBoth = true;
    }

    env_p = std::getenv("HPAC_DB_FILE");
    if (env_p) {
      db = new HDF5DB(env_p);
    } else {
      db = new HDF5DB("test.h5");
    }

    env_p = std::getenv("SURROGATE_MODEL");
    if (env_p) {
      Model.set_model(env_p);
    }

    env_p = std::getenv("EXECUTE_MODE");
    if (!env_p) {
      Mode = EXECUTE;
    } else{
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

    env_p = std::getenv("PERFO_STEP");
    if (env_p) {
      perfoStep = atoi(env_p);
      __approx_perfo_step__ = perfoStep;
    }

    env_p = std::getenv("PERFO_RATE");
    if (env_p) {
      perfoRate = atof(env_p);
      __approx_perfo_rate__ = perfoRate;
    }

    env_p = std::getenv("PETRUBATE_TYPE");
    if (env_p) {
      const char *type = env_p;
      env_p = std::getenv("PETRUBATE_FILE");
      const char *fName = env_p; 
      register_petrubate(fName, type);
    }


 // This is not the optimal way. Since, we will 
 // always use the same random numbers.
    int numThreads = 32; //omp_get_max_threads();
    randomNumbers = new float[RAND_SIZE*numThreads];
    static std::default_random_engine generator;
    static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    for (int i = 0 ; i < RAND_SIZE*numThreads; i++){
     randomNumbers[i] = distribution(generator);
    }
  }

  ~ApproxRuntimeConfiguration(){
    delete [] randomNumbers;
    delete db;
    deinitPetrubate();
  }

  ExecuteMode getMode(){return Mode;}

  bool getExecuteBoth(){ return ExecuteBoth; }

};

ApproxRuntimeConfiguration RTEnv;
ThreadMemoryPool<HPACRegion> HPACRegions;

#define NUM_CHUNKS 8


int getPredictionSize() { return RTEnv.predictionSize;}
int getHistorySize() { return RTEnv.historySize; }
int getTableSize() { return RTEnv.tableSize; }
float getThreshold(){ return RTEnv.threshold;}


bool __approx_skip_iteration(unsigned int i, float pr) {
    static thread_local int index = 0;
    static thread_local int threadId = -1;
    if ( threadId == -1 ){
        threadId = 0;
        if (omp_in_parallel()){
            threadId = omp_get_thread_num();
        }
    }
         
    if (RTEnv.randomNumbers[threadId*RAND_SIZE + index++] <= pr) {
        return true;
    }
    index = (index+1)%RAND_SIZE;
    return false;
}

static inline void
create_snapshot_packet(HPACPacket &dP, void (*user_fn)(void *),
                       const char *region_name, approx_var_info_t *inputs,
                       int num_inputs, approx_var_info_t *outputs,
                       int num_outputs) {
  // thread_local int threadId = -1;
  // thread_local HPACRegion *curr;
  // if(region_name == nullptr) {
  //   region_name = "unknown";
  // }
  // if (threadId == -1) {
  //   if (omp_in_parallel())
  //     threadId = omp_get_thread_num();
  //   else
  //     threadId = 0;
  // }

  // if (curr && (curr->accurate != (unsigned long)user_fn ||
  //              curr->getName() != region_name))
  //   curr = HPACRegions.findMemo(threadId, (unsigned long)user_fn, region_name);

  // if (!curr) {
  //   int IElem = computeNumElements(inputs, num_inputs);
  //   int OElem = computeNumElements(outputs, num_outputs);
  //   if (RTEnv.db != nullptr) {
  //     curr = new HPACRegion((uintptr_t)user_fn, IElem, OElem, NUM_CHUNKS,
  //                           region_name);
  //     void *dbRId =
  //         RTEnv.db->InstantiateRegion((uintptr_t)user_fn, region_name, inputs,
  //                                     num_inputs, outputs, num_outputs, curr->getNumRows());
  //     curr->setDB(RTEnv.db);
  //     curr->setDBRegionId(dbRId);
  //     HPACRegions.addNew(threadId, curr);
  //   } else {
  //     curr = new HPACRegion((uintptr_t)user_fn, IElem, OElem, NUM_CHUNKS,
  //                           region_name);
  //     HPACRegions.addNew(threadId, curr);
  //   }
  // }

  // double *dPtr = reinterpret_cast<double *>(curr->allocate());
  // dP.inputs = dPtr;
  // dP.outputs = dPtr + curr->IElem;
  // dP.feature = curr;
  // return;
}

// This is the main driver of the HPAC approach.
void __snapshot_call__(void (*_user_fn_)(void *), void *args,
                       const char *region_name, void *inputs, int num_inputs,
                       void *outputs, int num_outputs) {
  HPACPacket dP;
  approx_var_info_t *input_vars = (approx_var_info_t *)inputs;
  approx_var_info_t *output_vars = (approx_var_info_t *)outputs;

  create_snapshot_packet(dP, _user_fn_, region_name, input_vars, num_inputs,
                         output_vars, num_outputs);

  packVarToVec(input_vars, num_inputs, dP.inputs); // Copy from application
                                                   // space to library space
  // When true we will use HPAC Model for this output
  _user_fn_(args);
  packVarToVec(output_vars, num_outputs, dP.outputs);
}

enum class TensorsFound : char { NONE = 0, OUTPUT, INPUT, BOTH };

bool is_ml(MLType type) {
  return type < MLType::ML_END;
}

struct ml_argdesc_t {
  void (*accurateFN)(void *);
  void *accurateFN_arg;
  const char *region_name;
  TensorsFound have_tensors;
  approx_var_info_t *input_vars;
  approx_var_info_t *output_vars;
  std::vector<void *> ipts;
  std::vector<void *> opts;
};

void ml_infer(ml_argdesc_t &arg) {
  internal_repr_metadata_t *ipt_metadata = nullptr;
  internal_repr_metadata_t *opt_metadata = nullptr;

  switch(arg.have_tensors) {
    case TensorsFound::NONE:
      RTEnv.Model.evaluate(static_cast<ApproxType>(arg.input_vars[0].data_type),
                           arg.input_vars[0].num_elem, arg.ipts, arg.opts);
      break;
    case TensorsFound::INPUT:
      std::cerr << "Input only not supported yet\n";
      arg.accurateFN(arg.accurateFN_arg);
      // ipt_metadata = static_cast<internal_repr_metadata_t *>(input_vars[0].ptr);
      // RTEnv.Model.evaluate(static_cast<ApproxType>(input_vars[0].data_type),
                          //  input_vars[0].num_elem, ipt_metadata->Tensors[0], opts);
      break;
    case TensorsFound::OUTPUT:
      std::cerr << "Output only not supported yet\n";
      arg.accurateFN(arg.accurateFN_arg);
      // RTEnv.Model.evaluate(static_cast<ApproxType>(output_vars[0].data_type),
                          //  output_vars[0].num_elem, ipts, opts);
      break;
    case TensorsFound::BOTH:
      ipt_metadata = static_cast<internal_repr_metadata_t *>(arg.input_vars[0].ptr);
      opt_metadata = static_cast<internal_repr_metadata_t *>(arg.output_vars[0].ptr);
      RTEnv.Model.evaluate(static_cast<ApproxType>(arg.input_vars[0].data_type),
                           *ipt_metadata, *opt_metadata);
      break;
  }
}

void ml_offline_train(ml_argdesc_t &arg) {
  internal_repr_metadata_t *ipt_metadata = nullptr;
  internal_repr_metadata_t *opt_metadata = nullptr;

  switch(arg.have_tensors) {
    case TensorsFound::NONE:
      RTEnv.Model.evaluate(static_cast<ApproxType>(arg.input_vars[0].data_type),
                           arg.input_vars[0].num_elem, arg.ipts, arg.opts);
      break;
    case TensorsFound::INPUT:
      std::cerr << "Input only not supported yet\n";
      arg.accurateFN(arg.accurateFN_arg);
      // ipt_metadata = static_cast<internal_repr_metadata_t *>(input_vars[0].ptr);
      // RTEnv.Model.evaluate(static_cast<ApproxType>(input_vars[0].data_type),
                          //  input_vars[0].num_elem, ipt_metadata->Tensors[0], opts);
      break;
    case TensorsFound::OUTPUT:
      std::cerr << "Output only not supported yet\n";
      arg.accurateFN(arg.accurateFN_arg);
      // RTEnv.Model.evaluate(static_cast<ApproxType>(output_vars[0].data_type),
                          //  output_vars[0].num_elem, ipts, opts);
      break;
    case TensorsFound::BOTH:
      ipt_metadata = static_cast<internal_repr_metadata_t *>(arg.input_vars[0].ptr);
      opt_metadata = static_cast<internal_repr_metadata_t *>(arg.output_vars[0].ptr);

      auto ipt = ipt_metadata->get_wrapped_tensor(0).perform_indirection();
      EventRecorder::GPUEvent idtohost = EventRecorder::CreateGPUEvent("Input to CPU");
      EventRecorder::GPUEvent ihtodisk = EventRecorder::CreateGPUEvent("Input to Disk");
      EventRecorder::GPUEvent odtohost = EventRecorder::CreateGPUEvent("Output to CPU");
      EventRecorder::GPUEvent ohtodisk = EventRecorder::CreateGPUEvent("Output to Disk");
      idtohost.recordStart();
      ipt = TensorImpl::to(ipt, TensorImpl::CPU);
      idtohost.recordEnd();

      auto regionAddr = RTEnv.db->InstantiateRegion((uintptr_t) arg.accurateFN, arg.region_name, 25);
      HDF5DB *db = static_cast<HDF5DB *>(RTEnv.db);

      ihtodisk.recordStart();
      db->TensorToDB(regionAddr, ipt, ipt_metadata->underlying_type);
      ihtodisk.recordEnd();
      
      // this breaks our abstract tensor assumption,
      // but temporary while we measure.
      auto ipt_total_bytes = ipt.numel() * ipt.itemsize();

      arg.accurateFN(arg.accurateFN_arg);

      auto opt_tens = opt_metadata->update_from_memory();
      auto opt_total_bytes = opt_tens.numel() * opt_tens.itemsize();

      odtohost.recordStart();
      opt_tens = TensorImpl::to(opt_tens, TensorImpl::CPU);
      odtohost.recordEnd();

      ohtodisk.recordStart();
      db->TensorToDB(regionAddr, opt_tens, opt_metadata->underlying_type);
      ohtodisk.recordEnd();

      EventRecorder::LogEvent(idtohost);
      EventRecorder::LogEvent(ihtodisk);
      std::cout << "EVENT Input bytes: " << ipt_total_bytes << std::endl;
      EventRecorder::LogEvent(odtohost);
      EventRecorder::LogEvent(ohtodisk);
      std::cout << "EVENT Output bytes: " << opt_total_bytes << std::endl;
      break;
  }
}

void ml_invoke(MLType type, void (*accurateFN)(void *), void *arg,
               const char *region_name, void *inputs, int num_inputs,
               void *outputs, int num_outputs) {
  approx_var_info_t *input_vars = (approx_var_info_t *)inputs;
  approx_var_info_t *output_vars = (approx_var_info_t *)outputs;

  TensorsFound have_tensors = TensorsFound::NONE;

  if(region_name == nullptr){
    region_name = "unknown";
  }

    if(input_vars[0].is_tensor) {
      assert(num_inputs == 1 && "Only one tensor input is supported");
      have_tensors = TensorsFound::INPUT;
    }
    if(output_vars[0].is_tensor) {
      if(have_tensors == TensorsFound::INPUT) {
        have_tensors = TensorsFound::BOTH;
      } else {
        have_tensors = TensorsFound::OUTPUT;
      }
    }

    std::vector<void *> ipts;
    std::vector<void *> opts;

    if (have_tensors != TensorsFound::NONE) {
      ipts.reserve(num_inputs);
      opts.reserve(num_outputs);

      for (int i = 0; i < num_inputs; i++) {
        ipts.push_back(input_vars[i].ptr);
      }
      for (int i = 0; i < num_outputs; i++) {
        opts.push_back(output_vars[i].ptr);
      }
    }

    ml_argdesc_t ml_arg = {accurateFN, arg, region_name, have_tensors,
                           input_vars, output_vars, ipts, opts};

  if(type == ML_INFER) {
    ml_infer(ml_arg);
  } else if(type == ML_ONLINETRAIN) {
    std::cerr << "Online training not supported yet\n";
    accurateFN(arg);
  } else if(type == ML_OFFLINETRAIN) {
    ml_offline_train(ml_arg);
  } else {
    std::cerr << "Unknown ML type\n";
    accurateFN(arg);
  }
}

void __approx_exec_call(void (*accurateFN)(void *), void (*perfoFN)(void *),
                        void *arg, bool cond, const char *region_name,
                        void *perfoArgs, int memo_type, int petru_type,
                        int ml_type, void *inputs,
                        int num_inputs, void *outputs, int num_outputs) {
  approx_perfo_info_t *perfo = (approx_perfo_info_t *)perfoArgs;
  approx_var_info_t *input_vars = (approx_var_info_t *)inputs;
  approx_var_info_t *output_vars = (approx_var_info_t *)outputs;

  TensorsFound have_tensors = TensorsFound::NONE;

  if (petru_type & PETRUBATE_IN){
    petrubate(accurateFN, input_vars, num_inputs, region_name);
  }

  if ( perfoFN ){
      perforate(accurateFN, perfoFN, arg, input_vars, num_inputs, output_vars, num_outputs, RTEnv.getExecuteBoth());
  } else if (memo_type == MEMO_IN) {
    memoize_in(accurateFN, arg, input_vars, num_inputs, output_vars,
               num_outputs, RTEnv.getExecuteBoth(), RTEnv.tableSize, RTEnv.threshold );
  } else if (memo_type == MEMO_OUT) {
    memoize_out(accurateFN, arg, output_vars, num_outputs);
  } 
  else if (is_ml((MLType) ml_type)){
    ml_invoke((MLType) ml_type, accurateFN, arg, region_name, inputs, num_inputs, outputs, num_outputs);
  } else if(petru_type & PETRUBATE_OUT){
    petrubate(accurateFN, output_vars, num_outputs, region_name);
  } else {
    std::cerr << "Unknown execution type\n";
    accurateFN(arg);
  }
}


const float approx_rt_get_percentage(){
  return RTEnv.perfoRate;
}

const int approx_rt_get_step(){
  return RTEnv.perfoStep;
}
