#include <cfloat>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <stack>
#include <limits>
#include <cmath>
#include <cstring>
#include <omp.h>

#include "approx_data_util.h"
#include "approx_internal.h"
#include "thread_storage.h"

#include <chrono> 
using namespace std::chrono; 

using namespace std;

/*
class Profiler{
    std::unordered_map<const char *, std::pair<int,double>> profileTime;
    public:
    Profiler(){}
    ~Profiler();
    void storeData(const char *, double time);
};

Profiler profiler; 

Profiler *getProfiler(){
    return &profiler;
}

Profiler::~Profiler(){
    for (auto v: profileTime){
        std::cout<< "PROFILE: " << v.first << " : " << v.second.second / (double) v.second.first   <<std::endl;
    }
}

void Profiler::
storeData(const char *name, double time){
    auto elem = profileTime.find(name);
    if (elem == profileTime.end()){
        profileTime[name] = std::make_pair(1, time);
    }
    else{
        elem->second.first++;
        elem->second.second += time;
    }
    return;
}
*/

class MemoizeInput {
  public:
  float **inTable;
  float **outTable;
  float *iTemp;
  float *oTemp;
  double outAverage[2];
  double totalDiff;
  long totalInvocations;
  void (*accurate)(void *);
  int num_inputs;
  int num_outputs;
  int tSize;
  float threshold;
  int input_index, output_index;
  long accurately;
  long approximately;
  int iSize, oSize;
//  Profiler *profile;

  public:
  MemoizeInput(void (*acc)(void *), int num_inputs, int num_outputs,
               approx_var_info_t *inputs, approx_var_info_t *outputs, int tSize, float threshold)
      : accurate(acc), num_inputs(num_inputs), num_outputs(num_outputs),
        tSize(tSize), threshold(threshold), input_index(0), output_index(0), 
        accurately(0), approximately(0) {

//          profile=getProfiler();
    if (tSize == 0){
      printf("Should Never happen\n");
      exit(-1);
    }

    int iCols= 0;
    int oCols= 0;

    for (int i = 0; i <  num_inputs; i++){
      iCols+= inputs[i].num_elem;
    }


    for (int i = 0; i <  num_outputs; i++){
      oCols += outputs[i].num_elem;
    }

    iTemp = new float[iCols];
    oTemp = new float[oCols];
    inTable = create2DArray<float>(tSize, iCols);
    outTable = create2DArray<float>(tSize, oCols);
    iSize = iCols;
    oSize = oCols;
    outAverage[0] = 0;
    outAverage[1] = 0;
    totalDiff = 0;
    totalInvocations = 0;
}

~MemoizeInput(){
  delete2DArray(inTable);
  delete2DArray(outTable);
  double MAPE =0.0f;

  if (totalInvocations != 0){
    outAverage[0] = outAverage[0]/(float)(totalInvocations*oSize);
    outAverage[1] = outAverage[1]/(float)(totalInvocations*oSize);
    MAPE = fabs(outAverage[0]-outAverage[1])/fabs(outAverage[0]);
  }

  delete [] iTemp;
  delete [] oTemp;

  if (totalInvocations != 0){
    cout << "REGION_ERROR:"<<  MAPE << endl;
  }

  cout << "APPROX:"
    << (double)approximately / (double)(accurately + approximately)
    << ":" << approximately<< ":" << accurately << endl; 
}

void convertFrom(approx_var_info_t *values, int num_values, float *vector){
  for (int i = 0; i < num_values; i++){
    convertToFloat(vector, values[i].ptr, values[i].num_elem, (ApproxType)values[i].data_type);
    vector += values[i].num_elem;
  }
}

void convertTo(approx_var_info_t *values, int num_values, float *vector){
  for (int i = 0; i < num_values; i++){
    convertFromFloat(values[i].ptr, vector, values[i].num_elem, (ApproxType)values[i].data_type);
    vector += values[i].num_elem;
  }
}

void execute(void *args, approx_var_info_t *inputs, approx_var_info_t *outputs){
//    auto start_time = high_resolution_clock::now();
    convertFrom(inputs, num_inputs, iTemp);
//    auto stop_time = high_resolution_clock::now();
//    auto duration = duration_cast<microseconds>(stop_time - start_time); 
//    double time = duration.count(); 
//    profile->storeData("ConvertFrom", time);
    float minDist = std::numeric_limits<float>::max(); 
    int index = -1;
    // Iterate in table and find closest input value
//    start_time = high_resolution_clock::now();
    for (int i = 0; i < input_index; i++){
      float *temp = inTable[i];
      float dist = 0.0f;
      for (int j = 0; j < iSize; j++){
        if (temp[j] != 0.0f)
          dist += fabs((iTemp[j] - temp[j])/temp[j]);
         else
          dist += fabs((iTemp[j] - temp[j]));
      }
      dist = dist/(float)iSize;
      if (dist < minDist){
        minDist = dist;
        index = i;
        if (minDist < threshold)
          break;
      }
    }

    if (minDist > threshold){
      index = -1;
    }

//    stop_time = high_resolution_clock::now();
//    duration = duration_cast<microseconds>(stop_time - start_time); 
//    time = duration.count(); 
//    profile->storeData("HashTable", time);

    if (index == -1){
      // I need to execute accurately
//      start_time = high_resolution_clock::now();
      accurately++;
      accurate(args);
//      stop_time = high_resolution_clock::now();
//      duration = duration_cast<microseconds>(stop_time - start_time); 
//      time = duration.count(); 
//      profile->storeData("accurate", time);


//      start_time = high_resolution_clock::now();
      if (input_index < tSize ){
        std::memcpy(inTable[input_index], iTemp, sizeof(float)*iSize);
        convertFrom(outputs, num_outputs, outTable[input_index]);
        input_index +=1;
      }
//      stop_time = high_resolution_clock::now();
//      duration = duration_cast<microseconds>(stop_time - start_time); 
//      time = duration.count(); 
//      profile->storeData("create_table", time);
    }
    else{
      approximately++;
//      start_time = high_resolution_clock::now();
      convertTo(outputs, num_outputs, outTable[index]);
//      stop_time = high_resolution_clock::now();
//      duration = duration_cast<microseconds>(stop_time - start_time); 
//      time = duration.count(); 
//      profile->storeData("approximate", time);
    }
  }

  void execute_both(void *args, approx_var_info_t *inputs, approx_var_info_t *outputs){
    convertFrom(inputs, num_inputs, iTemp);
    //Execute accurate
    accurate(args);
    convertFrom(outputs,num_outputs, oTemp);
    
    float minDist = std::numeric_limits<float>::max(); 
    int index = -1;
    // Iterate in table and find closest input value
    for (int i = 0; i < input_index; i++){
      float *temp = inTable[i];
      float dist = 0.0f;
      for (int j = 0; j < iSize; j++){
        if (temp[j] != 0.0f)
          dist += fabs((iTemp[j] - temp[j])/temp[j]);
         else
          dist += fabs((iTemp[j] - temp[j]));
      }
      dist = dist/(float)iSize;
      if (dist < minDist){
        minDist = dist;
        index = i;
        if (minDist < threshold)
          break;
      }
    }

    if (minDist > threshold){
      index = -1;
    }

    for (int i = 0; i < oSize; i++){
      outAverage[0] += oTemp[i];
    }

    double diff = 0.0;
    if (index == -1){
      // I need to execute accurately
      for (int i = 0; i < oSize; i++){
        outAverage[1] += oTemp[i];
      }
      accurately++;
      if (input_index < tSize ){
        std::memcpy(inTable[input_index], iTemp, sizeof(float)*iSize);
        std::memcpy(outTable[input_index], oTemp, sizeof(float)*oSize);
        convertFrom(outputs, num_outputs, outTable[input_index]);
        input_index +=1;
      }
    }
    else{
      approximately++;
      for (int i = 0; i < oSize; i++){
        outAverage[1] += outTable[index][i];
      }
      convertTo(outputs, num_outputs, outTable[index]);
    }
    totalInvocations++;
  }
};

MemoPool<MemoizeInput> inputMemo;

void memoize_in(void (*accurate)(void *), void *arg, approx_var_info_t *inputs,
                int num_inputs, approx_var_info_t *outputs, int num_outputs, bool ExecBoth, int tSize, float threshold) {
  int threadId = 0;
  MemoizeInput *curr;
  if (omp_in_parallel()){
    threadId = omp_get_thread_num();
  }

  curr = inputMemo.findMemo(threadId, (unsigned long)accurate);
  if (!curr){ 
    curr = inputMemo.addNew(threadId, new MemoizeInput(accurate, num_inputs, num_outputs, inputs, outputs,tSize, threshold));
  }

  if (ExecBoth)
    curr->execute_both(arg, inputs, outputs);
  else
    curr->execute(arg, inputs, outputs);
}
