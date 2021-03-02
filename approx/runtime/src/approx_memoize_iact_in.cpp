#include <cfloat>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <stack>
#include <limits>
#include <cmath>
#include <cstring>

#include <approx_data_util.h>
#include <approx_internal.h>

using namespace std;

#define MEMORY_BLOCKS 1024

#define MAX_REGIONS 20
#define NOTFOUND -1

#ifdef __OLD_MEMO__
class MemoizeInput {

  typedef enum : uint8_t {
    INIT,
    CREATE,
    COPY_IN,
    COPY_OUT,
    SEARCH,
    INSERT,
    ACCURATE,
    APPROXIMATE,
    END
  } CODE_REGIONS;

  struct KeyData {
    MemoizeInput *parent;
    mutable uint8_t *ptr;
    mutable approx_var_info_t *inputs;
    bool operator==(const KeyData &other) const {
      const size_t *offsets = parent->input_offsets;
      const ApproxType *types = parent->input_types;
      const size_t *num_elements = parent->input_shape;
      const real_t threshold = parent->threshold;
      for (int i = 0; i < parent->num_inputs; i++) {
        void *this_ptr =
            (inputs == nullptr) ? (void *)&ptr[offsets[i]] : inputs[i].ptr;
        void *other_ptr = (other.inputs == nullptr)
                              ? (void *)&other.ptr[offsets[i]]
                              : other.inputs[i].ptr;
        if (rel_error_larger(this_ptr, other_ptr, num_elements[i], types[i],
                             threshold))
          return false;
      }
      return true;
    }
  };

  class KeyDataHasher {
  public:
    std::size_t operator()(const KeyData &key) const {
      size_t seed = 5381;
      const size_t *offsets = key.parent->input_offsets;
      const size_t *num_elements = key.parent->input_shape;
      const size_t *sz_types = key.parent->input_sz_type;

      for (int i = 0; i < key.parent->num_inputs; i++) {
        size_t bytes = sz_types[i] * num_elements[i];
        uint8_t *ptr = (key.inputs == nullptr) ? &key.ptr[offsets[i]]
                                               : (uint8_t *)key.inputs[i].ptr;
        for (size_t j = 0; j < bytes; j++) {
          seed = ((seed << 5) + seed) + ptr[j];
        }
      }
      return seed;
    }
  };

  typedef std::unordered_map<KeyData, uint8_t *, KeyDataHasher> KeyDataHashMap;
  typedef std::unordered_map<KeyData, uint8_t *, KeyDataHasher>::iterator
      KeyDataHashMapIterator;

public:
  std::stack<uint8_t *> input_memory_pool;
  std::stack<uint8_t *> output_memory_pool;
  void (*accurate)(void *);
  int num_inputs;
  int num_outputs;
  real_t threshold;

  uint8_t *input_memory;
  uint8_t *output_memory;
  size_t input_index;
  size_t output_index;

  size_t *input_shape;
  ApproxType *input_types;
  size_t *input_offsets;
  size_t *input_sz_type;
  size_t total_input_size;
  int *output_shape;
  ApproxType *output_types;
  size_t *output_offsets;
  size_t total_output_size;
  KeyDataHashMap storage;

  /// Stat. Counts how many invocations where performed accurately.
  int accurately;
  /// Stat. Counts how many invocations where performed approximately.
  int approximately;
  int counter;

public:
  MemoizeInput(void (*acc)(void *), int num_inputs, int num_outputs,
               approx_var_info_t *inputs, approx_var_info_t *outputs)
      : accurate(acc), num_inputs(num_inputs), num_outputs(num_outputs),
        input_index(0), output_index(0), accurately(0), approximately(0) {
    int i;
    threshold = 0.0;
    const char *env_p = std::getenv("THRESHOLD");
    if (env_p) {
      threshold = atof(env_p);
    }

    input_shape = new size_t[num_inputs];
    input_types = new ApproxType[num_inputs];
    input_offsets = new size_t[num_inputs];
    input_sz_type = new size_t[num_inputs];

    output_shape = new int[num_outputs];
    output_types = new ApproxType[num_outputs];
    output_offsets = new size_t[num_outputs];

    size_t curr_offset = 0;

    for (i = 0; i < num_inputs; i++) {
      input_shape[i] = inputs[i].num_elem;
      input_types[i] = (ApproxType)inputs[i].data_type;

      input_sz_type[i] = inputs[i].sz_elem;

      size_t rem = curr_offset % inputs[i].sz_elem;
      if (rem != 0)
        curr_offset += inputs[i].sz_elem - rem;

      input_offsets[i] = curr_offset;
      curr_offset += input_shape[i] * inputs[i].sz_elem;
    }

    size_t rem = curr_offset % sizeof(uint64_t);
    if (rem != 0) {
      curr_offset += sizeof(uint64_t) - rem;
    }
    total_input_size = curr_offset;
    size_t elements = total_input_size;
    input_memory = new uint8_t[elements * MEMORY_BLOCKS]();
    input_memory_pool.push(input_memory);

    curr_offset = 0;
    for (i = 0; i < num_outputs; i++) {
      output_shape[i] = outputs[i].num_elem;
      output_types[i] = (ApproxType)outputs[i].data_type;
      size_t rem = curr_offset % outputs[i].sz_elem;
      if (rem != 0)
        curr_offset += outputs[i].sz_elem - rem;

      output_offsets[i] = curr_offset;
      curr_offset += output_shape[i] * outputs[i].sz_elem;
    }

    rem = curr_offset % sizeof(uint64_t);
    if (rem != 0) {
      curr_offset += sizeof(uint64_t) - rem;
    }
    total_output_size = curr_offset;
    elements = total_output_size;
    output_memory = new uint8_t[elements * MEMORY_BLOCKS]();
    output_memory_pool.push(output_memory);

    counter = 0;
  }

  ~MemoizeInput() {
    while (!input_memory_pool.empty()){
      delete[] input_memory_pool.top();
      input_memory_pool.pop();
    }

    while (!output_memory_pool.empty()){
      delete[] output_memory_pool.top();
      output_memory_pool.pop();
    }


    delete[] input_shape;
    delete[] input_types;
    delete[] input_offsets;
    delete[] input_sz_type;

    delete[] output_types;
    delete[] output_shape;
    delete[] output_offsets;

    cout << "APPROX:"
         << (double)approximately / (double)(accurately + approximately)
         << ":" << approximately<< ":" << accurately << endl;

  };

  uint8_t *copy_inputs(approx_var_info_t *inputs) {
    if (input_index + total_input_size > (total_input_size * MEMORY_BLOCKS)){
      input_memory = new uint8_t[total_input_size * MEMORY_BLOCKS]();
      input_memory_pool.push(input_memory);
      input_index = 0;
    }
    uint8_t *ptr = &input_memory[input_index];
    input_index += total_input_size;
    for (int i = 0; i < num_inputs; i++) {
      copyData((void *)&ptr[input_offsets[i]], inputs[i].ptr, input_shape[i],
               input_types[i]);
    }
    return ptr;
  }

  uint8_t *copy_outputs(approx_var_info_t *outputs) {
    if (output_index + total_output_size> (total_output_size * MEMORY_BLOCKS)){
      output_memory= new uint8_t[total_output_size* MEMORY_BLOCKS]();
      output_memory_pool.push(output_memory);
      output_index= 0;
    }
    uint8_t *ptr = &output_memory[output_index];
    output_index += total_output_size;
    for (int i = 0; i < num_outputs; i++) {
      copyData((void *)&ptr[output_offsets[i]], (void *)outputs[i].ptr,
               output_shape[i], output_types[i]);
    }
    return ptr;
  }

  void copy_results(uint8_t *values, approx_var_info_t *outputs) {
    for (int i = 0; i < num_outputs; i++) {
      copyData(outputs[i].ptr, (void *)&(values[output_offsets[i]]),
               output_shape[i], output_types[i]);
    }
  }

  void execute(void *args, approx_var_info_t *inputs,
               approx_var_info_t *outputs) {

    KeyData new_values = {this, nullptr, inputs};
    auto ret = storage.insert({new_values, nullptr});

    if (ret.second == false) {
      copy_results((uint8_t *)ret.first->second, outputs);
      approximately++;
    } else {
      (*ret.first).first.ptr = copy_inputs(inputs);
      (*ret.first).first.inputs = nullptr;
      accurately++;
      accurate(args);
      uint8_t *output_ptr = copy_outputs(outputs);
      (*ret.first).second = output_ptr;
    }
    counter++;
  }
};

#else

class MemoizeInput {
  public:
  float **inTable;
  float **outTable;
  float *iTemp;
  float *oTemp;
  double outAverage[2];
  double totalDiff;
  long totalInvocations;
  int tSize;
  void (*accurate)(void *);
  int num_inputs;
  int num_outputs;
  float threshold;
  int input_index, output_index;
  int accurately;
  int approximately;
  int iSize, oSize;

  public:
  MemoizeInput(void (*acc)(void *), int num_inputs, int num_outputs,
               approx_var_info_t *inputs, approx_var_info_t *outputs)
      : accurate(acc), num_inputs(num_inputs), num_outputs(num_outputs),
        input_index(0), output_index(0), accurately(0), approximately(0) {
    threshold = 0.0;
    const char *env_p = std::getenv("THRESHOLD");
    if (env_p) {
      threshold = atof(env_p);
    }

    tSize = 0;
    env_p = std::getenv("TABLE_SIZE");
    if (env_p){
      tSize = atoi(env_p);
    }

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
    convertFrom(inputs, num_inputs, iTemp);
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

    if (index == -1){
      // I need to execute accurately
      accurately++;
      accurate(args);
      if (input_index < tSize ){
        std::memcpy(inTable[input_index], iTemp, sizeof(float)*iSize);
        convertFrom(outputs, num_outputs, outTable[input_index]);
        input_index +=1;
      }
    }
    else{
      approximately++;
      convertTo(outputs, num_outputs, outTable[index]);
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

#endif

int last_memoIn = 0;

MemoizeInput *memo_regions[MAX_REGIONS];

class GarbageCollectorMemoIn {
public:
  GarbageCollectorMemoIn(){};
  ~GarbageCollectorMemoIn() {
    for (int i = 0; i < last_memoIn; i++) {
      delete memo_regions[i];
    }
  }
};

GarbageCollectorMemoIn memoCleaner;

int memo_find_index(unsigned long Addr) {
  for (int i = 0; i < last_memoIn; i++) {
    if (((unsigned long)(memo_regions[i]->accurate) == Addr))
      return i;
  }
  return NOTFOUND;
}

void memoize_in(void (*accurate)(void *), void *arg, approx_var_info_t *inputs,
                int num_inputs, approx_var_info_t *outputs, int num_outputs, bool ExecBoth) {
  unsigned long Addr = (unsigned long)accurate;
  int curr_index = memo_find_index(Addr);
  if (curr_index == NOTFOUND) {
    if (last_memoIn >= MAX_REGIONS) {
      cout << "I Reached maximum memo_regions exiting\n";
      exit(0);
    }
    curr_index = last_memoIn;
    memo_regions[last_memoIn++] =
        new MemoizeInput(accurate, num_inputs, num_outputs, inputs, outputs);
  }
  if (ExecBoth)
    memo_regions[curr_index]->execute_both(arg, inputs, outputs);
  else
    memo_regions[curr_index]->execute(arg, inputs, outputs);
}
