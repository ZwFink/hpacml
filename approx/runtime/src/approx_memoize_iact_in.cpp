#include <cfloat>
#include <iostream>
#include <memory>
#include <unordered_map>

#include <approx_data_util.h>
#include <approx_internal.h>

using namespace std;

#define MAX_REGIONS 20
#define NOTFOUND -1

struct KeyData {
  uint8_t *ptr;
  size_t *offsets;
  ApproxType *types;
  int *num_elements;
  int num_vars;
  real_t threshold;
  size_t size;
  bool operator==(const KeyData &other) const {
    for (int i = 0; i < num_vars ; i++) {
      if (rel_error_larger((void *)&ptr[offsets[i]],
                           (void *)&other.ptr[offsets[i]], num_elements[i],
                           types[i], threshold))
        return false;
    }
    return true;
  }
};

class KeyDataHasher {
public:
  std::size_t operator()(const KeyData &key) const {
    size_t seed = 5381;
    for (size_t i = 0; i < key.size; i++) {
      seed = ((seed << 5) + seed) + key.ptr[i];
    }
    return seed;
  }
};

class MemoizeInput {
  typedef std::unordered_map<KeyData, uint8_t*, KeyDataHasher> KeyDataHashMap;
  typedef std::unordered_map<KeyData, uint8_t*, KeyDataHasher>::iterator
      KeyDataHashMapIterator;

public:
  void (*accurate)(void *);
  int num_inputs;
  int num_outputs;
  real_t threshold;

  int *input_shape;
  ApproxType *input_types;
  size_t *input_offsets;
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

public:
  MemoizeInput(void (*acc)(void *), int num_inputs, int num_outputs,
               approx_var_info_t *inputs, approx_var_info_t *outputs)
      : accurate(acc), num_inputs(num_inputs), num_outputs(num_outputs),
        accurately(0), approximately(0) {
    int i;
    input_shape = new int[num_inputs];
    input_types = new ApproxType[num_inputs];
    input_offsets = new size_t[num_inputs];

    output_shape = new int[num_outputs];
    output_types = new ApproxType[num_outputs];
    output_offsets = new size_t[num_outputs];

    size_t curr_offset = 0;

    for (i = 0; i < num_inputs; i++) {
      input_shape[i] = inputs[i].num_elem;
      input_types[i] = (ApproxType)inputs[i].data_type;
      size_t rem = curr_offset % inputs[i].sz_elem;
      curr_offset += inputs[i].sz_elem - rem;

      input_offsets[i] = curr_offset;
      curr_offset += input_shape[i] * inputs[i].sz_elem;
    }

    total_input_size = curr_offset;

    curr_offset = 0;
    for (i = 0; i < num_outputs; i++) {
      output_shape[i] = outputs[i].num_elem;
      output_types[i] = (ApproxType)outputs[i].data_type;
      size_t rem = curr_offset % outputs[i].sz_elem;
      curr_offset += outputs[i].sz_elem - rem;

      output_offsets[i] = curr_offset;
      curr_offset += output_shape[i] * outputs[i].sz_elem;
    }

    total_output_size = curr_offset;
  };
  ~MemoizeInput() {
    for ( auto v : storage){
      delete[] v.first.ptr;
      delete[] v.second;
    }

    delete [] input_shape;
    delete [] input_types;
    delete [] input_offsets;

    delete [] output_types;
    delete [] output_shape;
    delete [] output_offsets;

    cout << "APPROX:"
         << (double)approximately / (double)(accurately + approximately)
         << endl;
  };

  uint8_t* copy_inputs(approx_var_info_t *inputs) {
    uint8_t *ptr = new uint8_t[total_input_size]();
    for (int i = 0; i < num_inputs; i++) {
      memcpy((void *)&ptr[input_offsets[i]], (void *)inputs[i].ptr,
             inputs[i].sz_elem * inputs[i].num_elem);
    }
    return ptr;
  }

  uint8_t *copy_outputs(approx_var_info_t *outputs) {
    uint8_t *ptr = new uint8_t[total_output_size]();
    for (int i = 0; i < num_outputs; i++) {
      memcpy((void *)&ptr[output_offsets[i]], (void *)outputs[i].ptr,
             outputs[i].sz_elem * outputs[i].num_elem);
    }
    return ptr;
  }

  void copy_results(uint8_t *values, approx_var_info_t *outputs) {
    for (int i = 0; i < num_outputs; i++) {
      memcpy(outputs[i].ptr, (void *)&(values[output_offsets[i]]),
             outputs[i].sz_elem * outputs[i].num_elem);
    }
  }

  void execute(void *args, approx_var_info_t *inputs,
               approx_var_info_t *outputs) {
    uint8_t *ptr = copy_inputs(inputs);
    KeyData new_values = {ptr, input_offsets, input_types,
                          input_shape,          num_inputs,    threshold,
                          total_input_size};
    KeyDataHashMapIterator it = storage.find(new_values);
    if (it != storage.end()) {
      copy_results((uint8_t*) it->second, outputs);
      delete[] ptr;
      approximately++;
    } else {
      accurately++;
      accurate(args);
      uint8_t *output_ptr = copy_outputs(outputs);
      storage.insert({new_values, output_ptr});
    }
  }
};

int last_memoIn = 0;

MemoizeInput *memo_regions[MAX_REGIONS];

class GarbageCollectorMemoIn {
public:
  GarbageCollectorMemoIn(){};
  ~GarbageCollectorMemoIn() {
    for (int i = 0; i < last_memoIn; i++) {
      printf("Deleting memory region\n");
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
                int num_inputs, approx_var_info_t *outputs, int num_outputs) {
  unsigned long Addr = (unsigned long)accurate;
  int curr_index = memo_find_index(Addr);
  if (curr_index == NOTFOUND) {
    if (last_memoIn >= MAX_REGIONS) {
      std::cout << "I Reached maximum memo_regions exiting\n";
      exit(0);
    }
    curr_index = last_memoIn;
    memo_regions[last_memoIn++] =
        new MemoizeInput(accurate, num_inputs, num_outputs, inputs, outputs);
  }
  memo_regions[curr_index]->execute(arg, inputs, outputs);
}
