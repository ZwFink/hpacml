//Copyright (c) Lawrence Livermore National Security, LLC and other AMS
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute

#ifndef __SURROGATE_HPP__
#define __SURROGATE_HPP__
#define __ENABLE_TORCH__ 1

#define NUM_ITEMS 4194304
#include <string>

#include <torch/script.h>  // One-stop header.
#include "database/database.h"

#include <cuda_runtime.h>
inline void DtoDMemcpy(void *dest, void *src, size_t nBytes)
{
  cudaMemcpy(dest, src, nBytes, cudaMemcpyDeviceToDevice);
}

inline void HtoHMemcpy(void *dest, void *src, size_t nBytes)
{
  std::memcpy(dest, src, nBytes);
}

inline void HtoDMemcpy(void *dest, void *src, size_t nBytes)
{
  cudaMemcpy(dest, src, nBytes, cudaMemcpyHostToDevice);
};

inline void DtoHMemcpy(void *dest, void *src, size_t nBytes)
{
  cudaMemcpy(dest, src, nBytes, cudaMemcpyDeviceToHost);
}

void print_shape(at::Tensor &t)
{
  std::cout << "Shape: ";
  for (int i = 0; i < t.dim(); i++)
  {
    std::cout << t.size(i) << " ";
  }
  std::cout << std::endl;
}

class CPUExecutionPolicy {
  public:
    c10::Device device = c10::Device("cpu");

    static inline void transferToDevice(void *dest, void *src, size_t nBytes)
    {
      HtoHMemcpy(dest, src, nBytes);
    }

    static inline void transferFromDevice(void *dest, void *src, size_t nBytes)
    {
      HtoHMemcpy(dest, src, nBytes);
    }
};

// TODO: We may want to later differentiate between 
// CUDA and other GPU devices
class GPUExecutionPolicy {
  public:
    c10::Device device = c10::Device("cuda");
    static inline void transferToDevice(void *dest, void *src, size_t nBytes)
    {
      HtoDMemcpy(dest, src, nBytes);
    }
    static inline void transferFromDevice(void *dest, void *src, size_t nBytes)
    {
      DtoDMemcpy(dest, src, nBytes);
    }
};

template <typename TypeInValue> class TensorTranslator {
  public:
    at::Tensor &tensor;
    size_t insert_index = 0;
    TensorTranslator(at::Tensor &tensor) : tensor(tensor) {tensor = tensor.pin_memory();}
    TensorTranslator() = delete;
    TensorTranslator(at::Tensor &&) = delete;

    bool isFull() { return insert_index == NUM_ITEMS; }

    // TODO: This can probably be optimized if we change the layout of 
    // this to match the other case (i.e., insert when it's in column-major, then transpose)
    at::Tensor arrayToTensor(long numRows, long numCols, TypeInValue **array) {
      auto tensorOptions = tensor.options();
      for (int i = 0; i < numCols; i++) {
        auto column =
            tensor.select(1, i).slice(0, insert_index, insert_index + numRows);
        auto data = reinterpret_cast<TypeInValue *>(array[i]);
        column.copy_(torch::from_blob(data, {numRows}, tensorOptions), false);
      }
      insert_index += numRows;
      return tensor;
    }

    at::Tensor prepareForInference(at::Tensor& t)
    {
      return t;
    }

    void reset()
    {
      insert_index = 0;
    }

    void tensorToArray(at::Tensor tensor, long numRows, long numCols,
                       TypeInValue **array) {}
};

template <typename TypeInValue, bool use_pinned=true>
class MemcpyTensorTranslator : public TensorTranslator<TypeInValue> {
  public:
    TypeInValue *input_data;
    torch::Tensor tensor;
    MemcpyTensorTranslator(at::Tensor &tensor) : TensorTranslator<TypeInValue>{tensor}
    {
        this->input_data = nullptr;
        if (use_pinned) {
        cudaMallocHost((void **)&this->input_data,
                       NUM_ITEMS * 5 * sizeof(TypeInValue));
        } else {
        this->input_data = new TypeInValue[NUM_ITEMS * 5];
        }
    }

    at::Tensor &arrayToTensor(long numRows, long numCols, TypeInValue **array) {
      // TypeInValue *input_data = this->tensor.template data_ptr<TypeInValue>();
      for (int i = 0; i < numCols; i++) {
        std::memcpy(this->input_data +
                        (i*NUM_ITEMS + this->insert_index),
                    reinterpret_cast<TypeInValue *>(array[i]),
                    numRows * sizeof(TypeInValue));
      }
      this->insert_index += numRows;
      if(this->insert_index == NUM_ITEMS)
        this->tensor = torch::from_blob(this->input_data, {numCols, NUM_ITEMS}, torch::kFloat64);

      return this->tensor;
    }

    void reset()
    {
      this->insert_index = 0;
    }

    void tensorToArray(at::Tensor tensor, long numRows, long numCols,
                       TypeInValue **array) {}

    at::Tensor prepareForInference(at::Tensor& t)
    {
      return t.transpose(1, 0);
    }

    ~MemcpyTensorTranslator()
    {
      if(use_pinned)
        cudaFreeHost(this->input_data);
      else
        delete[] this->input_data;
    }
};

template <typename TypeInValue>
class CatTensorTranslator : public TensorTranslator<TypeInValue> {
  public:
    std::vector<torch::Tensor> allocatedTensors;
    torch::Tensor tensor = torch::empty({0, 5}, torch::kFloat64);
    CatTensorTranslator(at::Tensor &tensor)
        : TensorTranslator<TypeInValue>{tensor} {
          for(int i = 0; i < 5; i++)
            allocatedTensors.push_back(torch::empty({NUM_ITEMS,1}, torch::kFloat64));
        }

    at::Tensor &arrayToTensor(long numRows, long numCols, TypeInValue **array) {
      for (int i = 0; i < numCols; i++) {
        at::Tensor temp = torch::from_blob((TypeInValue *)array[i],
                                           {numRows, 1}, torch::kFloat64);

        allocatedTensors[i].narrow(0, this->insert_index, numRows).copy_(temp);
      }

      auto tensor = torch::nested::as_nested_tensor(allocatedTensors);
      this->tensor = tensor;
      this->insert_index += numRows;
      return this->tensor;
    }

    void reset()
    {
      this->tensor = torch::empty({0, 5}, torch::kFloat64);
      this->insert_index = 0;
    }

    void tensorToArray(at::Tensor tensor, long numRows, long numCols,
                       TypeInValue **array) {}

    at::Tensor prepareForInference(at::Tensor& t)
    {
      return t;
    }

};

//! ----------------------------------------------------------------------------
//! An implementation for a surrogate model
//! ----------------------------------------------------------------------------
template <typename ExecutionPolicy, typename TensorTranslator,
typename TypeInValue>
class SurrogateModel : public ExecutionPolicy
{

  static_assert(std::is_floating_point<TypeInValue>::value,
                "SurrogateModel supports floating-point values (floats, "
                "doubles, or long doubles) only!");

private:
  const std::string model_path;
  const bool is_cpu;
  at::IntArrayRef input_shape;
  at::IntArrayRef output_shape;
  at::Tensor input_tensor;
  at::Tensor output_tensor;
  std::unique_ptr<TensorTranslator> translator;


#ifdef __ENABLE_TORCH__
  // -------------------------------------------------------------------------
  // variables to store the torch model
  // -------------------------------------------------------------------------
  torch::jit::script::Module module;
  c10::TensorOptions tensorOptions;

  inline void tensorToArray(at::Tensor tensor,
                            long numRows,
                            long numCols,
                            TypeInValue** array)
  {
    // Transpose to get continuous memory and
    // perform single memcpy.
    tensor = tensor.transpose(1, 0);
      for (long j = 0; j < numCols; j++) {
        auto tmp = tensor[j].contiguous();
        TypeInValue* ptr = tmp.data_ptr<TypeInValue>();
        ExecutionPolicy::transferFromDevice(array[j], ptr,
                                            sizeof(TypeInValue) * numRows);
      }
    }

  // -------------------------------------------------------------------------
  // loading a surrogate model!
  // -------------------------------------------------------------------------
  void _load_torch(const std::string& model_path,
                   const c10::Device& device,
                   at::ScalarType dType)
  {
    try {
      module = torch::jit::load(model_path);
      module.to(device);
      module.to(dType);
      module.eval();
      tensorOptions = torch::TensorOptions().dtype(dType).pinned_memory(true);
    } catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
    }
  }

  template <typename T,
            std::enable_if_t<std::is_same<T, double>::value>* = nullptr>
  inline void _load(const std::string& model_path,
                    const c10::Device& device)
  {
    _load_torch(model_path, device, torch::kFloat64);
  }

  template <typename T,
            std::enable_if_t<std::is_same<T, float>::value>* = nullptr>
  inline void _load(const std::string& model_path,
                    const c10::Device& device)
  {
    _load_torch(model_path, device, torch::kFloat64);
  }

  // -------------------------------------------------------------------------
  // evaluate a torch model
  // -------------------------------------------------------------------------
  inline void _evaluate(long num_elements,
                        long num_in,
                        size_t num_out,
                        TypeInValue** inputs,
                        TypeInValue** outputs)
  {

    auto input = translator->arrayToTensor(num_elements, num_in, inputs);
    if(translator->isFull())
    {
      input = input.to(ExecutionPolicy::device, true);
      input = translator->prepareForInference(input);

      at::Tensor output = module.forward({input}).toTensor();
      cudaDeviceSynchronize();
      // tensorToArray(output, num_elements, num_out, outputs);
      // output = output.to(at::kCPU);
      translator->reset();
      // sync the output tensor

    }
  }

#else
  template <typename T>
  inline void _load(const std::string& model_path,
                    const std::string& device_name)
  {
  }

  inline void _evaluate(long num_elements,
                        long num_in,
                        size_t num_out,
                        TypeInValue** inputs,
                        TypeInValue** outputs)
  {
  }

#endif

  // -------------------------------------------------------------------------
  // public interface
  // -------------------------------------------------------------------------
public:
  SurrogateModel(const char* model_path, at::IntArrayRef &&ipt_shape, at::IntArrayRef &&opt_shape, bool is_cpu = true)
      : model_path(model_path), is_cpu(is_cpu), input_shape(ipt_shape), output_shape(opt_shape)
  {
    _load<TypeInValue>(model_path, ExecutionPolicy::device);
    input_tensor = at::empty(input_shape, at::TensorOptions().dtype(torch::kFloat64));
    output_tensor = at::empty(output_shape, at::TensorOptions().dtype(torch::kFloat64).device(ExecutionPolicy::device));
    translator = std::make_unique<TensorTranslator>(input_tensor);
  }

  inline void evaluate(long num_elements,
                       long num_in,
                       size_t num_out,
                       TypeInValue** inputs,
                       TypeInValue** outputs)
  {
    _evaluate(num_elements, num_in, num_out, inputs, outputs);
  }

  inline void evaluate(long num_elements,
                       std::vector<const TypeInValue*> inputs,
                       std::vector<TypeInValue*> outputs)
  {
    _evaluate(num_elements,
              inputs.size(),
              outputs.size(),
              static_cast<const TypeInValue**>(inputs.data()),
              static_cast<TypeInValue**>(outputs.data()));
  }
};

#endif
