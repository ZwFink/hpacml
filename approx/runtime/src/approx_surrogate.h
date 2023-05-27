// Copyright (c) Lawrence Livermore National Security, LLC and other AMS
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute

#ifndef __SURROGATE_HPP__
#define __SURROGATE_HPP__
#define __ENABLE_TORCH__ 1

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


//! ----------------------------------------------------------------------------
//! An implementation for a surrogate model
//! ----------------------------------------------------------------------------
template <typename TypeInValue>
class SurrogateModel
{

  static_assert(std::is_floating_point<TypeInValue>::value,
                "SurrogateModel supports floating-point values (floats, "
                "doubles, or long doubles) only!");

private:
  const std::string model_path;
  const bool is_cpu;


#ifdef __ENABLE_TORCH__
  // -------------------------------------------------------------------------
  // variables to store the torch model
  // -------------------------------------------------------------------------
  torch::jit::script::Module module;
  c10::TensorOptions tensorOptions;


  // -------------------------------------------------------------------------
  // conversion to and from torch
  // -------------------------------------------------------------------------
  inline at::Tensor arrayToTensor(long numRows,
                                  long numCols,
                                  TypeInValue** array)
  {
    c10::SmallVector<at::Tensor, 8> Tensors;
    for (int i = 0; i < numCols; i++) {
      Tensors.push_back(torch::from_blob((TypeInValue*)array[i],
                                         {numRows, 1}
                                         ));
    }
    at::Tensor tensor = at::reshape(at::cat(Tensors, 1), {numRows, numCols});
    // temporary hack: using the TensorOptions caused a CUDA error
    tensor = tensor.to(torch::Device("cuda"));
    return tensor;
  }

  inline at::Tensor arrayToTensor(long numRows,
                                  long numCols,
                                  const TypeInValue** array)
  {
    c10::SmallVector<at::Tensor, 8> Tensors;
    for (int i = 0; i < numCols; i++) {
      Tensors.push_back(torch::from_blob((TypeInValue*)array[i],
                                         {numRows, 1},
                                         tensorOptions));
    }
    at::Tensor tensor = at::reshape(at::cat(Tensors, 1), {numRows, numCols});
    return tensor;
  }

  inline void tensorToArray(at::Tensor tensor,
                            long numRows,
                            long numCols,
                            TypeInValue** array)
  {
    // Transpose to get continuous memory and
    // perform single memcpy.
    tensor = tensor.transpose(1, 0);
    if (is_cpu) {
      for (long j = 0; j < numCols; j++) {
        auto tmp = tensor[j].contiguous();
        TypeInValue* ptr = tmp.data_ptr<TypeInValue>();
        HtoHMemcpy(array[j], ptr, sizeof(TypeInValue) * numRows);
      }
    } else {
      for (long j = 0; j < numCols; j++) {
        auto tmp = tensor[j].contiguous();
        TypeInValue* ptr = tmp.data_ptr<TypeInValue>();
        DtoDMemcpy(array[j], ptr, sizeof(TypeInValue) * numRows);
      }
    }
  }

  // -------------------------------------------------------------------------
  // loading a surrogate model!
  // -------------------------------------------------------------------------
  void _load_torch(const std::string& model_path,
                   c10::Device&& device,
                   at::ScalarType dType)
  {
    try {
      module = torch::jit::load(model_path);
      module.to(device);
      module.to(dType);
      tensorOptions = torch::TensorOptions().dtype(dType).device(device);
    } catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
    }
  }

  template <typename T,
            std::enable_if_t<std::is_same<T, double>::value>* = nullptr>
  inline void _load(const std::string& model_path,
                    const std::string& device_name)
  {
    _load_torch(model_path, torch::Device(device_name), torch::kFloat64);
  }

  template <typename T,
            std::enable_if_t<std::is_same<T, float>::value>* = nullptr>
  inline void _load(const std::string& model_path,
                    const std::string& device_name)
  {
    _load_torch(model_path, torch::Device(device_name), torch::kFloat32);
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
    auto input = arrayToTensor(num_elements, num_in, inputs);
    at::Tensor output = module.forward({input}).toTensor();

    tensorToArray(output, num_elements, num_out, outputs);
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
  SurrogateModel(const char* model_path, bool is_cpu = true)
      : model_path(model_path), is_cpu(is_cpu)
  {

    if (is_cpu)
      _load<TypeInValue>(model_path, "cpu");
    else
      _load<TypeInValue>(model_path, "cuda");
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
