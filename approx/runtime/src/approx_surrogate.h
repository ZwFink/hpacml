//Copyright (c) Lawrence Livermore National Security, LLC and other AMS
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute

#ifndef __SURROGATE_HPP__
#define __SURROGATE_HPP__
#define __ENABLE_TORCH__ 1

#define NUM_ITEMS 4194304
#include <string>

#include <torch/script.h>  // One-stop header.
#include <type_traits>
#include "database/database.h"
#include "approx_internal.h"
#include "event.h"
#include <type_traits>


#include <cuda_runtime.h>
#include "approx_tensor.h"


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

    static inline void transferWithinDevice(void *dest, void *src, size_t nBytes)
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
      DtoHMemcpy(dest, src, nBytes);
    }

    static inline void transferWithinDevice(void *dest, void *src, size_t nBytes)
    {
      DtoDMemcpy(dest, src, nBytes);
    }
};

typedef struct internal_tensor_repr_data {
  ApproxType underlying_type;
	int type;
  TensorType::Device original_device{TensorType::CPU};
  Direction direction;
	std::vector<WrappedTensor> Tensors;

	~internal_tensor_repr_data() {
	}

	void set_library_type(int t) {
		type = t;
	}

  void set_device(TensorType::Device d) {
    original_device = d;
  }

  void set_underlying_type(ApproxType t) {
    underlying_type = t;
  }

  size_t get_num_tensors() const {
    return Tensors.size();
  }

  void set_direction(Direction d) {
    direction = d;
  }
  void add_tensor(WrappedTensor Tens) {
    Tensors.push_back(Tens);
  }

  WrappedTensor &get_wrapped_tensor(size_t idx) {
    return Tensors[idx];
  }

  TensorType::tensor_t &get_tensor(size_t idx, size_t tens_idx = 0) {
    return Tensors[idx].get_tensors()[tens_idx];
  }

  // TODO: This should be the implementationf or update_to_memory for 
  // output tensors
  void update_to_memory(TensorType::tensor_t &T) {
    if (Tensors.size() == 1) {
      auto &opt = Tensors[0];
      if (opt.sizes() == T.sizes()) {
        opt.copy_(T);
      } else {
        T = T.squeeze(0);
        if(opt.sizes() == T.sizes()) {
          opt.copy_(T);
          return;
        }
        std::cout << "Error: The output tensor does not match the expected shape\n"
                  << "Expected: " << opt.sizes() << " Got: " << T.sizes() << "\n"
                  << "Skipping update\n"
                  << "Please check the model architecture\n";
      }
      return;
    }

    int col_start = 0;
    int col_end = 1;
    for (int i = 0; i < Tensors.size(); i++) {
      // get the rightmost item in the sahpe
      auto &opt = Tensors[i];
      auto opt_shape = opt.sizes();
      auto opt_dim = opt_shape.size();
      auto opt_rightmost = opt_shape[opt_dim - 1];
      col_end = col_start + opt_rightmost;

      // get T[col_start:col_end] columns
      auto T_cols = T.narrow(1, col_start, col_end - col_start);
      opt.copy_(T_cols);
      col_start = col_end;
    }
  }

  TensorType::tensor_t update_from_memory() {
    std::vector<TensorType::tensor_t> tensors;
    for (auto &t : Tensors) {
      tensors.push_back(t.update_from_memory().get_tensors()[0]);
    }
    return TensorType::cat(tensors, -1);
  }

} internal_repr_metadata_t;
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
    template<typename DataType>
    at::Tensor arrayToTensor(long numRows, long numCols, DataType **array) {
      auto tensorOptions = tensor.options();
      for (int i = 0; i < numCols; i++) {
        auto column =
            tensor.select(1, i).slice(0, insert_index, insert_index + numRows);
        auto data = reinterpret_cast<DataType *>(array[i]);
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

template <typename TypeInValue>
class CatTensorTranslator : public TensorTranslator<TypeInValue> {
  public:
    std::vector<TensorType::tensor_t> allocatedTensors;
    TensorType::tensor_t tensor =
        TensorType::empty({0, 5}, TensorType::float64);
    CatTensorTranslator(at::Tensor &tensor) : TensorTranslator<TypeInValue> {
      tensor
    }
    {
      for (int i = 0; i < 5; i++)
        allocatedTensors.push_back(
            TensorType::empty({NUM_ITEMS, 1}, TensorType::float64));
    }

    template <typename DataType>
    at::Tensor &arrayToTensor(long numRows, long numCols, DataType **array) {
      for (int i = 0; i < numCols; i++) {
        auto temp = TensorType::from_blob((TypeInValue *)array[i], {numRows, 1},
                                          TensorType::float64);

        allocatedTensors[i].narrow(0, this->insert_index, numRows).copy_(temp);
      }

      // auto tensor = torch::nested::as_nested_tensor(allocatedTensors);
      auto tensor = TensorType::cat(allocatedTensors, 1);
      this->tensor = tensor;
      this->insert_index += numRows;
      return this->tensor;
    }

    void reset() {
      this->tensor = TensorType::empty({0, 5}, TensorType::float64);
      this->insert_index = 0;
    }

    void tensorToArray(at::Tensor tensor, long numRows, long numCols,
                       TypeInValue **array) {}

    TensorType::tensor_t prepareForInference(TensorType::tensor_t &t) {
      return t;
    }
};

namespace {
template<typename Model>
struct EvalDispatcher {
  private:
  void EvaluateDispatchForType(long num_elements, size_t num_in, size_t num_out,
                       void **inputs, void **outputs, ApproxType Underlying, Model& M) {
    switch(Underlying) {
      #define APPROX_TYPE(Enum, CType, nameOfType) \
      case Enum:  \
        M._evaluate(num_elements, num_in, num_out, (CType **)inputs, (CType **)outputs); \
        break;
      #include "clang/Basic/approxTypes.def"
      case INVALID:
        std::cout << "INVALID DATA TYPE passed in argument list\n";
    }
  }

  void EvaluateOnlyDispatchForType(long num_elements, size_t num_in, size_t num_out,
                       void *ipt_tensor, void **outputs, ApproxType Underlying, Model& M) {
    switch(Underlying) {
      #define APPROX_TYPE(Enum, CType, nameOfType) \
      case Enum:  \
        M._eval_only(num_elements, num_in, num_out, ipt_tensor, (CType **)outputs); \
        break;
      #include "clang/Basic/approxTypes.def"
      case INVALID:
        std::cout << "INVALID DATA TYPE passed in argument list\n";
      }
  }

    void EvaluateTensorInputsOutputsForType(internal_repr_metadata_t &ipt_tensor,
                                            internal_repr_metadata_t &outputs,
                                            ApproxType Underlying, Model & M) {
      M._eval_only(ipt_tensor, outputs);
  }
  public:
    inline void evaluate(long num_elements, size_t num_in, size_t num_out,
                         void **inputs, void **outputs, ApproxType Underlying,
                         Model &M) {
    EvaluateDispatchForType(num_elements, num_in, num_out, inputs, outputs,
                            Underlying, M);
    }

    inline void evaluate_only(long num_elements, size_t num_out, void *ipt,
                              void **outputs, ApproxType Underlying, Model &M) {
    EvaluateOnlyDispatchForType(num_elements, 1, num_out, ipt, outputs,
                                Underlying, M);
    }

    inline void evaluate(internal_repr_metadata_t &ipt_tensor,
                         internal_repr_metadata_t &outputs,
                         ApproxType Underlying, Model &M) {
      EvaluateTensorInputsOutputsForType(ipt_tensor, outputs, Underlying, M);
    }
};
}

//! ----------------------------------------------------------------------------
//! An implementation for a surrogate model
//! ----------------------------------------------------------------------------
template <typename ExecutionPolicy, typename TensorTranslator,
typename TypeInValue>
class SurrogateModel : public ExecutionPolicy
{

  template<typename>
  friend class EvalDispatcher;
  static_assert(std::is_floating_point<TypeInValue>::value,
                "SurrogateModel supports floating-point values (floats, "
                "doubles, or long doubles) only!");

private:
  const std::string model_path;
  const bool is_cpu;
  std::unique_ptr<TensorTranslator> translator;
  c10::InferenceMode guard{true};


#ifdef __ENABLE_TORCH__
  // -------------------------------------------------------------------------
  // variables to store the torch model
  // -------------------------------------------------------------------------
  torch::jit::script::Module module;
  c10::TensorOptions tensorOptions;

  template <typename DataType>
  inline void tensorToArray(at::Tensor tensor,
                            long numRows,
                            long numCols,
                            DataType** array)
  {
    // Transpose to get continuous memory and
    // perform single memcpy.
    auto DTOHEv = EventRecorder::CreateGPUEvent("From Tensor");
    DTOHEv.recordStart();
    tensor = TensorType::transpose(tensor, {1, 0});
      for (long j = 0; j < numCols; j++) {
        auto tmp = tensor[j].contiguous();
        DataType* ptr = static_cast<DataType*>(tmp.data_ptr());
        ExecutionPolicy::transferFromDevice(array[j], ptr,
                                            sizeof(TypeInValue) * numRows);
        // ExecutionPolicy::transferWithinDevice(array[j], ptr,
                                           // sizeof(DataType) * numRows);
      }
    DTOHEv.recordEnd();
    EventRecorder::LogEvent(DTOHEv);
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
  template<typename DataType>
  inline void _evaluate(long num_elements,
                        long num_in,
                        size_t num_out,
                        DataType** inputs,
                        DataType** outputs)
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

  public:
  template<typename DataType>
  inline void _eval_only(long num_elements,
                        long num_in,
                        size_t num_out,
                        internal_repr_metadata_t &input,
                        DataType** outputs)
  {
      torch::NoGradGuard no_grad;
      auto FPEvent = EventRecorder::CreateGPUEvent("Forward Pass");
      FPEvent.recordStart();
      auto &ipt_tens = input.get_tensor(0);
      at::Tensor output = module.forward({ipt_tens}).toTensor();
      FPEvent.recordEnd();
      EventRecorder::LogEvent(FPEvent);
      // tensorToArray(output, num_elements, num_out, outputs);

  }

  inline void _eval_only(
   internal_repr_metadata_t &inputs, internal_repr_metadata_t &outputs) {
      auto FPEvent = EventRecorder::CreateGPUEvent("Forward Pass");
      auto FromTens = EventRecorder::CreateGPUEvent("From Tensor");
      auto &ipt_tens = inputs.get_tensor(0);

      FPEvent.recordStart();
      at::Tensor output = module.forward({ipt_tens}).toTensor();
      FPEvent.recordEnd();


      FromTens.recordStart();
      outputs.update_to_memory(output);
      FromTens.recordEnd();
      EventRecorder::LogEvent(FPEvent);
      EventRecorder::LogEvent(FromTens);
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
  SurrogateModel(std::string&& model_path, bool is_cpu = true)
      : model_path(model_path), is_cpu(is_cpu)
  {
    if(model_path.empty())
      return; 

    _load<TypeInValue>(model_path, ExecutionPolicy::device);
  }

  void set_model(std::string&& model_path)
  {
    _load<TypeInValue>(model_path, ExecutionPolicy::device);
  }

  inline void evaluate(ApproxType Underlying,
                       long num_elements,
                       long num_in,
                       size_t num_out,
                       void** inputs,
                       void** outputs)
  {
    EvalDispatcher<std::remove_reference_t<decltype(*this)>> functor;
    functor.evaluate(num_elements, num_in, num_out, inputs, outputs, Underlying, *this);
  }

  inline void evaluate(ApproxType Underlying, long num_elements,
                       std::vector<void *> inputs,
                       std::vector<void *> outputs) {
    evaluate(Underlying, num_elements, inputs.size(), outputs.size(),
             reinterpret_cast<void **>(inputs.data()),
             reinterpret_cast<void **>(outputs.data()));
  }

  inline void evaluate(ApproxType Underlying, long num_elements, size_t num_out, void *ipt,
                       void **outputs) {
    eval_with_tensor_input(Underlying, num_elements, 1, ipt, outputs);
  }

  inline void evaluate(ApproxType Underlying, long num_elements,
                       void *ipt_tensor,
                       std::vector<void*> outputs)
  {
    eval_with_tensor_input(Underlying, num_elements, outputs.size(), ipt_tensor, reinterpret_cast<void**>(outputs.data()));
  }

  inline void evaluate(ApproxType Underlying,
                       internal_repr_metadata_t &ipt_tensor,
                       internal_repr_metadata_t &outputs)
  {
    eval_with_tensor_input_output(Underlying, ipt_tensor, outputs);
  }

  inline void eval_with_tensor_input(ApproxType Underlying, long num_elements,
                       size_t num_out, void *ipt_tensor,
                        void **outputs
                       )
  {
    EvalDispatcher<std::remove_reference_t<decltype(*this)>> functor;
    functor.evaluate_only(num_elements, num_out, ipt_tensor, outputs, Underlying, *this);
  }

  inline void eval_with_tensor_input_output(ApproxType Underlying,
                       internal_repr_metadata_t &ipt_tensor,
                       internal_repr_metadata_t &outputs
                       )
  {
    EvalDispatcher<std::remove_reference_t<decltype(*this)>> functor;
    functor.evaluate(ipt_tensor, outputs, Underlying, *this);
  }
};

#endif