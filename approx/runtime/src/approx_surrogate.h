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

template <typename TensorImpl>
class AbstractTensor : private TensorImpl {
  public:
  using tensor_t = typename TensorImpl::tensor_t;
  using tensor_options_t = typename TensorImpl::tensor_options_t;
  using Device = typename TensorImpl::Device;
  using Shape = typename TensorImpl::Shape;
  template<typename T>
  using ArrayRef = typename TensorImpl::template ArrayRef<T>;
  using TensorDataTypeType = typename TensorImpl::TensorDataTypeType;
  using TensorDeviceInstanceType = typename TensorImpl::TensorDeviceInstanceType;
  static constexpr auto CUDA = TensorImpl::CUDA;
  static constexpr auto CPU = TensorImpl::CPU;
  static constexpr auto float64 = TensorImpl::float64;
  static constexpr auto float32 = TensorImpl::float32;
  template <typename Tensors>
  static tensor_t cat(Tensors& T, int dim)
  {
    return TensorImpl::cat(T, dim);
  }

  static tensor_t empty(Shape shape, tensor_options_t opts)
  {
    return TensorImpl::empty(shape, opts);
  }

  static tensor_t transpose(tensor_t t, Shape newShape)
  {
    return TensorImpl::transpose(t, newShape);
  }

  static tensor_t from_blob(void *mem, Shape shape, tensor_options_t opts) {
    return TensorImpl::from_blob(mem, shape, opts);
  }
  static tensor_t from_blob(void *mem, Shape shape, Shape strides, tensor_options_t opts) {
    return TensorImpl::from_blob(mem, shape, strides, opts);
  }

template<typename T>
  static ArrayRef<T> makeArrayRef(T *ptr, size_t size)
  {
    return TensorImpl::makeArrayRef(ptr, size);
  }

  static int getTensorLibraryType() {
    return TensorImpl::getTensorLibraryType();
  }

  template<typename T>
  static TensorDataTypeType getTensorType() {
    return TensorImpl::template getTensorType<T>();
  }

  static TensorDataTypeType getTensorDataTypeTypeFromApproxType(ApproxType Type) {
    switch(Type) {
      #define APPROX_TYPE(Enum, CType, nameOfType) \
      case Enum:  \
        return getTensorType<CType>();
      #include "clang/Basic/approxTypes.def"
      case INVALID:
        std::cout << "INVALID DATA TYPE passed in argument list\n";
    }
  }

  static Device getDeviceForPointer(void *ptr) {
    if (ptr == nullptr) {
      return Device(CPU);
    }
    cudaPointerAttributes attributes;
    cudaPointerGetAttributes(&attributes, ptr);
    if (attributes.type == cudaMemoryTypeDevice || attributes.type == cudaMemoryTypeManaged) {
      return {CUDA, static_cast<char>(attributes.device)};
    } else {
      return {CPU};
    }
  }
};

using TensorLibraryType = __approx_tensor_library_type;

class TorchTensorImpl {
  public:

  using tensor_t = torch::Tensor;
  using tensor_options_t = torch::TensorOptions;
  using Device = c10::Device;
  template<typename T>
  using ArrayRef = torch::ArrayRef<T>;
  static constexpr auto CUDA = torch::kCUDA;
  static constexpr auto CPU = torch::kCPU;
  using TensorDeviceInstanceType = decltype(torch::kCUDA);
  using TensorDataTypeType = decltype(torch::kDouble);
  using Shape = torch::IntArrayRef;
  static constexpr auto float64 = torch::kDouble;
  static constexpr auto float32 = torch::kFloat;

  static int getTensorLibraryType() {
    return (int) TensorLibraryType::TORCH;
  }

  template<typename Tensors>
  static torch::Tensor cat(Tensors& T, int dim)
  {
    return torch::cat(T, dim);
  }
  static torch::Tensor empty(Shape shape, tensor_options_t opts)
  {
    return torch::empty(shape, opts);
  }

  static torch::Tensor transpose(torch::Tensor t, Shape newShape)
  {
    return t.permute(newShape);
  }

  static torch::Tensor from_blob(void *mem, Shape shape, tensor_options_t opts) {
    return torch::from_blob(mem, shape, opts);
  }
  static torch::Tensor from_blob(void *mem, Shape shape, Shape strides, tensor_options_t opts) {
    return torch::from_blob(mem, shape, strides, opts);
  }

  template<typename T>
  static torch::ArrayRef<T> makeArrayRef(T *ptr, size_t size)
  {
    return torch::ArrayRef<T>(ptr, size);
  }

  template<typename T>
  static TensorDataTypeType getTensorType() {
    if (std::is_same<T, double>::value) {
      return torch::kDouble;
    } else if (std::is_same<T, float>::value) {
      return torch::kFloat;
    } else if (std::is_same<T, int>::value) {
      return torch::kInt;
    } else if (std::is_same<T, long>::value) {
      return torch::kLong;
    } else if (std::is_same<T, short>::value) {
      return torch::kShort;
    } else if (std::is_same<T, unsigned char>::value) {
      return torch::kByte;
    } else {
      assert(False && "Invalid type passed to getTensorType");
    }
  }

};


using TensorType = AbstractTensor<TorchTensorImpl>;

typedef struct internal_tensor_repr_data {
  ApproxType underlying_type;
	int type;
  TensorType::Device original_device{TensorType::CPU};
	void *data;

	~internal_tensor_repr_data() {
		TensorType::tensor_t *T = (TensorType::tensor_t *)data;
		delete T;
	}

	void set_library_type(int t) {
		type = t;
	}

	void set_data(void *d) {
		data = d;
	}

  void set_device(TensorType::Device d) {
    original_device = d;
  }

  void set_underlying_type(ApproxType t) {
    underlying_type = t;
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

  inline void tensorToArray(at::Tensor tensor,
                            long numRows,
                            long numCols,
                            TypeInValue** array)
  {
    // Transpose to get continuous memory and
    // perform single memcpy.
    auto DTOHEv = EventRecorder::CreateGPUEvent("From Tensor");
    DTOHEv.recordStart();
    tensor = TensorType::transpose(tensor, {1, 0});
      for (long j = 0; j < numCols; j++) {
        auto tmp = tensor[j].contiguous();
        TypeInValue* ptr = tmp.data_ptr<TypeInValue>();
        // ExecutionPolicy::transferFromDevice(array[j], ptr,
                                            // sizeof(TypeInValue) * numRows);
        ExecutionPolicy::transferWithinDevice(array[j], ptr,
                                            sizeof(TypeInValue) * numRows);
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
                        void *ipt_tens,
                        DataType** outputs)
  {
      torch::NoGradGuard no_grad;
      at::Tensor input = *(at::Tensor *)ipt_tens;
      auto FPEvent = EventRecorder::CreateGPUEvent("Forward Pass");
      FPEvent.recordStart();
      at::Tensor output = module.forward({input}).toTensor();
      FPEvent.recordEnd();
      EventRecorder::LogEvent(FPEvent);
      // tensorToArray(output, num_elements, num_out, outputs);

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

  inline void eval_with_tensor_input(ApproxType Underlying, long num_elements,
                       size_t num_out, void *ipt_tensor,
                        void **outputs
                       )
  {
    EvalDispatcher<std::remove_reference_t<decltype(*this)>> functor;
    functor.evaluate_only(num_elements, num_out, ipt_tensor, outputs, Underlying, *this);
  }
};

#endif
