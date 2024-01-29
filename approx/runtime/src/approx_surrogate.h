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

enum class Direction : int8_t {
  TENSOR_TO_MEM = 0,
  MEM_TO_TENSOR = 1,
};


template <typename Tensor>
class IndirectionPolicy {
  // we'll use virtual functions here because our actual compute is likely quite heavy
  // -- we're calling GPU kernels that operate on large tensors. So, we expect the 
  // overhead of virtual functions to be negligible. We can expect that optimizations
  // like using different cuda streams parallelize these operations on a GPU
  // will be much more impactful.
  public:
  
  /*
  * Apply indirection to the tensors in C. Directly modify C.
  * The size of C after this method is called depends on the policy used.
  * Indirection assumes that the outermost tensor comes first, as the code
  * is written. For instance, a[b[c]] will give the vector [a, b, c].
  */
  virtual void compute_result(std::vector<Tensor> &C) = 0;
  virtual Tensor get_result() = 0;
  virtual ~IndirectionPolicy() = default;
  virtual Tensor write_to(Tensor &T) = 0;
  virtual std::unique_ptr<IndirectionPolicy<Tensor>> clone() = 0;
  virtual Tensor memory_update(std::vector<Tensor> &C,  Direction ToFrom) = 0;
  
  /*
  * Copy the data from T into C. This method is called
  * AFTER indirection has been applied with compute_result.
  * Consequently, this is equivalent to a[b[c]] = T.
  */
  virtual void copy_(std::vector<Tensor> &C, Tensor &T) = 0;

  /*
  * Get the direction of our tensor translation.
  * Are we translating tensors to memory, or tensors to memory?
  * This method is used to communicate to client code.
  * Nevertheless, the direction affects the decisions made when applying indirection.
  */
  virtual Direction get_direction() = 0;

};

template<typename Tensor>
class MemToTensorIndirectionWrapper : public IndirectionPolicy<Tensor> {
  public:
  void compute_result(std::vector<Tensor> &C) override {
    Tensor ThisTens = C.back();
    C.pop_back();

    auto size = C.size();

    for(auto i = 0; i < size; i++) {
      auto OldTens = ThisTens;
      auto original_shape = ThisTens.sizes();
      ThisTens = C.back();
      ThisTens = ThisTens.flatten();
      ThisTens = ThisTens.index({OldTens.flatten()});
      ThisTens = ThisTens.reshape(original_shape);

      C.pop_back();
    }
    C.push_back(ThisTens);
  }

  Tensor get_result() override {
    return Tensor();
  }

  Tensor write_to(Tensor &T) override {
    return T;
  }

  std::unique_ptr<IndirectionPolicy<Tensor>> clone() override {
    return std::make_unique<MemToTensorIndirectionWrapper<Tensor>>();
  }

  void copy_(std::vector<Tensor> &Tens, Tensor &T) override {
    Tens[0].copy_(T);
  }

  Direction get_direction() override {
    return Direction::MEM_TO_TENSOR;
  }

  Tensor memory_update(std::vector<Tensor>& C, Direction ToFrom) override {
    // nothing to do here
    std::cerr << "Update not implemented for MemToTensorIndirectionWrapper\n";
    return Tensor();
  }
};

template<typename Tensor>
class TensorToMemIndirectionWrapper : public IndirectionPolicy<Tensor> {

  private:
  Tensor update_from(std::vector<Tensor>& C) {
    auto C0_base = C[0].data_ptr();
    auto new_tens = copy_to_new(C);
    auto new_tens_base = new_tens.data_ptr();
    if(C0_base == new_tens_base) {
      return new_tens.clone();
    }
    return new_tens;
  }
  void update_to(std::vector<Tensor>& C) {

  }
  public:
  Tensor write_to(Tensor &T) override {
    return T;
  }

  Tensor get_result() override {
    return Tensor();
  }

  /**
   * Apply indirection to the tensors in C. This method defers the last level
   * of indirection to when we actually copy. This is because the 
   * index method creates a copy of the original tensor. We need to make sure
   * we're keeping track of the original memory so we can copy to it.
   * If the user specified a[b[c]], we will compute b'= b[c] and carry that along
   * so we can later apply a[b'].
  */
  void compute_result(std::vector<Tensor> &C) override {
    // we already have the data in the form we need.
    if(C.size() <= 2) return;

    Tensor ThisTens = C.back();
    C.pop_back();

    auto size = C.size();

    for(int i = size; i > 1; i--) {
      auto OldTens = ThisTens;
      auto original_shape = ThisTens.sizes();
      ThisTens = C.back();
      ThisTens = ThisTens.flatten();
      ThisTens = ThisTens.index({OldTens.flatten()});
      ThisTens = ThisTens.reshape(original_shape);

      C.pop_back();
    }
    // now we may have two tensors: the final indirection tensor
    // and the tensor wrappign the original output tensor
    C.push_back(ThisTens);
  }

  std::unique_ptr<IndirectionPolicy<Tensor>> clone() override {
    return std::make_unique<TensorToMemIndirectionWrapper<Tensor>>();
  }

  void copy_(std::vector<Tensor> &C, Tensor &T) override {
    if (C.size() == 2) {
      auto C0 = C[0].flatten();
      auto C1 = C[1].flatten();
      auto Tflat = T.flatten();
      C0.index_put_({C1}, Tflat);
    } else {
      assert(C.size() == 1 && "Invalid number of tensors in TensorToMemIndirectionWrapper");
      C[0].copy_(T);
    }
  }

  Tensor copy_to_new(std::vector<Tensor> &C) {
    if(C.size() == 2) {
      auto C0 = C[0];
      auto original_shape = C0.sizes();
      C0 = C[0].flatten();
      auto C1 = C[1].flatten();
      auto indexed =  C0.index({C1});
      return indexed.reshape(original_shape);
    } else {
      assert(C.size() == 1 && "Invalid number of tensors in TensorToMemIndirectionWrapper");
      return C[0];
    }
  }

  Direction get_direction() override {
    return Direction::TENSOR_TO_MEM;
  }

  Tensor memory_update(std::vector<Tensor> &C,  Direction ToFrom) override {
    if(ToFrom == Direction::TENSOR_TO_MEM) {
      update_to(C);
      return Tensor();
    } else {
      return update_from(C);
    }
  }
};

template<typename Tensor>
class TensorWrapper {
  std::vector<Tensor> tensors;
  using DeviceTy = decltype(Tensor().device());
  // if we move the wrapped memory between devices, we can't just
  // copy the tensor back to the original device. We need to keep track
  // of the original tensor so we can copy to the memory it holds.
  Tensor FirstTensorOriginal;
  DeviceTy OriginalDevice = torch::kCPU;
  std::unique_ptr<IndirectionPolicy<Tensor>> IP;

  public:
    TensorWrapper(std::unique_ptr<IndirectionPolicy<Tensor>> &&IP)
        : IP(std::move(IP)) {}

    TensorWrapper() = default;

      TensorWrapper(const TensorWrapper& other) {
    tensors = other.tensors;
    FirstTensorOriginal = other.FirstTensorOriginal;
    OriginalDevice = other.OriginalDevice;
    if (other.IP) {
      IP = other.IP->clone();
    }
  }

  // Custom copy assignment operator (if cloning is possible)
  TensorWrapper& operator=(const TensorWrapper& other) {
    if (this != &other) {
      tensors = other.tensors;
      FirstTensorOriginal = other.FirstTensorOriginal;
      OriginalDevice = other.OriginalDevice;
      if (other.IP) {
        IP = other.IP->clone();
      } else {
        IP = nullptr;
      }
    }
    return *this;
  }

    IndirectionPolicy<Tensor> &get_indirection_policy() {
      return *IP;
  }

  std::vector<Tensor> &get_tensors() {
    return tensors;
  }

  void add_tensor(Tensor T) {
    tensors.push_back(T);
  }

  void add_tensor(TensorWrapper<Tensor> &TW) {
    // copy the tensors in TW to this
    auto TW_tensors = TW.get_tensors();
    tensors.insert(tensors.end(), TW_tensors.begin(), TW_tensors.end());
  }

  Tensor compute_result() {
    IP->compute_result(tensors);
    return tensors[0];
  }

  Tensor perform_indirection() {
    return compute_result();
  }

  void copy_(Tensor &T) {
    IP->copy_(tensors, T);
    if(OriginalDevice != T.device()) {
      std::cout << "Original device: " << OriginalDevice << "\n";
      std::cout << "Current device: " << T.device() << "\n";
      std::cout << "Copying the data\n";
      FirstTensorOriginal.copy_(tensors[0]);
    }
  }

  
  auto sizes(size_t idx = 0) const {
    return tensors[idx].sizes();
  }

  auto strides(size_t idx = 0) const {
    return tensors[idx].strides();
  }

  auto dim(size_t idx = 0) const {
    return tensors[idx].dim();
  }

  static std::vector<Tensor> concat(std::vector<TensorWrapper<Tensor>> &T) {
    std::vector<Tensor> result;
    for(auto &t : T) {
      auto tensors = t.get_tensors();
      result.insert(result.end(), tensors.begin(), tensors.end());
    }
    return result;
  }

  void to(DeviceTy d, bool non_blocking = false) {
    FirstTensorOriginal = tensors[0];
    OriginalDevice = FirstTensorOriginal.device();

    // TODO: This is something I think should go to the indirection policy.
    // that will let us avoid copying output without indirection to the GPU, as
    // we /shouldn't/ need to do that. I expect that the cost of applying indirection
    // will be much lower on the GPU, so we'll want to do it in that case. However,
    // with this implementation we can NEVER copy single tensors between devices,
    // which violates what we would expect from this method.
    bool am_tensor_to_mem = IP->get_direction() == Direction::TENSOR_TO_MEM;
    if(tensors.size() == 1 && am_tensor_to_mem)
      return;
    for(auto &t : tensors) {
      t = t.to(d, non_blocking);
    }
  }

  TensorWrapper<Tensor> update_from_memory() {
    auto T = IP->memory_update(tensors, Direction::MEM_TO_TENSOR);
    auto Wrapper = TensorWrapper<Tensor>(std::make_unique<TensorToMemIndirectionWrapper<Tensor>>());
    Wrapper.add_tensor(T);
    return Wrapper;
  }

  Tensor get_tensor(size_t idx = 0) {
    return tensors[idx];
  }
};

template<typename Tensor>
struct TensorWrapperTensorToMem {
  TensorWrapper<Tensor> operator()() const {
    return TensorWrapper<Tensor>(std::make_unique<TensorToMemIndirectionWrapper<Tensor>>());
  }

  TensorWrapper<Tensor> operator()(Tensor &T) const {
    TensorWrapper<Tensor> TW(std::make_unique<TensorToMemIndirectionWrapper<Tensor>>());
    TW.add_tensor(T);
    return TW;
  }

  TensorWrapper<Tensor> operator()(Tensor &&T) const {
    TensorWrapper<Tensor> TW(std::make_unique<TensorToMemIndirectionWrapper<Tensor>>());
    TW.add_tensor(T);
    return TW;
  }
};

template<typename Tensor>
struct TensorWrapperMemToTensor {
  TensorWrapper<Tensor> operator()() const {
    return TensorWrapper<Tensor>(std::make_unique<MemToTensorIndirectionWrapper<Tensor>>());
  }

TensorWrapper<Tensor> operator()(Tensor &T) const {
    TensorWrapper<Tensor> TW(std::make_unique<MemToTensorIndirectionWrapper<Tensor>>());
    TW.add_tensor(T);
    return TW;
  }

  TensorWrapper<Tensor> operator()(Tensor &&T) const {
    TensorWrapper<Tensor> TW(std::make_unique<MemToTensorIndirectionWrapper<Tensor>>());
    TW.add_tensor(T);
    return TW;
  }
};


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

  static size_t getElementSizeForType(TensorDataTypeType T) {
    return TensorImpl::getElementSizeForType(T);
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

  static size_t getElementSizeForType(TensorDataTypeType T) {
    return torch::elementSize(T);
  }
};


using TensorType = AbstractTensor<TorchTensorImpl>;
using WrappedTensor = TensorWrapper<TensorType::tensor_t>;

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
      ipt_tens = ipt_tens.squeeze(-1);
      ipt_tens = ipt_tens.unsqueeze(1);

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