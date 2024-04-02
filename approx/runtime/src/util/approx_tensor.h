#ifndef __APPROX_TENSOR__
#define __APPROX_TENSOR__
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
  using Slices = typename TensorImpl::Slices;
  using InferenceGuard = typename TensorImpl::InferenceGuard;
  static constexpr auto CUDA = TensorImpl::CUDA;
  static constexpr auto CPU = TensorImpl::CPU;
  static constexpr auto float64 = TensorImpl::float64;
  static constexpr auto float32 = TensorImpl::float32;
  template <typename Tensors>
  static tensor_t cat(Tensors& T, int dim)
  {
    return TensorImpl::cat(T, dim);
  }

  static int64_t dim(tensor_t &t) {
    return TensorImpl::dim(t);
  }

  static tensor_t empty(Shape shape, tensor_options_t opts)
  {
    return TensorImpl::empty(shape, opts);
  }

  static tensor_t to(tensor_t &tens, Device d, bool non_blocking = false) {
    return TensorImpl::to(tens, d, non_blocking);
  }

  static tensor_t transpose(tensor_t &t, Shape newShape)
  {
    return TensorImpl::transpose(t, newShape);
  }

  static tensor_t from_blob(void *mem, Shape shape, tensor_options_t opts) {
    return TensorImpl::from_blob(mem, shape, opts);
  }
  static tensor_t from_blob(void *mem, Shape shape, Shape strides, tensor_options_t opts) {
    return TensorImpl::from_blob(mem, shape, strides, opts);
  }

  static Shape shapeFromVector(std::vector<int64_t>& vec) {
    return TensorImpl::shapeFromVector(vec);
  }

  static tensor_t index(const tensor_t& t, Slices& slices) {
    return TensorImpl::index(t, slices);
  }

  // TODO: #include <utility> for std::pair
  static Slices initSlices(const std::vector<std::pair<int64_t, int64_t>>& bounds) {
    return TensorImpl::initSlices(bounds);
  }

  static void modSlices(Slices& slices, const std::pair<int64_t, int64_t>& bound, int index) {
    return TensorImpl::modSlices(slices, bound, index);
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
  static T* data_ptr(tensor_t &t) {
    return TensorImpl::template data_ptr<T>(t);
  }

  static void *data_ptr(tensor_t &t) {
    return TensorImpl::data_ptr(t);
  }

  template<typename T>
  static TensorDataTypeType getTensorType() {
    return TensorImpl::template getTensorType<T>();
  }

  static TensorDataTypeType getTensorDataType(tensor_t &t) {
    return TensorImpl::getTensorDataType(t);
  }

  static size_t numel(tensor_t &t) {
    return TensorImpl::numel(t);
  }

  static size_t size_bytes(tensor_t &t, ApproxType DType) {
    return numel(t) * getElementSizeForType(getTensorDataTypeTypeFromApproxType(DType));
  }

  static TensorDataTypeType getTensorDataTypeTypeFromApproxType(ApproxType Type) {
    switch(Type) {
      #define APPROX_TYPE(Enum, CType, nameOfType) \
      case Enum:  \
        return getTensorType<CType>();
      #include "clang/Basic/approxTypes.def"
      case INVALID:
        std::cout << "INVALID DATA TYPE passed in argument list\n";
      default:
        std::cerr << "Unknown DType value: " << static_cast<int>(Type) << std::endl;
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
  using Slices = std::vector<torch::indexing::TensorIndex>;
  static constexpr auto float64 = torch::kDouble;
  static constexpr auto float32 = torch::kFloat;
  using InferenceGuard = at::InferenceMode;

  static int getTensorLibraryType() {
    return (int) TensorLibraryType::TORCH;
  }

  template<typename Tensors>
  static torch::Tensor cat(Tensors& T, int dim)
  {
    return torch::cat(T, dim);
  }

  static int64_t dim(tensor_t &t) {
    return t.dim();
  }

  template<typename T>
  static T* data_ptr(tensor_t &t) {
    return t.data_ptr<T>();
  }

  static void* data_ptr(tensor_t &t) {
    return t.data_ptr();
  }

  static size_t numel(tensor_t &t) {
    return t.numel();
  }

  static torch::Tensor empty(Shape shape, tensor_options_t opts)
  {
    return torch::empty(shape, opts);
  }

  static torch::Tensor transpose(torch::Tensor &t, Shape newShape)
  {
    return t.permute(newShape);
  }

  static torch::Tensor to(tensor_t &tens, Device d, bool non_blocking = false) {
    return tens.to(d, non_blocking);
  }

  static torch::Tensor from_blob(void *mem, Shape shape, tensor_options_t opts) {
    return torch::from_blob(mem, shape, opts);
  }
  static torch::Tensor from_blob(void *mem, Shape shape, Shape strides, tensor_options_t opts) {
    return torch::from_blob(mem, shape, strides, opts);
  }

  static Shape shapeFromVector(std::vector<int64_t>& vec) {
    Shape shape(vec);
    return shape;
  }

  static torch::Tensor index(const tensor_t &t, Slices &slices) {
    return t.index(slices);
  }

  static Slices initSlices(const std::vector<std::pair<int64_t, int64_t>> &bounds) {
    Slices slices;
    for (size_t i = 0; i < bounds.size(); ++i) {
        slices.push_back(torch::indexing::Slice(bounds[i].first, bounds[i].second, 1)); // Slice from 0 to vec[i] for the ith dimension
    }
    slices.push_back(torch::indexing::Ellipsis);
    return slices; 
  }

  static void modSlices(Slices &slices, const std::pair<int64_t, int64_t> &bound, int index) {
    if (index >= slices.size()-1)
      std::cerr << "invalid index for modSlices, must be less than " << slices.size() << "\n";
    
    slices[index] = torch::indexing::Slice(bound.first, bound.second, 1);
    return;
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

  static TensorDataTypeType getTensorDataType(tensor_t &t) {
    auto scalarType = t.scalar_type();
    return scalarType;
  }

  static size_t getElementSizeForType(TensorDataTypeType T) {
    return torch::elementSize(T);
  }
};


using TensorType = AbstractTensor<TorchTensorImpl>;
using TensorImpl = AbstractTensor<TorchTensorImpl>;
using WrappedTensor = TensorWrapper<TensorType::tensor_t>;
#endif // __APPROX_TENSOR__