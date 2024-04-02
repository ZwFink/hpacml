#ifndef __DATABASE__
#define __DATABASE__

#include <hdf5.h>
#include "hdf5_hl.h" 
#include <vector>
#include <string>
#include <cstddef>
#include <iostream>
#include <cassert>
#include <cstring>

#include "../approx_internal.h"
#include "../include/approx.h"
#include "span.h"
#include "../approx_surrogate.h"
#include "approx_tensor.h"

#define HDF5_ERROR(id)                                                         \
  if (id < 0) {                                                                \
    fprintf(stderr, "Error Happened in Line:%s:%d\n", __FILE__, __LINE__);     \
    exit(-1);                                                                  \
  }

namespace {
  template<typename T>
  static hid_t getHDF5DataType() {
    if (std::is_same<T, double>::value) {
      return H5T_NATIVE_DOUBLE;
    } else if (std::is_same<T, float>::value) {
      return H5T_NATIVE_FLOAT;
    } else if (std::is_same<T, int>::value) {
      return H5T_NATIVE_INT;
    } else if (std::is_same<T, long>::value) {
      return H5T_NATIVE_LONG;
    } else if (std::is_same<T, short>::value) {
      return H5T_NATIVE_SHORT;
    } else if (std::is_same<T, unsigned char>::value) {
      return H5T_NATIVE_UCHAR;
    } else {
      assert(False && "Invalid type passed to getHDF5DataType");
    }
  }

  static hid_t getHDF5TypeFromApproxType(ApproxType Type) {
    switch(Type) {
      #define APPROX_TYPE(Enum, CType, nameOfType) \
      case Enum:  \
        return getHDF5DataType<CType>();
      #include "clang/Basic/approxTypes.def"
      case INVALID:
        std::cout << "INVALID DATA TYPE passed in argument list\n";
    }
  }
};

class BaseDB {
public:
  virtual void *InstantiateRegion(uintptr_t Addr, const char *Name) = 0;

  virtual void DataToDB(void *Region, double *Data, size_t NumRows,
                        int NumCols) = 0;

  virtual void RegisterMemory(const char *gName, const char *name, void *ptr,
                              size_t numBytes, HPACDType dType) = 0;
  virtual ~BaseDB() {}
};

class HDF5TensorRegionView {
  class TensorData {
    public:
    std::string dset_name;
    hid_t dset;
    hid_t memSpace;
    std::vector<hsize_t> shape;
    bool initialized = false;
    ApproxType approx_type;
    hid_t hdf_native_type;
    std::vector<hsize_t> chunk_vector;

    bool isInitialized() const { return initialized; }
    ~TensorData() {
      if (initialized) {
        auto errcode = H5Dclose(dset);
        HDF5_ERROR(errcode);
      }
    }
  };

  hid_t file;
  hid_t regionGroup;
  uintptr_t addr;
  std::string regionName;
  TensorData IptTensorData;
  TensorData OptTensorData;
  TensorData RuntimeData;

  bool expect_input = true;

  template<typename Tensor>
  void writeTensorData(TensorData &tensorData, Tensor &tensor, std::string name, ApproxType DType) {
    if(!tensorData.isInitialized()) {
      initializeTensorData(tensorData, name, tensor, DType);
    }

    auto memSpace = H5Dget_space(tensorData.dset);
    HDF5_ERROR(memSpace);
    auto &shape = tensorData.shape;
    auto ndim = shape.size();
    std::vector<hsize_t> dims(ndim);
    std::vector<hsize_t> count(ndim);
    std::vector<hsize_t> start(ndim);

    // add one row to the dataset
    auto oldshape = shape[0];
    shape[0] = 1;
    shape[0] = oldshape + 1;

    auto errcode = H5Dset_extent(tensorData.dset, shape.data());
    HDF5_ERROR(errcode);

    hid_t fileSpace = H5Dget_space(tensorData.dset);
    HDF5_ERROR(fileSpace);

    start[0] = tensorData.shape[0] - 1;
    std::fill(start.begin() + 1, start.end(), 0);

    std::copy(tensorData.shape.begin(), tensorData.shape.end(), count.begin());

    count[0] = 1; // Only one row
    errcode = H5Sselect_hyperslab(fileSpace, H5S_SELECT_SET, start.data(), NULL, count.data(), NULL);
    HDF5_ERROR(errcode);

    hid_t newMemSpace = H5Screate_simple(count.size(), count.data(), NULL);
    HDF5_ERROR(newMemSpace);

    auto *data = TensorImpl::data_ptr(tensor);
    errcode = H5Dwrite(tensorData.dset, tensorData.hdf_native_type, newMemSpace, fileSpace, H5P_DEFAULT, data);
    HDF5_ERROR(errcode);

    // Close resources
    H5Sclose(fileSpace);
    H5Sclose(memSpace);
    H5Sclose(newMemSpace);
  }

  template<typename Tensor>
  void writeTensorDataChunk(TensorData &tensorData, Tensor &tensor, std::vector<int64_t>& loc_vector, bool first) {
    auto memSpace = H5Dget_space(tensorData.dset);
    HDF5_ERROR(memSpace);
    auto &shape = tensorData.shape;
    auto ndim = shape.size();
    std::vector<hsize_t> count(ndim);
    std::vector<hsize_t> start(ndim);

    // add one row to the dataset
    if (first)
      shape[0] += 1;

    auto errcode = H5Dset_extent(tensorData.dset, shape.data());
    HDF5_ERROR(errcode);

    hid_t fileSpace = H5Dget_space(tensorData.dset);
    HDF5_ERROR(fileSpace);

    start[0] = tensorData.shape[0] - 1;
    std::copy(loc_vector.begin(), loc_vector.end(), start.begin() + 1);

    count[0] = 1;
    std::copy(tensorData.chunk_vector.begin(), tensorData.chunk_vector.end(), count.begin() + 1);

    errcode = H5Sselect_hyperslab(fileSpace, H5S_SELECT_SET, start.data(), NULL, count.data(), NULL);
    HDF5_ERROR(errcode);

    hid_t newMemSpace = H5Screate_simple(count.size(), count.data(), NULL);
    HDF5_ERROR(newMemSpace);

    auto *data = TensorImpl::data_ptr(tensor);
    errcode = H5Dwrite(tensorData.dset, tensorData.hdf_native_type, newMemSpace, fileSpace, H5P_DEFAULT, data);
    HDF5_ERROR(errcode);

    // Close resources
    H5Sclose(fileSpace);
    H5Sclose(memSpace);
    H5Sclose(newMemSpace);
  }


  template<typename Tensor>
  void initializeTensorData(TensorData &tensorData, std::string name,
                            Tensor &tensor, ApproxType DType) {
    if (tensorData.initialized) {
      return;
    }

    auto hdftype = getHDF5TypeFromApproxType(DType);
    tensorData.hdf_native_type = hdftype;
    tensorData.approx_type = DType;
    tensorData.dset_name = name;
    auto ndim = tensor.dim();
    hsize_t dims[ndim + 1];
    hsize_t maxDims[ndim + 1];
    dims[0] = 0;
    maxDims[0] = H5S_UNLIMITED;
    for (int i = 0; i < ndim; i++) {
      dims[i + 1] = tensor.size(i);
      maxDims[i + 1] = tensor.size(i);
    }

    std::copy(dims, dims + ndim + 1, std::back_inserter(tensorData.shape));

    hid_t memSpace = H5Screate_simple(ndim + 1, dims, maxDims);
    HDF5_ERROR(memSpace);
    tensorData.memSpace = memSpace;

    hid_t pList = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_layout(pList, H5D_CHUNKED);
    hsize_t chunk_dims[ndim + 1];
    chunk_dims[0] = 1;
    for (int i = 0; i < ndim; i++) {
      chunk_dims[i + 1] = tensor.size(i);
    }
    // max chunk size is 4GB
    // we may need to reduce the chunk size.
    // For now, assume it suffices to change chunk_dims[1]
    // Yes, I know we can do 'size_bytes >> 32'
    size_t size_bytes = TensorImpl::size_bytes(tensor, DType);
    size_t reduction_factor = 1 + (size_bytes / (1ULL << 32));
    chunk_dims[1] /= reduction_factor;
    H5Pset_chunk(pList, ndim + 1, chunk_dims);
    hid_t dSetTmp = H5Dcreate(this->regionGroup, name.c_str(), hdftype, memSpace,
                              H5P_DEFAULT, pList, H5P_DEFAULT);
    HDF5_ERROR(dSetTmp);

    createTypeAttribute(dSetTmp, DType);

    tensorData.dset = dSetTmp;

    H5Pclose(pList);
    H5Sclose(memSpace);
    tensorData.initialized = true;
  }

  void initializeTensorDataChunk(TensorData &tensorData, std::string name,
                            const TensorImpl::Shape &tensor_shape, ApproxType DType, const std::vector<int64_t>& passed_chunk_vector) {
    if (tensorData.initialized) {
      return;
    }

    auto hdftype = getHDF5TypeFromApproxType(DType);
    tensorData.hdf_native_type = hdftype;
    tensorData.approx_type = DType;
    tensorData.dset_name = name;
    if (tensorData.chunk_vector.size() == passed_chunk_vector.size()) {
      std::copy(passed_chunk_vector.begin(), passed_chunk_vector.end(), tensorData.chunk_vector.begin());
    } else {
      tensorData.chunk_vector.resize(passed_chunk_vector.size());
      std::copy(passed_chunk_vector.begin(), passed_chunk_vector.end(), tensorData.chunk_vector.begin());
    }

    auto ndim = tensor_shape.size();
    hsize_t dims[ndim + 1];
    hsize_t maxDims[ndim + 1];
    dims[0] = 0;
    maxDims[0] = H5S_UNLIMITED;
    for (int i = 0; i < ndim; i++) {
      dims[i + 1] = tensor_shape[i];
      maxDims[i + 1] = tensor_shape[i];
    }

    std::copy(dims, dims + ndim + 1, std::back_inserter(tensorData.shape));

    hid_t memSpace = H5Screate_simple(ndim + 1, dims, maxDims);
    HDF5_ERROR(memSpace);
    tensorData.memSpace = memSpace;

    hid_t pList = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_layout(pList, H5D_CHUNKED);
    hsize_t chunk_dims[ndim + 1];
    chunk_dims[0] = 1;
    std::memcpy(&chunk_dims[1], passed_chunk_vector.data(), passed_chunk_vector.size() * sizeof(hsize_t));
    H5Pset_chunk(pList, ndim + 1, chunk_dims);
    hid_t dSetTmp = H5Dcreate(this->regionGroup, name.c_str(), hdftype, memSpace,
                              H5P_DEFAULT, pList, H5P_DEFAULT);
    HDF5_ERROR(dSetTmp);
    createTypeAttribute(dSetTmp, DType);

    tensorData.dset = dSetTmp;

    H5Pclose(pList);
    H5Sclose(memSpace);
    tensorData.initialized = true;
  }

  void initializeRuntime() {
    auto hdftype = H5T_NATIVE_FLOAT;
    ApproxType DType = FLOAT;
    RuntimeData.hdf_native_type = hdftype;
    RuntimeData.dset_name = "runtime";
    RuntimeData.approx_type = DType;

    hsize_t dims[1];
    hsize_t maxDims[1];
    dims[0] = 0;
    maxDims[0] = H5S_UNLIMITED;
    RuntimeData.shape.push_back(0);

    hid_t memSpace = H5Screate_simple(1, dims, maxDims);
    HDF5_ERROR(memSpace);
    RuntimeData.memSpace = memSpace;

    hid_t pList = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_layout(pList, H5D_CHUNKED);
    hsize_t chunk_dims[1] = {1024};
    H5Pset_chunk(pList, 1, chunk_dims);

    hid_t dSetTmp = H5Dcreate(this->regionGroup, RuntimeData.dset_name.c_str(), hdftype, memSpace, H5P_DEFAULT, pList, H5P_DEFAULT);
    HDF5_ERROR(dSetTmp);
    createTypeAttribute(dSetTmp, DType);
    RuntimeData.dset = dSetTmp;

    H5Pclose(pList);
    H5Sclose(memSpace);
    RuntimeData.initialized = true;
  }


private:
  void createDataSet(span<int64_t> shape, size_t chunkRows) {

  }
  void createTypeAttribute(hid_t dset, int type) {
    herr_t status;
    hid_t acpl;
    int type_int = static_cast<int>(type);
    
    hid_t attr_space = H5Screate(H5S_SCALAR);
    HDF5_ERROR(attr_space);

    hid_t attr = H5Acreate(dset, "type", H5T_NATIVE_INT, attr_space,
                           H5P_DEFAULT, H5P_DEFAULT);
    HDF5_ERROR(attr);

    status = H5Awrite(attr, H5T_NATIVE_INT, &type_int);
    HDF5_ERROR(status);

    status = H5Aclose(attr);
    HDF5_ERROR(status);
    status = H5Sclose(attr_space);
    HDF5_ERROR(status);
  }

public:
  HDF5TensorRegionView(uintptr_t Addr, const char *regionName, hid_t file);
  ~HDF5TensorRegionView();

  uintptr_t getAddr() { return addr; }
  std::string getName() { return regionName; }
  template<typename Tensor>
  void writeInputTensor(Tensor &tensor, ApproxType AT) {
    assert(expect_input && "Writing input tensor twice?");
    if(!IptTensorData.isInitialized()) {
      initializeTensorData(IptTensorData, "input", tensor, AT);
    }
    writeTensorData(IptTensorData, tensor, IptTensorData.dset_name, AT);
    expect_input = false;
  }

  template<typename Tensor>
  void writeOutputTensor(Tensor &tensor, ApproxType AT) {
    assert(!expect_input && "Writing output tensor before input tensor?");
    if(!OptTensorData.isInitialized()) {
      initializeTensorData(OptTensorData, "output", tensor, AT);
    }
    writeTensorData(OptTensorData, tensor, OptTensorData.dset_name, AT);
    expect_input = true;
  }

  template<typename Tensor>
  void writeTensor(Tensor &tensor, ApproxType AT) {
    if(expect_input) {
      writeInputTensor(tensor, AT);
    } else {
      writeOutputTensor(tensor, AT);
    }
  }

  void writeTensorChunkInit(const TensorImpl::Shape& full_shape, ApproxType AT, const std::vector<int64_t>& passed_chunk_vector, size_t num_chunks, bool is_input) {
    if(is_input) {
      initializeTensorDataChunk(IptTensorData, "input", full_shape, AT, passed_chunk_vector);
    } else {
      initializeTensorDataChunk(OptTensorData, "output", full_shape, AT, passed_chunk_vector);
    }
  }

  template<typename Tensor>
  void writeTensorChunk(Tensor &tensor, std::vector<int64_t>& loc_vector, bool is_input, bool first) {
    if(is_input) {
      writeTensorDataChunk(IptTensorData, tensor, loc_vector, first);
    } else {
      writeTensorDataChunk(OptTensorData, tensor, loc_vector, first);
    }
  }


  void writeRuntime(float runtime) {
    if (!RuntimeData.initialized) {
      initializeRuntime();
      RuntimeData.initialized = true;
    }

    auto memSpace = H5Dget_space(RuntimeData.dset);
    HDF5_ERROR(memSpace);
    auto &shape = RuntimeData.shape;
    std::vector<hsize_t> count(1);
    std::vector<hsize_t> start(1);

    // add one row to the dataset
    shape[0] += 1;

    auto errcode = H5Dset_extent(RuntimeData.dset, shape.data());
    HDF5_ERROR(errcode);

    hid_t fileSpace = H5Dget_space(RuntimeData.dset);
    HDF5_ERROR(fileSpace);

    start[0] = shape[0] - 1;
    count[0] = 1;

    errcode = H5Sselect_hyperslab(fileSpace, H5S_SELECT_SET, start.data(), NULL, count.data(), NULL);
    HDF5_ERROR(errcode);

    hid_t newMemSpace = H5Screate_simple(count.size(), count.data(), NULL);
    HDF5_ERROR(newMemSpace);

    errcode = H5Dwrite(RuntimeData.dset, RuntimeData.hdf_native_type, newMemSpace, fileSpace, H5P_DEFAULT, &runtime);
    HDF5_ERROR(errcode);

    // Close resources
    H5Sclose(fileSpace);
    H5Sclose(memSpace);
    H5Sclose(newMemSpace);
  }

};
class HDF5RegionView {
  hid_t file;
  hid_t group;
  hid_t dset;
  hid_t memSpace;
  uintptr_t addr;
  size_t totalNumRows;
  size_t totalNumCols;
  std::string Name;

private:
  int writeDataLayout(approx_var_info_t *vars, int numVars,
                      const char *groupName);
  void createDataSet(int totalElements, size_t chunkRows);

public:
  HDF5RegionView(uintptr_t Addr, const char *name, hid_t file,
                 approx_var_info_t *inputs, int numInputs,
                 approx_var_info_t *outputs, int numOutputs,
                 size_t ChunkRows);
  ~HDF5RegionView();
  void writeFeatureVecToFile(double *data, size_t numRows, int numCols);
  uintptr_t getAddr() { return addr; }
  std::string getName() { return Name; }
};

class HDF5DB : public BaseDB {
  hid_t file;
  std::vector<HDF5TensorRegionView *> regions;

public:
  HDF5DB(const char *fileName);
  ~HDF5DB();

  void *InstantiateRegion(uintptr_t Addr, const char *Name) final;

  void DataToDB(void *Region, double *Data, size_t NumRows, int NumCols) final;

  template<typename Tensor>
  void TensorToDB(void *Region, Tensor &tensor, ApproxType AT) {
    uintptr_t index = reinterpret_cast<uintptr_t>(Region);
    if (index >= regions.size()) {
      std::cout << "Index (" << index
                << " should never be larger than vector size (" << regions.size()
                << "\n";
      exit(-1);
    }
    regions[index]->writeTensor(tensor, AT);
  }

  void TensorToDBChunkInit(void *Region, const TensorImpl::Shape& full_shape, ApproxType AT, const std::vector<int64_t>& chunk_vector, size_t num_chunks, bool is_input) {
    uintptr_t index = reinterpret_cast<uintptr_t>(Region);
    if (index >= regions.size()) {
      std::cout << "Index (" << index
                << " should never be larger than vector size (" << regions.size()
                << "\n";
      exit(-1);
    }
    regions[index]->writeTensorChunkInit(full_shape, AT, chunk_vector, num_chunks, is_input);
  }

  template<typename Tensor>
  void TensorToDBChunk(void *Region, Tensor &tensor, std::vector<int64_t>& loc_vector, bool is_input, bool first) {
    uintptr_t index = reinterpret_cast<uintptr_t>(Region);
    if (index >= regions.size()) {
      std::cout << "Index (" << index
                << " should never be larger than vector size (" << regions.size()
                << "\n";
      exit(-1);
    }
    regions[index]->writeTensorChunk(tensor, loc_vector, is_input, first);
  }

  void RuntimeToDB(void *Region, float runtime) {
    uintptr_t index = reinterpret_cast<uintptr_t>(Region);
    if (index >= regions.size()) {
      std::cout << "Index (" << index
                << " should never be larger than vector size (" << regions.size()
                << "\n";
      exit(-1);
    }
    regions[index]->writeRuntime(runtime);
  }

  void RegisterMemory(const char *gName, const char *name, void *ptr,
                      size_t numBytes, HPACDType dType);
};

#undef HDF5_ERROR
#endif
