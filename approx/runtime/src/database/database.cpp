#include "database.h"

#include <cstdlib>
#include <iostream>
#include <sys/stat.h>

#include <H5Dpublic.h>
#include <H5Gpublic.h>
#include <H5Ppublic.h>
#include <H5Spublic.h>
#include <H5Tpublic.h>
#include <hdf5.h>

#include "../include/approx.h"

#define NUM_DIMS 2
#define NUM_ROWS (4096*16)

#define HDF5_ERROR(id)                                                         \
  if (id < 0) {                                                                \
    fprintf(stderr, "Error Happened in Line:%s:%d\n", __FILE__, __LINE__);     \
    exit(-1);                                                                  \
  }

static hid_t getHDF5DataType(HPACDType dType) {
  switch(dType) {
    case HUINT8:
      return H5T_NATIVE_SCHAR;
    case HINT8:
      return H5T_NATIVE_UCHAR;
    case HINT:
      return H5T_NATIVE_INT;
    case HFLOAT:
      return H5T_NATIVE_FLOAT;
    case HDOUBLE:
      return H5T_NATIVE_DOUBLE;
  }
}

static hsize_t getSizeInBytes(HPACDType dType) {
  switch(dType) {
    case HUINT8:
      return sizeof(char);
    case HINT8:
      return sizeof(char);
    case HINT:
      return sizeof(int);
    case HFLOAT:
      return sizeof(float);
    case HDOUBLE:
      return sizeof(double);
  }
}

static inline bool fileExists(char *FName) {
  struct stat Buf;
  return (stat(FName, &Buf) == 0);
}

static inline bool fileExists(const char *FName) {
  struct stat Buf;
  return (stat(FName, &Buf) == 0);
}

static inline bool componentExist(char *Name, hid_t Root) {
  return H5Lexists(Root, Name, H5P_DEFAULT) > 0;
}

static inline bool componentExist(const char *Name, hid_t Root) {
  return H5Lexists(Root, Name, H5P_DEFAULT) > 0;
}

hid_t createOrOpenGroup(char *RName, hid_t Root) {
  hid_t GId;
  if (componentExist(RName, Root)) {
    GId = H5Gopen1(Root, RName);
    HDF5_ERROR(GId);
  } else {
    GId = H5Gcreate1(Root, RName, H5P_DEFAULT);
    HDF5_ERROR(GId);
    if (GId < 0) {
      fprintf(stderr, "Error While Trying to create group %s\nExiting..,\n",
              RName);
      exit(-1);
    }
  }
  return GId;
}

hid_t createOrOpenGroup(const char *RName, hid_t Root) {
  hid_t GId;
  if (componentExist(RName, Root)) {
    GId = H5Gopen1(Root, RName);
    HDF5_ERROR(GId);
  } else {
    GId = H5Gcreate1(Root, RName, H5P_DEFAULT);
    HDF5_ERROR(GId);
  }
  return GId;
}

hid_t openHDF5File(const char *fileName) {
  hid_t file;
  if (fileExists(fileName)) {
    file = H5Fopen(fileName, H5F_ACC_RDWR, H5P_DEFAULT);
    HDF5_ERROR(file);
    fprintf(stderr, "Opening existing file\n");
  } else {
    file = H5Fcreate(fileName, H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
    HDF5_ERROR(file);
    if (file < 0) {
      fprintf(stderr, "Error While Opening File\n Aborting...\n");
      exit(-1);
    }
  }
  return file;
}

int HDF5RegionView::writeDataLayout(approx_var_info_t *vars, int numVars,
                                    const char *groupName) {
  herr_t status;
  int totalElements = 0;
  int *mem = new int[numVars * 2];
  int **dataInfo = new int *[numVars];

  for (int i = 0; i < numVars; ++i) {
    dataInfo[i] = &mem[i * 2];
  }

  for (int i = 0; i < numVars; i++) {
    dataInfo[i][0] = vars[i].num_elem;
    dataInfo[i][1] = vars[i].data_type;
    totalElements += dataInfo[i][0];
  }

  hsize_t dimensions[2] = {(hsize_t)numVars, (hsize_t)2};
  int dims = 2;
  hid_t tmpspace = H5Screate_simple(dims, dimensions, NULL);
  HDF5_ERROR(tmpspace);
  hid_t tmpdset =
      H5Dcreate1(group, groupName, H5T_NATIVE_INT32, tmpspace, H5P_DEFAULT);
  HDF5_ERROR(tmpdset);
  status =
      H5Dwrite(tmpdset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, mem);
  HDF5_ERROR(status);
  status = H5Dclose(tmpdset);
  HDF5_ERROR(status);
  status = H5Sclose(tmpspace);
  HDF5_ERROR(status);

  for (int i = 0; i < numVars; i++) {
    dataInfo[i] = nullptr;
  }

  delete[] mem;
  delete[] dataInfo;
  return totalElements;
}

void HDF5RegionView::writeFeatureVecToFile(double *data, int numRows,
                                           int numCols) {
  hsize_t dims[NUM_DIMS] = {(hsize_t)numRows, (hsize_t)numCols};
  hsize_t start[NUM_DIMS];
  hsize_t count[NUM_DIMS];
  if (totalNumRows == 0)
    memSpace = H5Screate_simple(NUM_DIMS, dims, NULL);
  else
    H5Sset_extent_simple(memSpace, NUM_DIMS, dims, NULL);

  dims[0] = totalNumRows + numRows;
  H5Dset_extent(dset, dims);

  hid_t fileSpace = H5Dget_space(dset);
  start[0] = totalNumRows;
  start[1] = 0;
  count[0] = numRows;
  count[1] = numCols;
  // Select hyperslab
  H5Sselect_hyperslab(fileSpace, H5S_SELECT_SET, start, NULL, count, NULL);
  H5Dwrite(dset, H5T_NATIVE_DOUBLE, memSpace, fileSpace, H5P_DEFAULT, data);
  totalNumRows += numRows;

  H5Sclose(fileSpace);
}

void HDF5RegionView::createDataSet(int totalElements) {
  hsize_t dims[NUM_DIMS] = {0, static_cast<hsize_t>(totalElements)};
  hsize_t maxDims[NUM_DIMS] = {H5S_UNLIMITED,
                               static_cast<hsize_t>(totalElements)};
  hid_t fileSpace = H5Screate_simple(NUM_DIMS, dims, maxDims);
  hid_t pList = H5Pcreate(H5P_DATASET_CREATE);
  H5Pset_layout(pList, H5D_CHUNKED);
  // CHUNK_DIMS impacts performance considerably.
  // TODO: give an option to the config file
  // for this option to work out.
  hsize_t chunk_dims[NUM_DIMS] = {NUM_ROWS,
                                  static_cast<hsize_t>(totalElements)};
  H5Pset_chunk(pList, NUM_DIMS, chunk_dims);
  dset = H5Dcreate(group, "data", H5T_NATIVE_DOUBLE, fileSpace, H5P_DEFAULT,
                   pList, H5P_DEFAULT);
  H5Sclose(fileSpace);
  H5Pclose(pList);
  return;
}

HDF5RegionView::HDF5RegionView(uintptr_t rAddr, const char *name, hid_t file,
                               approx_var_info_t *inputs, int numInputs,
                               approx_var_info_t *outputs, int numOutputs)
    : file(file), totalNumRows(0), totalNumCols(0), Name(name){
  group = createOrOpenGroup(name, file);

  int tmpNumInputs = writeDataLayout(inputs, numInputs, "ishape");
  int tmpNumOutputs = writeDataLayout(outputs, numOutputs, "oshape");
  createDataSet(tmpNumInputs + tmpNumOutputs);
  addr = rAddr;
  return;
}

HDF5RegionView::~HDF5RegionView() {
  H5Sclose(memSpace);
  H5Dclose(dset);
  H5Gclose(group);
}

HDF5DB::HDF5DB(const char *fileName) {
  file = openHDF5File(fileName);
  HDF5_ERROR(file);
}

void *HDF5DB::InstantiateRegion(uintptr_t addr, const char *name,
                                approx_var_info_t *inputs, int numInputs,
                                approx_var_info_t *outputs, int numOutputs) {
  for (auto it = regions.begin(); it != regions.end(); ++it) {
    if ((*it)->getAddr() == addr && (*it)->getName() == name ) {
      long index = it - regions.begin();
      return reinterpret_cast<void *>(index);
    }
  }
  uintptr_t index = reinterpret_cast<uintptr_t>(regions.size());
  regions.push_back(new HDF5RegionView(addr, name, file, inputs, numInputs,
                                       outputs, numOutputs));
  return reinterpret_cast<void *>(index);
}

void HDF5DB::DataToDB(void *region, double *data, int numRows, int numCols) {
  uintptr_t index = reinterpret_cast<uintptr_t>(region);
  if (index >= regions.size()) {
    std::cout << "Index (" << index
              << " should never be larger than vector size (" << regions.size()
              << "\n";
    exit(-1);
  }
  regions[index]->writeFeatureVecToFile(data, numRows, numCols);
}

HDF5DB::~HDF5DB() {
  for (auto &it : regions)
    delete it;

  H5Fclose(file);
}

void HDF5DB::RegisterMemory(const char *gName, const char *name, void *ptr,
                            size_t numBytes, HPACDType dType) {
  herr_t status;
  hsize_t bytes = getSizeInBytes(dType);
  hsize_t elements = numBytes / bytes;
  hid_t rId = createOrOpenGroup(gName, file);
  int dims = 1;
  hid_t tmpspace = H5Screate_simple(dims, &elements, NULL);
  HDF5_ERROR(tmpspace);
  hid_t tmpdset =
      H5Dcreate1(rId, name, getHDF5DataType(dType), tmpspace, H5P_DEFAULT);
  status = H5Dwrite(tmpdset, getHDF5DataType(dType), H5S_ALL, H5S_ALL,
                    H5P_DEFAULT, ptr);
  HDF5_ERROR(status);
  status = H5Dclose(tmpdset);
  HDF5_ERROR(status);
  status = H5Sclose(tmpspace);
  HDF5_ERROR(status);
  status = H5Gclose(rId);
  HDF5_ERROR(status);
}