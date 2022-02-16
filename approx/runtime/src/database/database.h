#ifndef __DATABASE__
#define __DATABASE__

#include <hdf5.h>
#include <vector>
#include <string>
#include <cstddef>

#include "../approx_internal.h"
#include "../include/approx.h"

class BaseDB {
public:
  virtual void *InstantiateRegion(uintptr_t Addr, const char *Name,
                                  approx_var_info_t *Inputs, int NumInputs,
                                  approx_var_info_t *Outputs,
                                  int NumOutputs,
                                  size_t ChunkRows) = 0;

  virtual void DataToDB(void *Region, double *Data, size_t NumRows,
                        int NumCols) = 0;
  virtual void RegisterMemory(const char *gName, const char *name, void *ptr,
                              size_t numBytes, HPACDType dType) = 0;
  virtual ~BaseDB() {}
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
  std::vector<HDF5RegionView *> regions;

public:
  HDF5DB(const char *fileName);
  ~HDF5DB();

  void *InstantiateRegion(uintptr_t Addr, const char *Name,
                          approx_var_info_t *Inputs, int NumInputs,
                          approx_var_info_t *Outputs, int NumOutputs,
                          size_t ChunkRows) final;

  void DataToDB(void *Region, double *Data, size_t NumRows, int NumCols) final;

  void RegisterMemory(const char *gName, const char *name, void *ptr,
                      size_t numBytes, HPACDType dType);
};

#endif