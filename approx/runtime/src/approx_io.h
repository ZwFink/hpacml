#ifndef __APPROX_IO__
#define __APPROX_IO__
#include <approx_internal.h>
#include <unordered_map>

#ifdef ENABLE_HDF5
#include <hdf5.h>
#endif
// This is Base Writer Class. The class records all the input/output data as the
// function is invoced.
class BaseDataWriter {
public:
  BaseDataWriter() {};
  virtual ~BaseDataWriter(const char *file_name) = 0;
  virtual void record_start(const char *region_name, approx_var_info_t *inputs,
                            int num_inputs, approx_var_info_t *outputs,
                            int num_outputs) = 0;

  virtual void record_end(const char *region_name, approx_var_info_t *outputs,
                          int num_outputs) = 0;
};

#ifdef ENABLE_HDF5
class HDF5RegionView {
  std::string region_name;
  hid_t file;
  hid_t group;
  hid_t dset;
  hid_t mem_space;
  double *mem;
  double **ptrs;
  int current_row;
  int current_col;
  size_t total_num_rows;
  size_t total_num_cols;

private:
  void allocate_buffers(size_t num_rows, size_t num_cols);
  void deallocate_buffers(size_t num_rows);
  void write_data_layout(approx_var_info_t *vers, int num_vars,
                         const char *group_name);
  void create_data_set();
  void write_to_file();

public:
  HDF5RegionView(const char *name, hid_t file, approx_var_info_t *inputs,
                 int num_inputs, approx_var_info_t *outputs, int num_outputs);
  ~HDF5RegionView();
  void record_start(approx_var_info_t *inputs, int num_inputs);
  void record_end(approx_var_info_t *outputs, int num_outputs);
};

// HDF5 output File Format. This will be the
// backbone of implementing any
// training/profiling method.
// The file format is as follows:
// There are 2 datasets for each approximate region.
// The first dataset contains information about the dimensions and types
// of the input/output variables.
// The second dataset contains the input-output pairs for each invocation
// of the executing region.
class HDF5DataWriter : public BaseDataWriter {
  hid_t file;
  std::unordered_map<std::string, HDF5RegionView *> code_regions;
public:
  HDF5DataWriter(const char *file_name);
  ~HDF5DataWriter();
  void record_start(const char *region_name, approx_var_info_t *inputs,
                    int num_inputs, approx_var_info_t *outputs,
                    int num_outputs);
  void record_end(const char *region_name, approx_var_info_t *outputs,
                  int num_outputs);
};
#endif
#endif