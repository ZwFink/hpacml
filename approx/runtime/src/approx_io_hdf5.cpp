#include <approx_io.h>
#include <iostream>
#include <string>

#include <hdf5.h>

#define NUM_DIMS 2
#define NUM_ROWS 16 

BaseDataWriter::~BaseDataWriter() {}

void HDF5RegionView::allocate_buffers(size_t num_rows, size_t num_cols) {
  mem = new double[num_rows * num_cols];
  ptrs = new double *[num_rows];
  for (size_t i = 0; i < num_rows; ++i) {
    ptrs[i] = &mem[i * num_cols];
  }
}

void HDF5RegionView::deallocate_buffers(size_t num_rows) {
  delete[] mem;
  for (hsize_t i = 0; i < num_rows; i++) {
    ptrs[i] = nullptr;
  }
  delete[] ptrs;
}

void HDF5RegionView::write_data_layout(approx_var_info_t *vars, int num_vars,
                                       const char *group_name) {
  herr_t status;
  int *mem = new int [num_vars* 2];
  int **data_info = new int*[num_vars];

  for (int i = 0; i < num_vars; ++i) {
    data_info[i] = &mem[i * 2];
  }

  for (int i = 0; i < num_vars; i++) {
    data_info[i][0] = vars[i].num_elem;
    data_info[i][1] = vars[i].data_type;
  }

  //  Create dataspace.
  hsize_t dimensions[2] = { (hsize_t)num_vars, (hsize_t)2 };
  int dims = 2;
  hid_t tmpspace = H5Screate_simple(dims, dimensions, NULL);
  // Create the dataset creation property list, set the layout to
  // Create the dataset.  We will use all default properties for this
  hid_t tmpdset = H5Dcreate1(group, group_name, H5T_NATIVE_INT32, tmpspace, H5P_DEFAULT);
  // Write the data to the dataset.
  status = H5Dwrite(tmpdset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                    mem);
  status = H5Dclose(tmpdset);
  status = H5Sclose(tmpspace);

  for (int i = 0; i < num_vars; i++) {
    data_info[i] = nullptr;
  }
  delete[] mem;
  delete[] data_info;

}

void HDF5RegionView::create_data_set() {
  hsize_t dims[NUM_DIMS] = {0, total_num_cols};
  hsize_t max_dims[NUM_DIMS] = {H5S_UNLIMITED, total_num_cols};
  // Create Space
  hid_t file_space = H5Screate_simple(NUM_DIMS, dims, max_dims);
  // Create Property List
  hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
  H5Pset_layout(plist, H5D_CHUNKED);
  // CHUNK_DIMS impacts performance considerably.
  hsize_t chunk_dims[NUM_DIMS] = {NUM_ROWS, total_num_cols};
  H5Pset_chunk(plist, NUM_DIMS, chunk_dims);
  std::cout << "- Property list created" << std::endl;
  // Create Dataset
  dset = H5Dcreate(group, "data", H5T_NATIVE_DOUBLE, file_space,
                   H5P_DEFAULT, plist, H5P_DEFAULT);
  std::cout << "- Dataset created" << std::endl;

  H5Sclose(file_space);
  H5Pclose(plist);

  return;
}

void HDF5RegionView::write_to_file() {
  hsize_t dims[NUM_DIMS] = {(hsize_t)current_row, (hsize_t)total_num_cols};
  hsize_t start[NUM_DIMS];
  hsize_t count[NUM_DIMS];
  if (total_num_rows == 0) {
    mem_space = H5Screate_simple(NUM_DIMS, dims, NULL);
    std::cout << "- Memory dataspace created" << std::endl;
  } else {
    H5Sset_extent_simple(mem_space, NUM_DIMS, dims, NULL);
    std::cout << "- Memory dataspace resized" << std::endl;
  }

  dims[0] = total_num_rows + current_row;
  H5Dset_extent(dset, dims);
  std::cout << "- Dataset extended" << std::endl;

  hid_t file_space = H5Dget_space(dset);
  start[0] = total_num_rows;
  start[1] = 0;
  count[0] = current_row;
  count[1] = total_num_cols;
  // Select hyperslab
  H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, count, NULL);
  std::cout << "- Next hyperslab selected" << std::endl;
  // Write hyperslab by extending it.
  H5Dwrite(dset, H5T_NATIVE_DOUBLE, mem_space, file_space, H5P_DEFAULT, mem);
  std::cout << "- Next Buffer written" << std::endl;
  // Update total number of rows written a.t.m
  total_num_rows += current_row;

  H5Sclose(file_space);
}

HDF5RegionView::HDF5RegionView(const char *name, hid_t file,
                               approx_var_info_t *inputs, int num_inputs,
                               approx_var_info_t *outputs, int num_outputs)
    : region_name(name), file(file) {
 // Create Group in the file to contain data about region
  group = H5Gcreate1(file, name , H5P_DEFAULT);

  // Count Objects and allocate memory for this region.
  total_num_cols = 0;
  total_num_rows = 0;
  for (int i = 0; i < num_inputs; i++) {
    total_num_cols += inputs[i].num_elem;
  }

  for (int i = 0; i < num_outputs; i++) {
    total_num_cols += outputs[i].num_elem;
  }

  // NUM_ROWS is a constant a.t.m set to 1024. It might lead to severe
  // performance degradation. Since the writing of the file
  // will happen once in a profiling run we can get away with it.
  allocate_buffers(NUM_ROWS, total_num_cols);
  current_row = 0;
  // Create dataset that describes this region
  const char *input_name  = "ishape";
  write_data_layout(inputs, num_inputs, input_name);
  const char *output_name  = "oshape";
  write_data_layout(outputs, num_outputs, output_name);
  // Create dataset that holds the actual data
  create_data_set();
}

HDF5RegionView::~HDF5RegionView() {
  if (current_row != 0) {
    write_to_file();
  }

  H5Sclose(mem_space);
  H5Dclose(dset);
  H5Gclose(group);
  deallocate_buffers(NUM_ROWS);
}

void HDF5RegionView::record_start(approx_var_info_t *inputs, int num_inputs) {
  if (current_row == NUM_ROWS) {
    write_to_file();
    current_row = 0;
  }

  current_col = 0;
  for (int i = 0; i < num_inputs; i++) {
    cast_and_assign(inputs[i].ptr, inputs[i].num_elem,
                    (ApproxType)inputs[i].data_type,
                    &ptrs[current_row][current_col]);
    current_col += inputs[i].num_elem;
  }
}

void HDF5RegionView::record_end(approx_var_info_t *outputs, int num_outputs) {
  for (int i = 0; i < num_outputs; i++) {
    cast_and_assign(outputs[i].ptr, outputs[i].num_elem,
                    (ApproxType)outputs[i].data_type,
                    &ptrs[current_row][current_col]);
    current_col += outputs[i].num_elem;
  }
  current_row++;
  current_col = 0;
}

HDF5DataWriter::HDF5DataWriter() {
  file = H5Fcreate("test.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
}

HDF5DataWriter::~HDF5DataWriter() {
  for (auto &it : code_regions) {
    std::cout << "Writing last data of region:" << it.first << std::endl;
    delete it.second;
  }
}

void HDF5DataWriter::record_start(const char *region_name,
                                  approx_var_info_t *inputs, int num_inputs,
                                  approx_var_info_t *outputs, int num_outputs) {
  auto ret = code_regions.insert({std::string(region_name), nullptr});
  if (ret.second == false) {
    ret.first->second->record_start(inputs, num_inputs);
  } else {
    HDF5RegionView *new_region = new HDF5RegionView(
        region_name, file, inputs, num_inputs, outputs, num_outputs);
    new_region->record_start(inputs, num_inputs);
    ret.first->second = new_region;
  }
}

void HDF5DataWriter::record_end(const char *region_name,
                                approx_var_info_t *outputs, int num_outputs) {
  auto ret = code_regions.find(std::string(region_name));
  if (ret == code_regions.end()) {
    fprintf(stderr, "This should never happen (%s:%d)\n", __FILE__, __LINE__);
    exit(0);
  } else {
    ret->second->record_end(outputs, num_outputs);
  }
}
