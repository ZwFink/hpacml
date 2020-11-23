#include <stdint.h>
#include <stdio.h>
#include <string>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <unordered_map>
#ifdef ENABLE_HDF5
#include <hdf5.h>
#endif

#include <approx.h>
#include <approx_data_util.h>
#include <approx_internal.h>

using namespace std;

#define MEMO_IN 1
#define MEMO_OUT 2

#define NUM_ROWS 1024
#define NUM_DIMS 2

enum ExecuteMode: uint8_t{
  EXECUTE,
  PROFILE_TIME,
  PROFILE_DATA
};

class ApproxRuntimeConfiguration{
  ExecuteMode Mode;
  public:
    ApproxRuntimeConfiguration(){
      const char *env_p = std::getenv("EXECUTE_MODE");
      if (!env_p){
        Mode = EXECUTE;
        return;
      }

      if (strcmp(env_p, "TIME_PROFILE") == 0){
        Mode = PROFILE_TIME;
      }
      else if ( strcmp(env_p, "DATA_PROFILE") == 0 ){
        Mode = PROFILE_DATA; 
      }
      else{
        Mode = EXECUTE;
      }
    }

    ~ApproxRuntimeConfiguration(){
    }

    ExecuteMode getMode(){return Mode;}
};

ApproxRuntimeConfiguration RTEnv;

void _printdeps(approx_var_info_t *vars, int num_deps) {
  for (int i = 0; i < num_deps; i++) {
    printf("%p, NE:%ld, SE:%ld, DT:%s, DIR:%d\n", vars[i].ptr, vars[i].num_elem,
           vars[i].sz_elem, getTypeName((ApproxType)vars[i].data_type),
           vars[i].dir);
  }
}

class Profiler{
  std::unordered_map<const char *, std::pair<long, double>> profileData;
  std::unordered_map<const char *, std::chrono::time_point<std::chrono::high_resolution_clock>> startTime;
  public:
    Profiler(){}
    ~Profiler() {
     for (auto p : profileData) {
       std::string region_name = p.first;
       double avg_time = p.second.second/ p.second.first;
       std::cout<< region_name << ":" << avg_time << ":" << p.second.second << ":" << p.second.first <<"\n";
     }
    }
    void start_time(const char *val){
      startTime[val] = std::chrono::high_resolution_clock::now(); 
    }
    void stop_time (const char *val){
      std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now(); 
      std::chrono::duration<double> elapsed_time = (end - startTime[val]);
      std::unordered_map<const char *, std::pair<long, double>>::iterator iter = profileData.find(val); 
      if (iter == profileData.end() )
        profileData.insert(std::make_pair(val,std::make_pair(1,elapsed_time.count())));
      else{
        iter->second.second += elapsed_time.count();
        iter->second.first++;
      }
    }
};

#ifdef ENABLE_HDF5
class DataWriter{
  hid_t file;
  hid_t file_space;
  hid_t dset;
  hid_t mem_space;
  double *mem;
  double **ptrs; 
  int current_row;
  int current_col;
  size_t total_num_rows;
  size_t total_num_cols;
  bool is_file_open;
  private:
    void allocate_buffers(size_t num_rows, size_t num_cols){
    mem = new double [num_rows* num_cols];
    ptrs = new double*[num_rows];
    for (size_t i = 0; i < num_rows; ++i){
        ptrs [i] = &mem[i * num_cols];
    }
  }

  void deallocate_buffers(size_t num_rows){
    delete [] mem;
    for ( hsize_t i = 0 ; i < num_rows; i++){
      ptrs[i] = nullptr;
    }
  }

  public:
    DataWriter(){
      mem=nullptr;
      ptrs = nullptr;
      total_num_rows = 0;
      total_num_cols = 0;
      is_file_open = false;
    }    

    ~DataWriter(){

      if ( current_row != 0 ){
        write_hdf5_file();
      }

      H5Dclose(dset);
      H5Fclose(file);
      deallocate_buffers(NUM_ROWS);
    }

    void write_hdf5_file(){
      hsize_t dims[NUM_DIMS];
      hsize_t start[NUM_DIMS];
      hsize_t count[NUM_DIMS];
      if (!is_file_open) {
          is_file_open = true;
          // Create HDF5 file
          file = H5Fcreate("test.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
          std::cout << "- File created" << std::endl;
          // Create Dataspace
          dims[0]  = 0;
          dims[1] =  total_num_cols;
          std::cout << "Total Number of cols are " << total_num_cols << std::endl;

          hsize_t max_dims[NUM_DIMS] = {H5S_UNLIMITED, total_num_cols};
          file_space = H5Screate_simple(NUM_DIMS, dims, max_dims);
          std::cout << "- Dataspace created" << std::endl;

          // Create Property List
          hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
          H5Pset_layout(plist, H5D_CHUNKED);

          // CHUNK_DIMS impacts performance considerably.
          hsize_t chunk_dims[NUM_DIMS] = {NUM_ROWS, total_num_cols};
          H5Pset_chunk(plist, NUM_DIMS, chunk_dims);
          std::cout << "- Property list created" << std::endl;

          // Create Dataset
          dset = H5Dcreate(file, "dataset", H5T_NATIVE_DOUBLE, file_space, H5P_DEFAULT, plist, H5P_DEFAULT);
          std::cout << "- Dataset created" << std::endl;

          // Create Memory Space
          dims[0] = current_row;
          dims[1] = total_num_cols;
          mem_space = H5Screate_simple(NUM_DIMS, dims, NULL);
          std::cout << "- Memory dataspace created" << std::endl;

          H5Dset_extent(dset, dims);

          file_space = H5Dget_space(dset);
          hsize_t start[2] = {0, 0};
          hsize_t count[2] = {(hsize_t)current_row, (hsize_t)total_num_cols };
          H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, count, NULL);
          std::cout << "- First hyperslab selected" << std::endl;


          std::cout<<"Writing space to " << current_row << " " << total_num_cols << std::endl;
          H5Dwrite(dset, H5T_NATIVE_DOUBLE, mem_space, file_space, H5P_DEFAULT, mem);
          std::cout << "- First buffer written" << std::endl;

          H5Sclose(file_space);
          H5Pclose(plist);
          is_file_open = true;
          total_num_rows = current_row;
          return;
      }

      dims[0] = current_row;
      dims[1] = total_num_cols;
      H5Sset_extent_simple(mem_space, NUM_DIMS, dims, NULL);
      std::cout << "- Memory dataspace resized" << std::endl;

      dims[0] = total_num_rows + current_row;
      H5Dset_extent(dset, dims);
      std::cout << "- Dataset extended" << std::endl;

      file_space = H5Dget_space(dset);
      start[0] = total_num_rows;
      start[1] = 0;
      count[0] = current_row;
      count[1] = total_num_cols;
      H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start, NULL, count, NULL);
      std::cout << "- Next hyperslab selected" << std::endl;

      H5Dwrite(dset, H5T_NATIVE_DOUBLE, mem_space, file_space, H5P_DEFAULT, mem);
      std::cout << "- Next Buffer written" << std::endl;

      total_num_rows += current_row;

      H5Sclose(file_space);
    }

    void record_start(const char *region_name, approx_var_info_t *inputs, int num_inputs, approx_var_info_t *outputs, int num_outputs){
      // First time entering code region
      std::cout<<"Starting Recording\n";
      if ( total_num_cols == 0){
        total_num_cols = 0;
        for (int i = 0; i < num_inputs; i++){
          total_num_cols += inputs[i].num_elem; 
        }

        for (int i = 0; i < num_outputs; i++){
          total_num_cols += outputs[i].num_elem; 
        }
        
        allocate_buffers(NUM_ROWS, total_num_cols);
        current_row = 0;    
      } 

      if ( current_row == NUM_ROWS){
        std::cout<<"I am writting to file on 243"<<std::endl;
        write_hdf5_file();
        current_row = 0;
      }

      current_col= 0;
      for ( int i = 0; i < num_inputs; i++){
        cast_and_assign(inputs[i].ptr, inputs[i].num_elem, (ApproxType) inputs[i].data_type, &ptrs[current_row][current_col]); 
        current_col+=inputs[i].num_elem;
      }
    }

    void record_end(const char *region_name, approx_var_info_t *outputs, int num_outputs){
      for ( int i = 0; i < num_outputs; i++){
        cast_and_assign(outputs[i].ptr, outputs[i].num_elem, (ApproxType) outputs[i].data_type, &ptrs[current_row][current_col]); 
        current_col += outputs[i].num_elem;
      }
      // Reset.
      current_row++;
      current_col = 0;
    }
};
DataWriter data_profiler;
#endif

Profiler profiler;

void __approx_exec_call(void (*accurate)(void *), void (*perforate)(void *),
                        void *arg, bool cond,  const char *region_name, void *perfoArgs, int memo_type,
                        void *inputs, int num_inputs, void *outputs,
                        int num_outputs) {
  approx_perfo_info_t *perfo = (approx_perfo_info_t *)perfoArgs;
  approx_var_info_t *input_vars = (approx_var_info_t *)inputs;
  approx_var_info_t *output_vars = (approx_var_info_t *)outputs;
  std::cout << "Inputs are:" << std::endl;
  _printdeps(input_vars, num_inputs);
  std::cout << "Outputs are:" << std::endl;
  _printdeps(output_vars, num_outputs);

  if (RTEnv.getMode() == PROFILE_TIME){
    profiler.start_time(region_name);
    accurate(arg);
    profiler.stop_time(region_name);
    return;
  } 
#ifdef ENABLE_HDF5
  else if (RTEnv.getMode() == PROFILE_DATA){
    data_profiler.record_start(region_name, input_vars, num_inputs, output_vars, num_outputs);
    accurate(arg);
    data_profiler.record_end(region_name, output_vars, num_outputs);
    return;
  }
#endif
  else{
    if (cond) {
      if (memo_type == MEMO_IN) {
        memoize_in(accurate, arg, input_vars, num_inputs, output_vars, num_outputs);
      } else if (memo_type == MEMO_OUT) {
        memoize_out(accurate, arg, output_vars, num_outputs);
      }
    } else {
      accurate(arg);
    }
  }
}
