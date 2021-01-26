#include <cstdlib>
#include <hdf5.h>

#include "io_helpers.h"

hid_t openHDF5File(const char *FName){
  hid_t file = H5Fopen(FName, H5F_ACC_RDWR, H5P_DEFAULT);
  if (file < 0 ){
    file =  H5Fcreate(FName, H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
    if ( file < 0 ){
        fprintf(stderr, "Error While Opening File\n Aborting...\n");
        exit(-1);
        return file;
    }
  }
  return file;
}