#include <cstdlib>
#include <hdf5.h>

#include "H5Dpublic.h"
#include "H5Gpublic.h"
#include "H5Ppublic.h"
#include "H5Spublic.h"
#include "H5Tpublic.h"
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

hid_t createOrOpenGroup(char *RName, hid_t Root){
  hid_t GId = H5Gopen1(Root, RName);
  if (GId < 0){
    GId = H5Gcreate1(Root, RName, H5P_DEFAULT);
    if (GId < 0 ){
      fprintf(stderr, "Error While Trying to create group %s\nExiting..,\n", RName);
      exit(-1);
    }
  }
  return GId;
}

void writeProfileData(char *Name, hid_t Root, double Value){
  hid_t DSet = H5Dopen2(Root, Name, H5P_DEFAULT);
  if (DSet < 0){
    const int NDims = 1;
    hsize_t Dims[NDims] = {1};
    hsize_t MDims[NDims] = {H5S_UNLIMITED};
    herr_t Status;
    hsize_t CDims[NDims] = {1};

    hid_t DSpace = H5Screate_simple(NDims, Dims, MDims);
    hid_t Prop = H5Pcreate(H5P_DATASET_CREATE);
    Status = H5Pset_chunk(Prop, NDims, CDims);
    DSet = H5Dcreate(Root, Name, H5T_NATIVE_DOUBLE, DSpace, H5P_DEFAULT, Prop, H5P_DEFAULT);
    Status = H5Dwrite(DSet, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &Value);
    H5Pclose(Prop);
    H5Sclose(DSpace);
  }
  else{
    hid_t DSpace = H5Dget_space(DSet);
    int NDims = H5Sget_simple_extent_ndims(DSpace);
    if (NDims != 1){
      fprintf(stderr, "Dimensions should be 1 in profiling data\nExiting...\n");
      exit(-1);
    }
    hsize_t Dims;
    H5Sget_simple_extent_dims(DSpace, &Dims, NULL);
    hsize_t Extend = 1;
    hsize_t EDims =  Dims + Extend;
    herr_t Status = H5Dextend(DSet, &EDims);
    hid_t FSpace = H5Dget_space(DSet);
    Status = H5Sselect_hyperslab(FSpace, H5S_SELECT_SET, &Dims, NULL, &Extend, NULL);
    hid_t MSpace = H5Screate_simple(NDims, &Extend, NULL);
    Status = H5Dwrite(DSet, H5T_NATIVE_DOUBLE, MSpace, FSpace, H5P_DEFAULT, (void *) &Value);
    Status = H5Sclose(DSpace);
    Status = H5Sclose(MSpace);
    Status = H5Sclose(FSpace);
  }
  H5Dclose(DSet);
  return;
}