#include <cstdlib>
#include <hdf5.h>
#include <sys/stat.h>

#include "H5Dpublic.h"
#include "H5Gpublic.h"
#include "H5Ppublic.h"
#include "H5Spublic.h"
#include "H5Tpublic.h"

#include "io_helpers.h"

inline bool fileExists(char *FName) {
    struct stat Buf;   
    return (stat(FName, &Buf) == 0); 
}

inline bool fileExists(const char *FName) {
    struct stat Buf;   
    return (stat(FName, &Buf) == 0); 
}

bool componentExist(char *Name, hid_t Root) {
    return H5Lexists(Root, Name, H5P_DEFAULT) > 0;
}

hid_t openHDF5File(const char *FName) {
    hid_t file;
    if (fileExists(FName)){
        file = H5Fopen(FName, H5F_ACC_RDWR, H5P_DEFAULT);
    }
    else{
        file = H5Fcreate(FName, H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
        if (file < 0) {
            fprintf(stderr, "Error While Opening File\n Aborting...\n");
            exit(-1);
        }
    }
    return file;
}

hid_t createOrOpenGroup(char *RName, hid_t Root) {
    hid_t GId;
    if (componentExist(RName, Root)) {
        GId = H5Gopen1(Root, RName);
    } else {
        GId = H5Gcreate1(Root, RName, H5P_DEFAULT);
        if (GId < 0) {
            fprintf(stderr, "Error While Trying to create group %s\nExiting..,\n",
                    RName);
            exit(-1);
        }
    }
    return GId;
}

void writeProfileData(char *Name, hid_t Root, double *Value, int NumStats) {
    if (!componentExist(Name, Root)) {
        const int NDims = 2;
        hsize_t Dims[NDims] = {1, (hsize_t) NumStats};
        hsize_t MDims[NDims] = {H5S_UNLIMITED, (hsize_t) NumStats};
        herr_t Status;
        hsize_t CDims[NDims] = {1, (hsize_t) NumStats};

        hid_t DSpace = H5Screate_simple(NDims, Dims, MDims);
        hid_t Prop = H5Pcreate(H5P_DATASET_CREATE);
        Status = H5Pset_chunk(Prop, NDims, CDims);
        hid_t DSet = H5Dcreate(Root, Name, H5T_NATIVE_DOUBLE, DSpace, H5P_DEFAULT,
                Prop, H5P_DEFAULT);
        Status = H5Dwrite(DSet, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                Value);
        H5Pclose(Prop);
        H5Sclose(DSpace);
        Status = H5Dclose(DSet);
    } else {
        hid_t DSet = H5Dopen2(Root, Name, H5P_DEFAULT);
        hid_t DSpace = H5Dget_space(DSet);
        int NDims = H5Sget_simple_extent_ndims(DSpace);
        if (NDims != 2) {
            fprintf(stderr, "Dimensions should be 2 in profiling data\nExiting...\n");
            exit(-1);
        }
        hsize_t Dims[2];
        H5Sget_simple_extent_dims(DSpace, Dims, NULL);
        hsize_t Extend[2] = { 1, (hsize_t) NumStats};
        hsize_t EDims[2];
        hsize_t Start[2] = {Dims[1],0};
        EDims[0] = Dims[0] + Extend[0];
        EDims[1] = NumStats;

        herr_t Status = H5Dextend(DSet, EDims);
        hid_t FSpace = H5Dget_space(DSet);
        Status =
            H5Sselect_hyperslab(FSpace, H5S_SELECT_SET, Start, NULL, Extend, NULL);
        hid_t MSpace = H5Screate_simple(NDims, Extend, NULL);
        Status = H5Dwrite(DSet, H5T_NATIVE_DOUBLE, MSpace, FSpace, H5P_DEFAULT,
                (void *)Value);
        Status = H5Sclose(DSpace);
        Status = H5Sclose(MSpace);
        Status = H5Sclose(FSpace);
        Status = H5Dclose(DSet);
    }
    return;
}
