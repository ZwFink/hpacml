#ifndef __IO_HELPERS__
#define __IO_HELPERS__
hid_t createOrOpenGroup(char *RName, hid_t Root);
hid_t openHDF5File(const char *FName);
void writeProfileData(char *Name, hid_t Root, double Value);
#endif