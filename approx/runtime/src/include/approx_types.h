#ifndef __APPROX_TYPES__
#define __APPROX_TYPES__


typedef struct approx_perfo_info_t {
  int type;
  int region;
  int step;
  float rate;
} approx_perfo_info_t;

typedef struct approx_var_info_t {
  void *ptr;
  size_t sz_bytes;
  size_t num_elem;
  size_t sz_elem;
  int16_t data_type;
  uint8_t dir;
} approx_var_info_t;

#endif