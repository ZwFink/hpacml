#ifndef __APPROX_DATA_UTIL__
#define __APPROX_DATA_UTIL__

#include <iostream>
#include <stdint.h>

#include <approx_internal.h>

void add(void *sum, void *augend, void *addend, ApproxType Type, size_t numElements);
void sub(void *difference, void *minuend, void *subtrahend, ApproxType Type, size_t numElements);
void multiply(void *product, void *multiplier, void *multiplicand, ApproxType Type, size_t numElements);
void divide(void *quotient, void *dividend, void *divisor, ApproxType Type, size_t numElements);
double average(void *dataPtr, size_t numElements, ApproxType Type);
const char *getTypeName(ApproxType Type);
void copyData(void *dest, void *src, size_t numElements, ApproxType Type);
float aggregate( void *vals , size_t numElements, ApproxType Type);

void convertToFloat(float *dest, void *src, size_t numElements, ApproxType Type);
void convertFromFloat(void *dest, float *src, size_t numElements, ApproxType Type);

template <typename T>
T** create2DArray(unsigned int nrows, unsigned int ncols, const T& val = T() ){
    if ( nrows == 0 ){
        printf("Invalid argument rows are 0\n");
        exit(-1);
    }

    if ( ncols == 0 ){
        printf("Invalid cols are 0\n");
    }
    T **ptr = new T*[nrows];;
    T *pool = new T[nrows*ncols]{val};
    for (unsigned int i = 0; i < nrows; ++i, pool+= ncols){
        ptr[i] = pool;
    }
    return ptr;
}

template<typename T>
void delete2DArray(T** arr){
    delete [] arr[0];
    delete [] arr;
}
#endif