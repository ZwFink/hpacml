#include "approx.h"
#include "approx_debug.h"
#include <iostream>
#include <vector>

// #define N 100


int main() {
    int N = 4;
    int UB = N*N*N*N;
    float *data = new float[UB];
    int *access1 = new int[UB];
    long *access2 = new long[UB];


    for(int i = 0; i < UB; i++){
        data[i] = i;
        access1[i] = N-1-i;
        access2[i] = i;
    }


    #pragma approx declare tensor_functor(fn: [i, 0:6] = ([i*3:i*3+3], [i*2:i*2+2], [i] ))
    int fgh = 5;
    #pragma approx declare tensor(ten: fn(data[0:1], data[0:2*N], data[10:11]))

    float *nrows = data;
    {
    #pragma approx declare tensor_functor(cnnipt: [niter, x, y, z, 0:2] = ([niter, x, y, z], [niter, x,y,z]))
    #pragma approx declare tensor(cnnten: cnnipt(data[0:N, 0:N, 0:N, 0:N], data[access1[0:N,0:N,0:N,0:N]]))
    }

    float scalar = 50;
    float *scalar_ptr = &scalar;
    #pragma approx declare tensor_functor(broadcast_scalar: [i, 0:2] = ([i], [i]))
    #pragma approx declare tensor(broadcast_scalar_ten: broadcast_scalar(data[0:N], scalar_ptr[0:1]))

}

