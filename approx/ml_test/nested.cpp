#include "approx.h"
#include "approx_debug.h"
#include <iostream>
#include <vector>

// #define N 100


int main() {
    int N = 10;
    float *data = new float[N*N];
    int *access1 = new int[N*N];
    long *access2 = new long[N*N];


    for(int i = 0; i < N; i++){
        data[i] = i;
        access1[i] = N-1-i;
        access2[i] = i;
    }


    std::cout << "data ptr is: " << (void*) data << std::endl;
    std::cout << "access 1 ptr is: " << (void*) access1 << std::endl;
    std::cout << "access 2 ptr is: " << (void*) access2 << std::endl;
    #pragma approx declare tensor_functor(fn: [i] = ([i]))
    #pragma approx declare tensor(ten: fn(data[access1[access2[0:N]]]))

    for(int i = 0; i < N*N; i++){
        data[i] = i;
        access1[i] = (N*N)-1-i;
        access2[i] = access1[i];
    }

    #pragma approx declare tensor_functor(fn2: [i,j] = ([i,j]))
    #pragma approx declare tensor(ten2: fn2(data[access1[access2[0:N,0:N]]]))

}
