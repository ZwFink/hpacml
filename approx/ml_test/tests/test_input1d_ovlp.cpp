#include <gtest/gtest.h>
#include "approx.h"



TEST(HPACML, Input1DOverlapping) {

    int n = 10;
    int ovlp_size = 3;
    int *input = new int[n+ovlp_size];
    int *output = new int[n*ovlp_size];

    for(int i = 0; i < n+ovlp_size; i++) {
        input[i] = i;
    }
    for(int i = 0; i < n*ovlp_size; i++) {
        output[i] = -1;
    }

    std::cout << "Hello world!\n";
    #pragma approx declare tensor_functor(ipt_functor2: [j, 0:1] = ([j:j+ovlp_size]))
    #pragma approx declare tensor(ipt_tens2: ipt_functor2(input[0:n]))

    #pragma approx declare tensor_functor(opt_functor2: [j, 0:1] = ([j]))

    #pragma approx ml(infer) in(ipt_tens2) out(opt_functor2(output[0:n*ovlp_size]))
    {
    }

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < ovlp_size; j++) {
            auto val = output[i*ovlp_size+j];
            // int expected = (i / ovlp_size) + j;
            int expected = 5;
            EXPECT_EQ(val, i+j) << "Mismatch at index " << i*ovlp_size+j;
        }
    }

    delete[] input;
    delete[] output;
}



// TEST(HPACML, Broadcast) {
//     int N = 4;
//     int UB = N*N*N*N;
//     float *data = new float[UB];
//     int *access1 = new int[UB];
//     long *access2 = new long[UB];


//     for(int i = 0; i < UB; i++){
//         data[i] = i;
//         access1[i] = N-1-i;
//         access2[i] = i;
//     }


//     #pragma approx declare tensor_functor(fn: [i, 0:6] = ([i*3:i*3+3], [i*2:i*2+2], [i] ))
//     int fgh = 5;
//     #pragma approx declare tensor(ten: fn(data[0:1], data[0:2*N], data[10:11]))

//     float *nrows = data;
//     {
//     #pragma approx declare tensor_functor(cnnipt: [niter, x, y, z, 0:2] = ([niter, x, y, z], [niter, x,y,z]))
//     #pragma approx declare tensor(cnnten: cnnipt(data[0:N, 0:N, 0:N, 0:N], data[access1[0:N,0:N,0:N,0:N]]))
//     }

//     float scalar = 50;
//     float *scalar_ptr = &scalar;
//     #pragma approx declare tensor_functor(broadcast_scalar: [i, 0:2] = ([i], [i]))
//     #pragma approx declare tensor(broadcast_scalar_ten: broadcast_scalar(data[0:N], scalar_ptr[0:1]))
// }

// TEST(HPACML, Input4D) {
//     int N = 4;
//     int UB = N*N*N*N;
//     float *data = new float[UB];
//     int *access1 = new int[UB];
//     long *access2 = new long[UB];


//     for(int i = 0; i < UB; i++){
//         data[i] = i;
//         access1[i] = N-1-i;
//         access2[i] = i;
//     }

//     std::cout << "Hello world!\n";
//     #pragma approx declare tensor_functor(cnnipt: [niter, x, y, z, 0:1] = ([niter, x, y, z]))
//     #pragma approx declare tensor_functor(cnnopt: [niter, x, y, z, 0:1] = ([niter, x, y, z]))
//     #pragma approx declare tensor(cnnten: cnnipt(data[0:N, 0:N, 0:N, 0:N]))

//     #pragma approx ml(infer) in(cnnten) out(cnnopt(data[0:N, 0:N, 0:N, 0:N]))
//     {
//     }

//     std::cout << "Hello world!\n";
//     for(int i = 0; i < UB; i++) {
//         auto val = data[i];
//         auto expected = i;
//         std::cout << "Val at " << i << " is " << val << " expected " << expected << "\n";
//         // EXPECT_EQ(val, expected) << "Mismatch at index " << i;
//     }

//     for(int i = 0; i < UB; i++) {
//         auto val = data[i];
//         auto expected = i;
//         EXPECT_EQ(val, expected) << "Mismatch at index " << i;
//     }
// }