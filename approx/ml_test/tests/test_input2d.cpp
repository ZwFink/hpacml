#include <gtest/gtest.h>
#include "approx.h"

TEST(HPACML, Input2D) {

    int n = 10;
    int m = 10;
    int blocksize = 3;
    int *input = new int[n*m];
    int *output = new int[n*m];

    std::fill(input, input + (m*n), -1);
    std::fill(output, output + (m*n), -1);

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            input[i*m + j] = i*m + j;
        }
    }

    #pragma approx declare tensor_functor(ipt_functor: [i, j, 0:9] = ([i, j]))
    #pragma approx declare tensor(ipt_tens: ipt_functor(input[0:n, 0:m]))

    #pragma approx declare tensor_functor(opt_functor: [i, 0:1] = ([i]))

    #pragma approx ml(infer) in(ipt_tens) out(opt_functor(output[0:n*m]))
    {
    }

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < m; j++) {
            auto val = output[i*m+j];
            auto expected = i*m + j;
            EXPECT_EQ(val, expected) << "Mismatch at index " << i << ", " << j;
        }
    }

    delete[] input;
    delete[] output;
}
