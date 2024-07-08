#include <gtest/gtest.h>
#include "approx.h"
TEST(HPACML, Input1D) {
    int n = 10;
    int *input = new int[n];
    int *output = new int[n];

    for(int i = 0; i < n; i++) {
        input[i] = i;
    }
    for(int i = 0; i < n; i++) {
        output[i] = -1;
    }

    #pragma approx declare tensor_functor(ipt_functor: [i, 0:1] = ([i]))
    #pragma approx declare tensor(ipt_tens: ipt_functor(input[0:n]))

    #pragma approx declare tensor_functor(opt_functor: [i, 0:1] = ([i]))

    #pragma approx ml(infer) in(ipt_tens) out(opt_functor(output[0:n]))
    {
    }

    for(int i = 0; i < n; i++) {
            auto val = output[i];
            EXPECT_EQ(val, i) << "Mismatch at index " << i;
    }

    delete[] input;
    delete[] output;

}