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

    #pragma approx declare tensor_functor(ipt_functor: [j, 0:1] = ([j:j+ovlp_size]))
    #pragma approx declare tensor(ipt_tens: ipt_functor(input[0:n]))

    #pragma approx declare tensor_functor(opt_functor: [j, 0:1] = ([j]))

    #pragma approx ml(infer) in(ipt_tens) out(opt_functor(output[0:n*ovlp_size]))
    {
    }

    for(int i = 0; i < n; i++) {
        for(int j = 0; j < ovlp_size; j++) {
            auto val = output[i*ovlp_size+j];
            int expected = (i / ovlp_size) + j;
            EXPECT_EQ(val, i+j) << "Mismatch at index " << i*ovlp_size+j;
        }
    }

    delete[] input;
    delete[] output;
}