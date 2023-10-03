#include "approx.h"
#include "approx_debug.h"
#include <iostream>
#include <vector>

#define N 100


int main() {
    int *data = new int[N];
    // int *opt_first = new int[N];
    // int *opt_second = new int[N];
    int *access1 = new int[N];
    int *access2 = new int[N];
    // int *access3 = new int[N];
    // std::vector<int*> accesses;
    // accesses.push_back(access2);
    // accesses.push_back(access1);
    // accesses.push_back(data);


    // for(int i = 0; i < N; i++) {
    //     opt_first[i] = -1;
    //     opt_second[i] = -1;
    //     data[i] = i;
    //     access1[i] = 99-i;
    //     access2[i] = 99-i;
    //     access3[i] = i;
    // }

    // for(int i = 0; i < N; i++) {
    //     opt_first[i] = data[access2[access1[i]]];
    // }
    // for(int i = 0; i < N; i++) {
    //     std::cout << opt_first[i] << " ";
    // }
    // std::cout << "\n";

    // for(int i = 0; i < N; i++) {
    //     opt_second[i] = accesses[0][i];
    // }
    // for(int i = 1; i < accesses.size(); i++) {
    //     for(int j = 0; j < N; j++) {
    //         opt_second[j] = accesses[i][(int)opt_second[j]];
    //     }
    // }

    // for(int i = 0; i < N; i++) {
    //     std::cout << opt_second[i] << " ";
    // }
    // std::cout << "\n";



    // #pragma approx declare tensor_functor(fn: [i] = ([i]))
    // #pragma approx declare tensor(ten: fn(data[0:N]))

    // auto data2 = data[access1[access2[50]]];

    // std::cout << data2 << "\n";
    #pragma approx declare tensor_functor(fn: [i] = ([[i]]))
    #pragma approx declare tensor(ten: fn(data[access1[access2[0:N]]]))

}
