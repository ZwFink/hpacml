#include <iostream>
#include <cstring>
#include <vector>
#include "approx.h"

int main()
{
	int N = 200;
	int *a = new int[10000];
	for(int i = 0; i < 10000; i++) {
		a[i] = i;
	}
	std::cout << "A is: " << a << "\n";
	int b = 0;
	char c = 0;
	b = a[0] + 2;

	#pragma approx declare tensor_functor(bs_ipt_1tensor: [i,0:6] = ([i*6:i*6+6]))
	#pragma approx declare tensor(bs_ipt_1t: bs_ipt_1tensor(a[0:N]))

	#pragma approx declare tensor_functor(blackscholes_ipt: [i,0:6] = ([i], [i], [i], [i], [i], [i]))
	#pragma approx declare tensor(bs_ipt: blackscholes_ipt(a[0:N], a[0:N], a[0:N], a[0:N], a[0:N], a[0:N]))

	#pragma approx declare tensor_functor(fn: [j, i] = ([i, j, k]))
	#pragma approx declare tensor(t: fn(a[0:N,0:2*N:2, 0:N]))
	{

	#pragma approx declare tensor_functor(fn25: [i, 0:5] = ([i*3:i*3+3], [i], [i]))
	#pragma approx declare tensor(tn38: fn25(a[0:N], a[0:N], a[0:N]))
	}

	#pragma approx declare tensor_functor(fn26: [i,j, 0:5] = ([i, j*3:j*3+3], [i,j], [j,i]))
	#pragma approx declare tensor(tn39: fn26(a[0:N, 0:N], a[0:N, 0:N], a[0:N, 0:N]))
}
