#include <iostream>


extern "C" {

typedef struct slice_info_t {
	uint32_t start;
	uint32_t stop;
	uint32_t step;
} slice_info_t;

// a tensor's shape is a list of integers,
// each representing the size of a dimension
// currently, this is distinct form array_info shapes
// which is just a list of integers
// but can be changed later to combine the two
typedef struct tensor_shape {
	int ndim;
	int *shapes;

	void operator=(const tensor_shape &other) {
		ndim = other.ndim;
		for(int i = 0; i < ndim; i++) {
			shapes[i] = other.shapes[i];
		}
	}

	int& operator[](int idx) {
		return shapes[idx];
	}
} tensor_shape_t;

typedef struct array_info {
	void *base;
	int8_t type;
	uint ndim;
	slice_info_t *slices;
	// yes, we end up duplicating ndim, but that's fine.
	// the cost is low and we don't have to re-allocate the shapes for the
	// RHS when doing our analysis
	tensor_shape_t *shapes;

	tensor_shape_t &shape() {
		return *shapes;
	}
} array_info_t;


// void __approx_runtime_shape_check(int numArgs, void *tensor, void* slice, tensor_shape *tensor_shape, tensor_shape *slice_shape) {

// }

void __approx_runtime_slice_conversion(int numArgs, void *tensor, void *slice) {
    std::cout << "Found " << numArgs << " arguments\n";
	void **tensor_args = (void **)tensor;
	void **slice_args = (void **)slice;
	for(int idx = 0; idx < numArgs; idx++) {
	array_info_t *tensor_info = (array_info_t *)tensor_args[idx];
	array_info_t &tinfo = *tensor_info;

	array_info_t *functor_info = (array_info_t *)slice_args[idx];
	array_info_t &finfo = *functor_info;

	for(int i = 0; i < tinfo.ndim; i++) {
		auto &t_slice = tinfo.slices[i];
		auto &f_slice = finfo.slices[i];
		auto *t_ptr = tinfo.base;
		auto t_type = tinfo.type;

		finfo.base = t_ptr;
		finfo.type = t_type;

		f_slice.start *= t_slice.start;
		f_slice.stop *= t_slice.stop;
		if(f_slice.step != 1) {
			std::cout << "Step is not 1, this is not supported yet\n";
		}
		f_slice.step = t_slice.step;

		finfo.shape()[i] *= tinfo.shape()[i];

	}
	}

	for(int idx = 0; idx < numArgs; idx++) {
	array_info_t *tensor_info = (array_info_t *)slice_args[idx];
	array_info_t &tinfo = *tensor_info;
	slice_info_t &sinfo = *tinfo.slices;

	std::cout << "Tensor has this many shapes: " << tinfo.ndim << "\n";
	std::cout << "Tensor vase pointer is: " << tinfo.base << "\n";

	for(int i = 0; i < tinfo.ndim; i++) {
		std::cout << "Slice " << i << " has shape: " 
		<< tinfo.slices[i].start << ", " << tinfo.slices[i].stop << ", " << tinfo.slices[i].step << "\n";
	}
	for (int i = 0; i < tinfo.ndim; i++) {
		std::cout << "Tensor shape: " << tinfo.shape()[i] << "\n";
	}
	}

	
}
}
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
	//#pragma approx ml(infer) in(a[0:10]) out(b)
	b = a[0] + 2;
	//#pragma approx declare tensor_functor(functor_name: [0:10] = ([::5], [0:10+c], [0:c*10:1]))
  	// #pragma approx declare tensor_functor(functor_name: [i, 0:6] = ([i], [i], [i], [i], [i], [i]))
  	// #pragma approx declare tensor_functor(fivepoint_stencil: [i, j, 0:5] = ([i,j-1:j+2], [i-1,j], [i+1,5]))
  	// #pragma approx declare tensor(first_tensor: functor_name(a[0:N], a[0:N], a[0:N], a[0:N], a[0:N], a[0:N]))
  	// #pragma approx declare tensor(second_tensor: functor_name(a[0:N], a[0:N], a[0:N], a[0:N], a[0:N], a[0:N]))

  	// #pragma approx declare tensor_functor(functor_name2: [0:c+10] = ([0:c,0:c+5:10]))
  	// #pragma approx declare tensor_functor(functor_name3: [c:c+10:10] = ([0:2:2]))
	{
	// #pragma approx declare tensor_functor(functor_name4: [c:j:10] = ([0:2:2]))
  	// #pragma approx declare tensor(third_tensor: functor_name4(a[0:N], a[0:N], a[0:N], a[0:N], a[0:N], a[0:N]))
	}
	// #pragma approx declare tensor_functor(functor_name4: [c:j:10] = ([0:2:2]))

	#pragma approx declare tensor_functor(bs_ipt_1tensor: [i,0:6] = ([i*6:i*6+6]))
	#pragma approx declare tensor(bs_ipt_1t: bs_ipt_1tensor(a[0:N]))
	#pragma approx declare tensor_functor(blackscholes_ipt: [i,0:6] = ([i], [i], [i], [i], [i], [i]))
	#pragma approx declare tensor(bs_ipt: blackscholes_ipt(a[0:N], a[0:N], a[0:N], a[0:N], a[0:N], a[0:N]))
	#pragma approx declare tensor_functor(fn: [i, j] = ([i, j, k]))
	#pragma approx declare tensor(t: fn(a[0:N,0:2*N:2, 0:N]))
}
