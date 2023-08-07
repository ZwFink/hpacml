#include <iostream>
#include <cstring>

extern "C" {

typedef struct slice_info_t {
	uint32_t start;
	uint32_t stop;
	uint32_t step;
	uint32_t aivrechildkind;
	int32_t aivrerepr;
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

enum InternalReprType {
	Memory = 0,
	Torch = 1,
	TensorFlow
};

typedef struct internal_tensor_repr_data {
	int type;
	// we want an agnostic way to represent the shape
	tensor_shape_t shape;
	void *data;
} internal_repr_metadata_t;


// void __approx_runtime_shape_check(int numArgs, void *tensor, void* slice, tensor_shape *tensor_shape, tensor_shape *slice_shape) {

// }

// convert numArgs tensors into one tensor that has one extra dimension if numargs > 0
// actually performs any copying that may need to be done.
void __approx_runtime_convert_to_internal_representation(int numArgs, void *tensor, void *internal_repr_location) {
	// this function is currently not updated to reflect other changes.
	void **tensor_args = (void **)tensor;
	array_info_t **tensor_infos = (array_info_t **)tensor_args;
	internal_repr_metadata_t& internal_repr = *(internal_repr_metadata_t *)internal_repr_location;

 	int totalShape = 0;
	// we need to get the total shape
	for(int i = 0; i < numArgs; i++) {
		totalShape += tensor_infos[i]->shape().ndim;
	}

	// this is currently incorrect: it will return (6, N, N, N, N, N, N) instead of (6,N)
	// it should be fixed
	if(numArgs > 1)
		totalShape += 1;

	internal_repr.shape.shapes = (int *)malloc(sizeof(int) * totalShape);

    internal_repr.shape.ndim = totalShape;
    if (numArgs > 1) {
       internal_repr.shape.shapes[0] = numArgs;

       int totalTraversed = 0;
       for (int i = 0; i < numArgs; i++) {
			for(int j = 0; j < tensor_infos[i]->shape().ndim; j++) {
                internal_repr.shape.shapes[1+totalTraversed] =
                tensor_infos[i]->shape().shapes[j];
				++totalTraversed;
            }
	   }
        } else {
                internal_repr.shape.shapes[0] = totalShape;
        }

	for(int i = 0; i < totalShape; i++) {
		std::cout << "Shape " << i << " is " << internal_repr.shape.shapes[i] << "\n";
	}
        // here we would pass the tensor_infos and the internal_repr to the runtime to fill in
}

enum AIVREChildKind {
    STANDALONE,
    BINARY_EXPR,
    NONE
};

void __approx_runtime_convert_to_higher_order_shapes(int numArgs, void *ipt_memory_regns, void *tensors) {
	// given numargs shapes, convert each of then to higher order shapes individually.
	// For instance, if we have shape [6*N], we want to convert it to [N,6]. Note that tensors already has
	// the space allocated for the conversion.

	void **ipt_memory_rgns_vpp = (void**) ipt_memory_regns;
	void **tensor_args_vpp = (void **)tensors;

	uint maxShape = 0;
	for(int idx = 0; idx < numArgs; idx++) {
		array_info_t &tensor_info = *(array_info_t *)tensor_args_vpp[idx];
		maxShape = std::max(maxShape,  tensor_info.ndim);
	}

	// perhaps a bad idea
	int *shape_copy = (int*) alloca(sizeof(int)*maxShape);

	for(int idx = 0; idx < numArgs; idx++) {
		array_info_t &ipt_memory_info = *(array_info_t *)ipt_memory_rgns_vpp[idx];
		array_info_t &tensor_info = *(array_info_t *)tensor_args_vpp[idx];
		std::memcpy(shape_copy, &tensor_info.shape()[0], sizeof(int)*tensor_info.ndim);

		int numAIVRFound = 0;
		for(int i = 0; i < tensor_info.ndim; i++) {
			auto &t_slice = tensor_info.slices[i];
			auto AIVREKind = t_slice.aivrechildkind;
			if(AIVREKind != AIVREChildKind::NONE) {
				numAIVRFound++;
			}
		}

  		// we reserve the first numAIVRFound for the AIVR representations
		int slice_insert_pt = numAIVRFound;
		int AIVRInsertPoint = 0;

		for(int i = 0; i < tensor_info.ndim; i++) {
			auto &t_slice = tensor_info.slices[i];
			auto &ipt_slice = ipt_memory_info.slices[i];
			auto AIVREKind = t_slice.aivrechildkind;
			if(AIVREKind != AIVREChildKind::NONE) {
				// here, we have a decl like: [i*3:i*3+3]
				// whose shape the compiler has found is [3*N]
				// we need to turn this into [N,3]. To do this, we need to 
				// find 3 by dividing N*3/N
				int inner = shape_copy[i] / ipt_memory_info.shape()[i];
				tensor_info.shape()[AIVRInsertPoint] = t_slice.aivrerepr;
				tensor_info.shape()[slice_insert_pt] = inner;
				++AIVRInsertPoint;
				++slice_insert_pt;
			} else {
				tensor_info.shape()[slice_insert_pt] = shape_copy[i];
				++slice_insert_pt;
			}
		}

		tensor_info.ndim = slice_insert_pt;

	}

	// now go through and print all the shapes of the tensors
	for(int idx = 0; idx < numArgs; idx++) {
		array_info_t &tensor_info = *(array_info_t *)tensor_args_vpp[idx];
		std::cout << "Shape of tensor " << idx << " is ";
		for(int i = 0; i < tensor_info.ndim; i++) {
			std::cout << tensor_info.shape()[i] << " ";
		}
		std::cout << "\n";
	}
}

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

void __approx_runtime_substitute_aivr_in_shapes(int ndim, void *_slices, void *_shapes) {

	slice_info_t *slices = (slice_info_t *)_slices;
	tensor_shape_t *shapes = (tensor_shape_t *)_shapes;

	for(int i = 0; i < ndim; i++) {
		auto &slice = slices[i];
		// the slice is separate in each dimension, but there is one shpae
		auto &shape = *shapes;
		if(slice.aivrechildkind != AIVREChildKind::NONE) {
			shape[i] = slice.aivrerepr;
		}
	}

 std::cout << "Printing out a shape during substitution\n";
	for(int i = 0; i < ndim; i++) {
		std::cout << (*shapes)[i] << ", ";
	}
	std::cout << "\n";
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
	#pragma approx declare tensor_functor(fn: [j, i] = ([i, j, k]))
	#pragma approx declare tensor(t: fn(a[0:N,0:2*N:2, 0:N]))
}
