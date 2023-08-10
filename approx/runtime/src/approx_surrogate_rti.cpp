// runtime interface for approx surrogate
#include "approx.h"

#include <iostream>
#include <cstring>
#include <vector>
#include <algorithm>


std::vector<int> get_transpose_vector(std::vector<int>& newShape, std::vector<int>& shape) {
	std::vector<int> extended;
	extended.reserve(std::max(shape.size(), newShape.size()));

	for(int i = 0; i < shape.size(); i++) {
		auto found = std::find(newShape.begin(), newShape.end(), shape[i]);
		if(found != newShape.end()) {
			extended.push_back(found - newShape.begin());
		} else {
			extended.push_back(i);
		}
	}

	return extended;
}

extern "C" {

void __approx_runtime_tensor_cleanup(void*) {
	std::cout << "Cleanup function is called\n";
}
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
}

std::ostream &operator<<(std::ostream &os, const tensor_shape_t &shape) {
        os << "(";
        for (int i = 0; i < shape.ndim; i++) {
                os << shape.shapes[i];
                if (i != shape.ndim - 1) {
                        os << ",";
                }
        }
        os << ")";
        return os;
}

extern "C" {

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

	void set_ndim(int ndim) {
		this->ndim = ndim;
		this->shapes->ndim = ndim;
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



void *__approx_runtime_convert_to_internal_representation(int nargsLHS, void *_slicesLHS, void *_shapesLHS, int nargsRHS, void *_argsRHS) {
	void **argsRHS_vpp = (void **)_argsRHS;
	array_info_t *argsRHS = (array_info_t *)argsRHS_vpp[0];

	slice_info_t *slicesLHS = (slice_info_t *)_slicesLHS;
	tensor_shape_t *shapesLHS = (tensor_shape_t *)_shapesLHS;

 	std::vector<int> LHSShape; 
	LHSShape.assign(shapesLHS->shapes, shapesLHS->shapes + shapesLHS->ndim);

	for(int RHSArg = 0; RHSArg < nargsRHS; RHSArg++) {
		array_info_t& argRHS = *(array_info_t *)argsRHS_vpp[RHSArg];
		std::vector<int> RHSShape;
		RHSShape.assign(argRHS.shapes->shapes, argRHS.shapes->shapes + argRHS.shapes->ndim);
		auto transpose_vec = get_transpose_vector(LHSShape, RHSShape);
		std::cout << "To transform shape: " << *argRHS.shapes << " to " << *shapesLHS << "\n";
		for(int i = 0; i < transpose_vec.size(); i++) {
			std::cout << transpose_vec[i] << " ";
		}
		std::cout << "\n";

	}

	return nullptr;
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

		tensor_info.set_ndim(slice_insert_pt);
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

	std::cout << "Outputting shapes in aivr substitution\n";
	for(int i = 0; i < ndim; i++) {
		auto &shape = *shapes;
		std::cout << shape[i] << " ";
	}
	std::cout << "\n";

}

}