// runtime interface for approx surrogate
#include "approx.h"

#include <iostream>
#include <cstring>
#include <vector>
#include <algorithm>
#include "approx_surrogate.h"

using Tensor = AbstractTensor<TorchTensorImpl>;

#ifdef DEBUG
#define dbgs() std::cout
#else
#define dbgs() if(0) std::cout
#endif

std::vector<int64_t> get_transpose_vector(Tensor::ArrayRef<int64_t> newShape, Tensor::ArrayRef<int64_t> shape) {
	std::vector<int64_t> extended;
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

typedef struct slice_info_t {
	int64_t start;
	int64_t stop;
	int64_t step;
	int aivrechildkind;
	int64_t aivrerepr;
} slice_info_t;

// a tensor's shape is a list of integers,
// each representing the size of a dimension
// currently, this is distinct form array_info shapes
// which is just a list of integers
// but can be changed later to combine the two
typedef struct tensor_shape {
	int ndim;
	int64_t *shapes;

	void operator=(const tensor_shape &other) {
		ndim = other.ndim;
		for(int i = 0; i < ndim; i++) {
			shapes[i] = other.shapes[i];
		}
	}

	int64_t& operator[](int idx) {
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

std::vector<int64_t> get_shape_from_slices(int nargs, slice_info_t *slices) {
	std::vector<int64_t> shape;
	for(int i = 0; i < nargs; i++) {
		int shp = (slices[i].stop - slices[i].start) / slices[i].step;
		if(shp % slices[i].step != 0) {
			shp++;
		}
		shape.push_back(shp);
	}
	return shape;
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
	tensor_shape_t *shapes_aivrsubstituted;
	uint ndim_presubstitution;

	tensor_shape_t &shape() {
		return *shapes;
	}

	tensor_shape_t &aivrshape() {
		return *shapes_aivrsubstituted;
	}

	void set_ndim(int ndim) {
		this->ndim_presubstitution = this->ndim;
		this->ndim = ndim;
		this->shapes->ndim = ndim;
		this->shapes_aivrsubstituted->ndim = ndim;
	}
} array_info_t;

}

std::vector<int64_t>
get_strides(array_info_t &arg) {
	auto num_dims = arg.ndim;
	std::vector<int64_t> strides(num_dims);

    if (num_dims - 1 >= arg.ndim_presubstitution-1) {
            strides[num_dims - 1] = 1;
    } else {
            strides[num_dims - 1] = arg.slices[num_dims - 1].step;
    }

    for(int i = num_dims - 2; i >= 0; i--) {
		int64_t cur_stride = strides[i+1];
		if(i +1 < arg.ndim_presubstitution) {
			cur_stride *= (arg.slices[i+1].stop - arg.slices[i+1].start);
		} else {
			cur_stride *= arg.shape()[i+1];
		}
			if(i < arg.ndim_presubstitution) {
				cur_stride *= arg.slices[i].step;
			}

		strides[i] = cur_stride;
	}

	return strides;
}

extern "C" {

enum InternalReprType {
	Memory = 0,
	Torch = 1,
	TensorFlow
};

typedef struct internal_tensor_repr_data {
	int type;
	void *data;

	~internal_tensor_repr_data() {
		Tensor *T = (Tensor *)data;
		delete T;
	}

	void set_library_type(int t) {
		type = t;
	}

	void set_data(void *d) {
		data = d;
	}

} internal_repr_metadata_t;


void __approx_runtime_tensor_cleanup(void* data) {
	dbgs() << "Cleanup function is called\n";
	internal_repr_metadata_t *metadata = (internal_repr_metadata_t *)data;
	delete metadata;
}

void *__approx_runtime_convert_to_internal_representation(int nargsLHS, void *_slicesLHS, void *_shapesLHS, int nargsRHS, void *_argsRHS) {
	void **argsRHS_vpp = (void **)_argsRHS;
	array_info_t *argsRHS = (array_info_t *)argsRHS_vpp[0];

	slice_info_t *slicesLHS = (slice_info_t *)_slicesLHS;
	tensor_shape_t *shapesLHS = (tensor_shape_t *)_shapesLHS;

	auto LHSShape = Tensor::makeArrayRef(shapesLHS->shapes, shapesLHS->ndim);

	for(int RHSArg = 0; RHSArg < nargsRHS; RHSArg++) {
		array_info_t& argRHS = *(array_info_t *)argsRHS_vpp[RHSArg];
		auto RHSShape = Tensor::makeArrayRef(argRHS.shapes_aivrsubstituted->shapes, argRHS.shapes_aivrsubstituted->ndim);
		auto transpose_vec = get_transpose_vector(LHSShape, RHSShape);
		#ifdef DEBUG
		dbgs() << "To transform shape: " << *argRHS.shapes_aivrsubstituted << " to " << *shapesLHS << "\n";
		for(int i = 0; i < transpose_vec.size(); i++) {
			std::cout << transpose_vec[i] << " ";
		}
		std::cout << "\n";
		#endif
	}

	std::vector<Tensor::tensor_t> RHSTensors;
	auto TypeOfTensorData = Tensor::getTensorDataTypeTypeFromApproxType((ApproxType) argsRHS->type);
	dbgs() << "Tensor data has type " << TypeOfTensorData << "\n";
	for(int RHSArg = 0; RHSArg < nargsRHS; RHSArg++) {
		array_info_t& argRHS = *(array_info_t *)argsRHS_vpp[RHSArg];
		auto SHP = Tensor::makeArrayRef(argRHS.shapes->shapes, argRHS.shapes->ndim);
		auto RHSShape = Tensor::makeArrayRef(argRHS.shapes_aivrsubstituted->shapes, argRHS.shapes_aivrsubstituted->ndim);
		auto Strides = get_strides(argRHS);

		#ifdef DEBUG
		dbgs() << "Strides are: ";
		for(auto s: Strides) {
			dbgs() << s << ", ";
		}
		dbgs() << "\n";
		#endif

		Tensor::tensor_t blob = Tensor::from_blob(argRHS.base, SHP, Strides, TypeOfTensorData);

		if(nargsRHS > 1) {
                        auto transpose_vec_ =
                            get_transpose_vector(LHSShape, RHSShape);
                        blob = Tensor::transpose(
                            blob, Tensor::makeArrayRef(transpose_vec_.data(),
                                                       transpose_vec_.size()));
                }

		RHSTensors.push_back(blob);
	}

    Tensor::tensor_t *LHSTensor = new Tensor::tensor_t();
    if (nargsRHS == 1) {
            *LHSTensor = RHSTensors[0];
    } else {
            *LHSTensor = Tensor::cat(RHSTensors, -1);
    }
    dbgs() << "Final tensor is: " << LHSTensor->sizes() << "\n";;

	auto LibraryType = Tensor::getTensorLibraryType();
	internal_repr_metadata_t *metadata = new internal_repr_metadata_t();
	metadata->set_library_type(LibraryType);
	metadata->set_data(LHSTensor);

	return metadata;
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
	int64_t *shape_copy = (int64_t*) alloca(sizeof(int64_t)*maxShape);

	for(int idx = 0; idx < numArgs; idx++) {
		array_info_t &ipt_memory_info = *(array_info_t *)ipt_memory_rgns_vpp[idx];
		array_info_t &tensor_info = *(array_info_t *)tensor_args_vpp[idx];
		std::memcpy(shape_copy, &tensor_info.shape()[0], sizeof(int64_t)*tensor_info.ndim);

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
				int64_t inner = shape_copy[i] / ipt_memory_info.shape()[i];
				tensor_info.shape()[AIVRInsertPoint] = ipt_memory_info.shape()[i];
				tensor_info.shape()[slice_insert_pt] = inner;

				tensor_info.aivrshape()[AIVRInsertPoint] = t_slice.aivrerepr;
				tensor_info.aivrshape()[slice_insert_pt] = inner;

				++AIVRInsertPoint;
				++slice_insert_pt;
			} else {
				tensor_info.shape()[slice_insert_pt] = shape_copy[i];
				tensor_info.aivrshape()[slice_insert_pt] = shape_copy[i];
				++slice_insert_pt;
			}
		}

		tensor_info.set_ndim(slice_insert_pt);
	}

	#ifdef DEBUG
	// now go through and print all the shapes of the tensors
	for(int idx = 0; idx < numArgs; idx++) {
		array_info_t &tensor_info = *(array_info_t *)tensor_args_vpp[idx];
		std::cout << "Shape of tensor " << idx << " is ";
		for(int i = 0; i < tensor_info.ndim; i++) {
			std::cout << tensor_info.shape()[i] << " ";
		}
		std::cout << "\n";
	}
	#endif
}

void __approx_runtime_slice_conversion(int numArgs, void *tensor, void *slice) {
    dbgs() << "Found " << numArgs << " arguments\n";
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
	    		std::cerr << "Step is not 1, this is not supported yet\n";
	    	}
	    	f_slice.step = t_slice.step;

	    	finfo.shape()[i] *= tinfo.shape()[i];

	    }
	}

	#ifdef DEBUG
	for(int idx = 0; idx < numArgs; idx++) {
	    array_info_t *tensor_info = (array_info_t *)slice_args[idx];
	    array_info_t &tinfo = *tensor_info;
	    slice_info_t &sinfo = *tinfo.slices;

	    dbgs() << "Tensor has this many shapes: " << tinfo.ndim << "\n";
	    dbgs() << "Tensor vase pointer is: " << tinfo.base << "\n";

	    for(int i = 0; i < tinfo.ndim; i++) {
	    	dbgs() << "Slice " << i << " has shape: " 
	    	<< tinfo.slices[i].start << ", " << tinfo.slices[i].stop << ", " << tinfo.slices[i].step << "\n";
	    }
	    for (int i = 0; i < tinfo.ndim; i++) {
	    	dbgs() << "Tensor shape: " << tinfo.shape()[i] << "\n";
	    }
	}
	#endif

	
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

	#ifdef DEBUG
	dbgs() << "Outputting shapes in aivr substitution\n";
	for(int i = 0; i < ndim; i++) {
		auto &shape = *shapes;
		dbgs() << shape[i] << " ";
	}
	dbgs() << "\n";
	#endif

}

}