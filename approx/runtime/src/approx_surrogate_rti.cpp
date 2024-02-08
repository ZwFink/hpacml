// runtime interface for approx surrogate
#include "approx.h"

#include <iostream>
#include <cstring>
#include <vector>
#include <algorithm>
#include <limits>
#include "approx_internal.h"
#include "approx_surrogate.h"
#include "event.h"
#include <torch/serialize.h>
#include <algorithm>
#include "approx_tensor.h"

using Tensor = AbstractTensor<TorchTensorImpl>;
using AccessBounds = std::vector<std::pair<size_t,size_t>>;

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
	intptr_t *bases;
	int8_t *types;
	uint n_indirections;
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
get_strides(array_info_t &arg, std::vector<std::pair<size_t,size_t>> &bounds) {
	auto num_dims = arg.ndim;
	std::vector<int64_t> strides(num_dims, std::allocator<int64_t>());

	std::vector<int64_t> smallest_accesses(num_dims, std::numeric_limits<int64_t>::max());
	std::vector<int64_t> largest_accesses(num_dims, std::numeric_limits<int64_t>::min());

 
    strides[num_dims - 1] = arg.slices[num_dims - 1].step;

	auto dim_size = arg.slices[num_dims - 1].stop - arg.slices[num_dims - 1].start;

	// if the last dimension has size 1, then we are accessing row major because
	// C/C++ default to row major
	bool row_major_case_1 = (dim_size == 1);
	// Otherwise, we have to check that our step is at least the size of our dimension
	bool row_major_case_2 = !(arg.slices[num_dims - 1].step >= arg.slices[num_dims - 1].stop - arg.slices[num_dims - 1].start);

	bool is_row_major = row_major_case_1 || row_major_case_2;
	bool is_column_major = !is_row_major;

	if(is_row_major) {
      for(int i = num_dims - 2; i >= 0; i--) {
	  	int64_t cur_stride = strides[i+1];
	  	if(i +1 < arg.ndim_presubstitution) {
	  		cur_stride *= (bounds[i+1].second - bounds[i+1].first);
	  	} else {
	  		cur_stride *= arg.shape()[i+1];
	  	}

	  	strides[i] = cur_stride;
	  }
	} else if(is_column_major) {
		strides[0] = arg.slices[0].step;
		for(int i = 1; i < num_dims; i++) {
			int64_t cur_stride = strides[i-1];
			if(i < arg.ndim_presubstitution) {
				cur_stride *= arg.slices[i].step; //(bounds[i].second - bounds[i].first);
			} else {
				cur_stride = arg.slices[i].step;
			}

			strides[i] = cur_stride;
		}
	} 

	return strides;
}

Tensor::tensor_t memory_to_tensor(array_info_t *memory_descr, int base, AccessBounds &access_bounds, int target_ndim) {
	auto TypeOfTensorData = Tensor::getTensorDataTypeTypeFromApproxType((ApproxType) memory_descr->types[0]);
	auto OriginalDevice = Tensor::getDeviceForPointer((void*)memory_descr->bases[base]);
	
	auto SHP = Tensor::makeArrayRef(memory_descr->shapes->shapes, memory_descr->shapes->ndim);
	auto RHSShape = Tensor::makeArrayRef(memory_descr->shapes_aivrsubstituted->shapes, memory_descr->shapes_aivrsubstituted->ndim);
	std::vector<int> OriginalSteps;

    // FIXME: What do you do when we access a[b[0:N:N]]? Should a
	// be accessed also with this stride? We'll decide no for now, but
	// we can later consider how the user can specify this.
	for(int i = 0; i < memory_descr->ndim; i++) {
		auto &slice = memory_descr->slices[i];
	}
	if(base == 0 && memory_descr->n_indirections > 1) {
		OriginalSteps.reserve(memory_descr->ndim);
		for(int i = 0; i < memory_descr->ndim; i++) {
			OriginalSteps.push_back(memory_descr->slices[i].step);
			memory_descr->slices[i].step = 1;
		}
	}

	auto Strides = get_strides(*memory_descr, access_bounds);
	size_t base_offset = 0;
	for(int dim = 0; dim < memory_descr->ndim_presubstitution; dim++) {
		auto &slice = memory_descr->slices[dim];

		base_offset += Strides[dim] * slice.start;
	}

	auto ThisType = Tensor::getTensorDataTypeTypeFromApproxType((ApproxType) memory_descr->types[base]);
	auto options = Tensor::tensor_options_t().dtype(ThisType).device(OriginalDevice).requires_grad(false);

	auto elem_size = Tensor::getElementSizeForType(ThisType);
	intptr_t offset_base_ptr = (intptr_t) memory_descr->bases[base] + base_offset*elem_size;

	auto blob = Tensor::from_blob((void*) offset_base_ptr, SHP, Strides, options);
	while(blob.dim() < target_ndim) {
		blob.unsqueeze_(-1);
	}

	if(base == 0 && memory_descr->n_indirections > 1) {
		for(int i = 0; i < memory_descr->ndim; i++) {
			memory_descr->slices[i].step = OriginalSteps[i];
		}
	}
	return blob;
}

std::vector<std::pair<size_t,size_t>> get_access_bounds(array_info_t **args, int nargs) {
	std::vector<std::pair<size_t,size_t>> bounds;
	array_info_t &_arg = *args[0];
	bounds.reserve(_arg.ndim_presubstitution);

	for(int i = 0; i < _arg.ndim_presubstitution; i++) {
		bounds.emplace_back(std::make_pair(0,0));
		auto& bound = bounds[i];
		bound.first = std::numeric_limits<size_t>::max();
		bound.second = std::numeric_limits<size_t>::min();
	}

	for(int arg = 0; arg < nargs; arg++) {
		array_info_t &arr_arg = *args[arg];
		for(int dim = 0; dim < arr_arg.ndim_presubstitution; dim++) {
			auto &slice = arr_arg.slices[dim];
			auto &bound = bounds[dim];
			bound.first = std::min(bound.first, static_cast<size_t>(slice.start));
			bound.second = std::max(bound.second, static_cast<size_t>(slice.stop));
		}
	}
	return bounds;
}

template<typename TensorGen>
std::vector<WrappedTensor> manually_broadcast(TensorGen Gen, std::vector<WrappedTensor> tensors) {
    // Compute the output shape by aligning the shapes on the right and filling in with ones
    size_t max_len = 0;
    for (const auto& tensor : tensors) {
        max_len = std::max(max_len, tensor.sizes().size());
    }

    std::vector<int64_t> output_shape(max_len, 1);
    for (const auto& tensor : tensors) {
        auto tensor_shape = tensor.sizes();
        for (size_t i = 0; i < tensor_shape.size(); ++i) {
            size_t reversed_index = max_len - 1 - i;
            size_t tensor_reversed_index = tensor_shape.size() - 1 - i;
            output_shape[reversed_index] = std::max(output_shape[reversed_index], tensor_shape[tensor_reversed_index]);
        }
    }


    std::vector<WrappedTensor> broadcasted_tensors;
    for (auto& tensor : tensors) {
        // Compute the broadcasted strides for each tensor
        auto tensor_shape = tensor.sizes();
        std::vector<int64_t> expanded_shape(max_len, 1);
        std::copy(tensor_shape.begin(), tensor_shape.end(), expanded_shape.end() - tensor_shape.size());
        std::vector<int64_t> tensor_strides = tensor.strides().vec();
        tensor_strides.resize(max_len, 0);  // Extend strides with zeros
        std::vector<int64_t> broadcasted_strides(max_len);

        for (size_t i = 0; i < max_len-1; ++i) {
            broadcasted_strides[i] = (expanded_shape[i] != 1) ? tensor_strides[i] : 0;
        }

		// Note: We'll use the original shape/stride for the final dimension: this is because
		// we are going to concatenate the tensors along this dimension, so we don't need them to
		// actually match.
		broadcasted_strides[max_len-1] = tensor.strides()[tensor.dim()-1];
		output_shape[max_len-1] = tensor_shape[tensor_shape.size()-1];

        // Create a view with the new shape and strides
        broadcasted_tensors.push_back(Gen(torch::as_strided(tensor.get_tensors()[0], output_shape, broadcasted_strides)));
    }

    return broadcasted_tensors;
}

template <typename TensorInstanceGen>
std::vector<WrappedTensor> wrap_memory_in_tensors(
    TensorInstanceGen Gen, internal_repr_metadata_t &metadata, int nargsLHS,
    void *_slicesLHS, void *_shapesLHS, int nargsRHS, void *_argsRHS) {
  void **argsRHS_vpp = (void **)_argsRHS;
  array_info_t *argsRHS = (array_info_t *)argsRHS_vpp[0];

  slice_info_t *slicesLHS = (slice_info_t *)_slicesLHS;
  tensor_shape_t *shapesLHS = (tensor_shape_t *)_shapesLHS;

  auto LHSShape = Tensor::makeArrayRef(shapesLHS->shapes, shapesLHS->ndim);

  std::vector<WrappedTensor> RHSTensors;
  auto TypeOfTensorData = Tensor::getTensorDataTypeTypeFromApproxType(
      (ApproxType)argsRHS->types[0]);
  dbgs() << "Tensor data has type " << TypeOfTensorData << "\n";
  auto AccessBounds = get_access_bounds((array_info_t **)argsRHS_vpp, nargsRHS);

  TensorType::Device OriginalDevice{TensorType::CPU};

  for (int RHSArg = 0; RHSArg < nargsRHS; RHSArg++) {
  	auto ThisTens = Gen();
    array_info_t *thisArg = (array_info_t *)argsRHS_vpp[RHSArg];
    for (int indirection = 0; indirection < thisArg->n_indirections;
         indirection++) {
      auto ThisTensInst =
          memory_to_tensor(thisArg, indirection, AccessBounds, LHSShape.size());
      ThisTens.add_tensor(ThisTensInst);
    }

    ThisTens.to(Tensor::CUDA, /*nonblocking=*/true);
    RHSTensors.push_back(ThisTens);
  }

  metadata.set_device(OriginalDevice);
  metadata.set_underlying_type((ApproxType)argsRHS->types[0]);

  return RHSTensors;
}

extern "C" {

void __approx_runtime_tensor_cleanup(void* data) {
	dbgs() << "Cleanup function is called\n";
	internal_repr_metadata_t *metadata = (internal_repr_metadata_t *)data;
	delete metadata;
}

void *__approx_runtime_convert_internal_tensor_to_mem(int nargsLHS,
                                                      void *_slicesLHS,
                                                      void *_shapesLHS,
                                                      int nargsRHS,
                                                      void *_argsRHS) {
  EventRecorder::GPUEvent TransferEvent =
      EventRecorder::CreateGPUEvent("Wrap output Memory");
  TransferEvent.recordStart();

  internal_repr_metadata_t *metadata = new internal_repr_metadata_t();
  TensorWrapperTensorToMem<Tensor::tensor_t> Gen;
  auto RHSTensors = wrap_memory_in_tensors(Gen, *metadata, nargsLHS, _slicesLHS,
                                           _shapesLHS, nargsRHS, _argsRHS);

  for(auto &Tens : RHSTensors) {
	  Tens.perform_indirection();
  }

  auto LibraryType = Tensor::getTensorLibraryType();
  metadata->set_library_type(LibraryType);
  metadata->set_direction(Direction::TENSOR_TO_MEM);

  for (auto &Tens : RHSTensors) {
    metadata->add_tensor(Tens);
	// dbgs() << "Wrapped memory is: " << Tens.sizes() << "\n";
  }
  TransferEvent.recordEnd();
  EventRecorder::LogEvent(TransferEvent);

  return metadata;
}

void *__approx_runtime_convert_internal_mem_to_tensor(int nargsLHS, void *_slicesLHS, void *_shapesLHS, int nargsRHS, void *_argsRHS) {
	EventRecorder::GPUEvent TransferEvent = EventRecorder::CreateGPUEvent("To Tensor");
	TransferEvent.recordStart();

	internal_repr_metadata_t *metadata = new internal_repr_metadata_t();
	TensorWrapperMemToTensor<Tensor::tensor_t> Gen;
    auto RHSTensors = wrap_memory_in_tensors(Gen, *metadata, nargsLHS, _slicesLHS, _shapesLHS, nargsRHS, _argsRHS);
	for(auto &Tens : RHSTensors) {
		Tens.perform_indirection();
	}

    WrappedTensor LHSTensor = Gen();
    if (nargsRHS == 1) {
        LHSTensor.add_tensor(RHSTensors[0]);
    } else {
      RHSTensors = manually_broadcast(Gen, RHSTensors);
      LHSTensor.add_tensor(
          torch::cat(TensorWrapper<Tensor::tensor_t>::concat(RHSTensors), -1));
    }

    dbgs() << "Final tensor is: " << LHSTensor.sizes() << "\n";

	auto LibraryType = Tensor::getTensorLibraryType();
	metadata->set_library_type(LibraryType);
	metadata->set_direction(Direction::MEM_TO_TENSOR);
	metadata->add_tensor(LHSTensor);

	TransferEvent.recordEnd();
	EventRecorder::LogEvent(TransferEvent);
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

				tensor_info.aivrshape()[AIVRInsertPoint] = t_slice.aivrerepr;

				if(inner != 1) {
					tensor_info.shape()[slice_insert_pt] = inner;
					tensor_info.aivrshape()[slice_insert_pt] = inner;
					tensor_info.slices[slice_insert_pt].start = t_slice.start;
					tensor_info.slices[slice_insert_pt].step = tensor_info.slices[i].step;
					tensor_info.slices[slice_insert_pt].stop = inner;

					tensor_info.slices[AIVRInsertPoint].step = 1;
					tensor_info.slices[AIVRInsertPoint].start = t_slice.start;
					tensor_info.slices[AIVRInsertPoint].stop = t_slice.stop;
					++slice_insert_pt;
				}
				++AIVRInsertPoint;
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

        // here we assign the pointers instead of values because
		// there will not be any bases allocated for the functor
		// so copying the values would be incorrect
        finfo.bases = tinfo.bases;
		finfo.types = tinfo.types;
		finfo.n_indirections = tinfo.n_indirections;

        for(int i = 0; i < tinfo.ndim; i++) {
	      auto &t_slice = tinfo.slices[i];
	      auto &f_slice = finfo.slices[i];
		  size_t base = 0;

		  base = f_slice.start - t_slice.start;

		  f_slice.start = t_slice.start;
		  f_slice.stop = t_slice.stop;

		  f_slice.start += base;
		  f_slice.stop += base;
		//   f_slice.step = t_slice.step;

    	//   if(f_slice.step != 1) {
			// std::cerr << "Found step " << f_slice.step << "\n";
    	  	// std::cerr << "Step is not 1, this is not supported yet\n";
    	//   }

    	  finfo.shape()[i] *= tinfo.shape()[i];
	    }
	}

	#ifdef DEBUG
	for(int idx = 0; idx < numArgs; idx++) {
	    array_info_t *tensor_info = (array_info_t *)slice_args[idx];
	    array_info_t &tinfo = *tensor_info;
	    slice_info_t &sinfo = *tinfo.slices;

	    dbgs() << "Tensor has this many shapes: " << tinfo.ndim << "\n";

	    for(int i = 0; i < tinfo.ndim; i++) {
	    	dbgs() << "Slice " << i << " has shape: " 
	    	<< tinfo.slices[i].start << ", " << tinfo.slices[i].stop << ", " << tinfo.slices[i].step << "\n";
	    }

		dbgs() << "Shape: (";
	    for (int i = 0; i < tinfo.ndim; i++) {
	    	dbgs() << tinfo.shape()[i] << ", ";
	    }
		dbgs() << ")\n";	
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