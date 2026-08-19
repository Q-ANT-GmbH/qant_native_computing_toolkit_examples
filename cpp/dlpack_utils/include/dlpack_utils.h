#ifndef DLPACK_UTILS_H
#define DLPACK_UTILS_H

#include <dlpack/dlpack.h>
#include <stdfloat> // for bfloat16_t  (C++23)
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>

void delete_mtensor(struct DLManagedTensorVersioned *self)
{
    delete[] self->dl_tensor.shape;
    // the rest should be cleaned up when the corresponding data managers
    // go out of scope
}

// creates a managed DL tensor as a non-owning view on the data of the xarray.
// Only the shape array is heap-allocated; the data stays owned by the xarray.
// Invoke the deleter once the view is no longer needed to free the shape array.
DLManagedTensorVersioned xarray_to_mtensor_view(xt::xarray<std::bfloat16_t> const &xarr)
{
    DLTensor tensor = DLTensor();

    int32_t ndim = xarr.dimension();
    auto shape = xarr.shape();
    tensor.ndim = ndim;
    tensor.shape = new int64_t[ndim];
    for (int i = 0; i < ndim; ++i)
    {
        tensor.shape[i] = shape[i];
    };
    tensor.strides = NULL;

    tensor.dtype.code = kDLBfloat;
    tensor.dtype.bits = 16;
    tensor.dtype.lanes = 1;

    tensor.data = const_cast<void *>(static_cast<const void *>(xarr.data()));
    tensor.byte_offset = 0;
    tensor.device.device_type = kDLCPU;
    tensor.device.device_id = 0;

    DLManagedTensorVersioned mtensor = DLManagedTensorVersioned();
    mtensor.version = DLPackVersion{major : DLPACK_MAJOR_VERSION, minor : DLPACK_MINOR_VERSION};
    mtensor.flags = 0;
    mtensor.dl_tensor = tensor;
    mtensor.manager_ctx = const_cast<void *>(static_cast<const void *>(&xarr));
    mtensor.deleter = delete_mtensor;

    return mtensor;
}

// creates a zero-copy view on the DLTensor data.
// The view is only valid while the managed tensor is alive. The caller
// remains responsible for invoking mtensor->deleter once the data is no
// longer needed.
auto mtensor_to_tensor_view(DLManagedTensorVersioned *mtensor)
{
    DLTensor &tensor = mtensor->dl_tensor;
    std::vector<size_t> shape(tensor.shape, tensor.shape + tensor.ndim);
    size_t len_data = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    return xt::adapt(static_cast<std::bfloat16_t *>(tensor.data), len_data, xt::no_ownership(), shape);
}

// copies the DLTensor data into a new xtensor array.
// If release is true, the managed tensor is released afterwards by calling
// its deleter, as required by the DLPack protocol. Pass false if the
// managed tensor is still needed; the caller is then responsible for
// invoking the deleter once it is no longer used.
xt::xarray<std::bfloat16_t> mtensor_to_tensor(DLManagedTensorVersioned *mtensor, bool const release)
{
    xt::xarray<std::bfloat16_t> result = mtensor_to_tensor_view(mtensor);
    if (release && mtensor->deleter != nullptr)
    {
        mtensor->deleter(mtensor);
    }
    return result;
}

#endif
