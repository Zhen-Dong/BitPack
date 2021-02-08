import torch
import numpy as np


def find_optimal_compress_dim(tensor_shape, bit):
    """
    A function to find the index of optimal dimemsion to apply packing on.

    """
    index = 0
    curr_index = 0
    max_dim = 0
    for dim in tensor_shape:
        if (dim * bit) % 8 == 0:
            index = curr_index
            break
        if dim > max_dim:
            index = curr_index
            max_dim = dim
        curr_index += 1
    return index


def pack_tensor(tensor, bit):
    """
    A function to pack tensor into a packed_tensor with only one dimension difference.
    binary_tensor is the binary counterpart of tensor.
    padded_binary_tensor is a vector reshaped from binary_tensor, it is padded with zeros
    to have the same amount of values as the target binary_packed_tensor.
    binary_packed_tensor is the binary counterpart of packed_tensor.
    packed_tensor is the output tensor.

    """
    tmp_tensor_shape = list(tensor.shape)
    tmp_tensor_shape.append(1)
    tmp_tensor = np.zeros(tuple(tmp_tensor_shape), dtype=np.uint8)
    tmp_tensor[..., 0] = tensor

    binary_tensor = np.unpackbits(tmp_tensor, axis=-1)
    binary_tensor_shape = tmp_tensor_shape
    binary_tensor_shape[-1] = 8

    i = find_optimal_compress_dim(tensor.shape, bit)
    packed_dim = int(tensor.shape[i] * bit / 8) + (tensor.shape[i] * bit % 8 > 0)
    binary_packed_tensor_shape = binary_tensor_shape
    binary_packed_tensor_shape[i] = packed_dim

    packed_tensor = np.zeros(tuple(binary_packed_tensor_shape), dtype=np.uint8)
    padded_binary_tensor = binary_tensor[..., 8-bit: 8].reshape((-1))
    padding_width = packed_tensor.size - padded_binary_tensor.size
    padded_binary_tensor = np.pad(padded_binary_tensor, (0, padding_width), 'constant', constant_values=(0, 0))
    binary_packed_tensor = padded_binary_tensor.reshape(tuple(binary_packed_tensor_shape))

    packed_tensor = np.packbits(binary_packed_tensor, axis=-1)
    packed_tensor = packed_tensor[..., 0]

    return packed_tensor


def unpack_tensor(packed_tensor, bit, target_shape):
    """
    A function to unpack a packed_tensor into the original tensor with target_shape.
    binary_packed_tensor is the binary counterpart of packed_tensor.
    unpadded_binary_packed_tensor is a vector reshaped from binary_packed_tensor,
    it is unpadded to have the same amount of values as binary_unpacked_tensor.
    binary_unpacked_tensor is the binary counterpart of unpacked_tensor.
    unpacked_tensor is the output tensor, it should be the same as the original tensor.

    """
    tmp_packed_tensor_shape = list(packed_tensor.shape)
    tmp_packed_tensor_shape.append(1)
    tmp_packed_tensor = np.zeros(tuple(tmp_packed_tensor_shape), dtype=np.uint8)
    tmp_packed_tensor[..., 0] = packed_tensor

    binary_packed_tensor = np.unpackbits(tmp_packed_tensor, axis=-1).reshape((-1))
    unpading_mask = np.arange(binary_packed_tensor.size - np.prod(target_shape) * bit) + np.prod(target_shape) * bit
    unpadded_binary_packed_tensor = np.delete(binary_packed_tensor, unpading_mask)

    binary_unpacked_tensor_shape = list(target_shape)
    binary_unpacked_tensor_shape.append(bit)
    unpadded_binary_packed_tensor = unpadded_binary_packed_tensor.reshape(tuple(binary_unpacked_tensor_shape))
    binary_unpacked_tensor_shape[-1] = 8

    binary_unpacked_tensor = np.zeros(tuple(binary_unpacked_tensor_shape), dtype=np.uint8)
    binary_unpacked_tensor[..., 8-bit: 8] = unpadded_binary_packed_tensor[..., :]

    unpacked_tensor = np.packbits(binary_unpacked_tensor, axis=-1).reshape(target_shape)

    return unpacked_tensor


def test():
    max_value = 64
    bit = 6
    assert max_value <= 2 ** bit

    tensor = np.random.randint(max_value, size=(68, 68, 7, 7), dtype=np.uint8)
    print(tensor[0, 0, :, :])
    print(tensor.shape)

    packed_tensor = pack_tensor(tensor, bit)
    print(packed_tensor[0, 0, :, :])
    print(packed_tensor.shape)

    unpacked_tensor = unpack_tensor(packed_tensor, bit, (68, 68, 7, 7))
    print(unpacked_tensor[0, 0, :, :])
    print(unpacked_tensor.shape)

    if np.array_equal(tensor, unpacked_tensor):
        print("Correct Compression")
        print("Compression Ratio: ", tensor.size / packed_tensor.size)
    else:
        print("Wrong Compression")


if __name__ == '__main__':
    test()
