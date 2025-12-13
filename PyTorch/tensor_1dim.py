# Introduction to 1-D tensors in PyTorch 
import numpy as np 
import torch

#1. From array or matrix 
data = [
    [1, 2, 3, 4], 
    [5, 6, 7, 8],
    [9, 10, 11, 12], 
    [13, 14, 15, 16],
]

x_data = torch.tensor(data)
print(x_data)

#2. Convert from numpy array 
x = [
    [1, 2, 3],
    [4, 5, 6],
]
array = np.array(x)

x_array = torch.from_numpy(array)
print('Tensor from Array: ', x_array)

#3. Convert and change tensor with another tensor object
# Create tensor's with all values as one with size of x_array
tensor_x = torch.ones_like(input= x_array, dtype= torch.float64)
print('Tensor with One`s value: ', tensor_x)

tensor_x2 = torch.rand_like(input= x_array, dtype= torch.float32)
print('Tensor with random float32 values: ', tensor_x2)

#4. Tensors from tuple values (tuple value = (x_dims, y_dims, z_dims))
tuple_data = (2, 3)
x1 = torch.rand(tuple_data)
x2 = torch.ones(tuple_data)
x3 = torch.zeros(tuple_data)

print(f'Tensor with random values \n: {x1} \n')
print(f'Tensor with random values \n: {x2} \n')
print(f'Tensor with random values \n: {x3} \n')

print(type(x3))

#5. Tensor attributes (shape, datatype, device)
print(f'Shape of the tensor x1 is: {x1.shape} \n')
print(f'Datatype of the tensor x1 is: {x1.dtype} \n')
print(f'Device on which tensor x1 is running: {x1.device} \n')

