# Important operations to be use in Tensors in PyTorch 
import numpy as np 
import torch

#1. Tensor Slicing
x = [[1, 2, 3, 4], [5, 6, 7, 9], [10, 11, 12, 13], [14, 15, 16, 17]]
data = torch.tensor(x)

print(f'First Row of tensor: {data[0]} \n')
print(f'First Column of tensor: {data[:,0]} \n')
print(f'Second Column of tensor: {data[:, 1]} \n')

data[:, 2] += 10
print(f'Tensor value after alternation: {data} \n')

#2. Tensor Concatenation
x_1 = torch.cat([data, data], dim= -1) 
print(f'Tensor Concatination: {x_1} \n')
print(x_1.shape)

x_2 = torch.stack([data, data], dim= 1)
print(f'Stack Operation: {x_2} \n')
print(x_2.shape)

#3. Arithematic Operations 
y_1 = torch.tensor(x)
y_2 = torch.ones_like(data, dtype= torch.int64)
y = torch.zeros_like(data, dtype= torch.int64)

matmul_y = torch.matmul(y_1, y_2, out= y)
print(f'Matrix multiplication with function: {matmul_y}\n')

y = y_1 @ y_2
print(f'Matrix multiplication: {y}\n')

#3.2 Matrix Multiplication element-wise 
mul_y = torch.mul(y_1, y_2, out= y)
print(f'Matrix multiplication Element-wise with function: {mul_y}\n')

y = y_1 * y_2
print(f'Matrix multiplication Element-wise: {y}\n')

#In-place operations: - 
print(f'Absolute value of first matrix: {torch.abs_(y_1)}')

# Each tensor element values to be added with a value 
print(f'Addition of first row values: {y_1.add_(-1)}')

#4. Tensor and Numpy Array conversion & relation (each having same memory pointer values)
a = torch.ones(5)
b = a.numpy()

print(f'Tensor values a: {a} \n')
print(f'Numpy array values b: {b} \n')

#Change in numpy array will reflect change in tensor as both are pointing to same memory location
a.add_(10)
print(f'Tensor values a: {a} \n')
print(f'Numpy array values b: {b} \n')

n = np.array([1, 2, 3, 4, 5, 6])
t = torch.from_numpy(n)

print(f'Numpy array n values : {n} \n')
print(f'Tensor t values: {t} \n')

np.add(n, 10, out= n)

print(f'Numpy array n values : {n} \n')
print(f'Tensor t values: {t} \n')