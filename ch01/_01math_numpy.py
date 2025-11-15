from typing import ClassVar
import numpy as np

print("=== 矩阵和向量 =============================================================\n")
x = np.array([1, 2, 3])
print(x.__class__)
print(x.ndim)

W = np.array([[1, 2, 3], [4, 5, 6]])
print(W.shape)
print(W.ndim)

print("\n=== 矩阵对应元素的运算 ======================================================\n")
W = np.array([[1, 2, 3], [4, 5, 6]])
X = np.array([[0, 1, 2], [3, 4, 5]])
print(W + X) # 对应元素相加
print(W * X) # 对应元素相乘

print("\n=== 广播 ====================================================================\n")
A = np.array([[1, 2], [3, 4]])
print(A * 10)

print("\n=== 向量内积 ==============================================================\n")
# 向量内积
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.dot(a, b))
print(a @ b)
print(a.dot(b))
print(a @ b)

print("\n=== 矩阵乘法 ==============================================================\n")
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(A @ B)
print(A.dot(B))
print(A @ B)

