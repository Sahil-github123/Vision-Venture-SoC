import numpy as np
arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
#ndim represents the number of dimensions. shape represents the size of the ndarray in each dimension
print(arr.shape)
print(arr.ndim)

print("\n")
#Reshaping, works only if the number calculations turns out correct
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
#1D to 2D
newarr = arr.reshape(4, 3)
print(newarr) 
print("\n")
#1D to 3D
newarr = arr.reshape(2, 3, 2)
print(newarr) 
#Notice that this is not a copy, but a view
print(newarr.base)

#2D to 3D
print("2D to 3D\n")
arr = np.array([[1,2,3], [4,5,6], [1,2,3], [4,5,6],[1,2,3], [4,5,6],[1,2,3], [4,5,6]])

print(arr.shape)
# changing the shape, -1 means any number which is suitable
newarr = np.reshape(arr, (4, 3, -1))
print(newarr) 
print(newarr.shape) 

#flatten an array, converts to 1D
arr = np.array([[1, 2, 3], [4, 5, 6]])
newarr = arr.reshape(-1)

