import numpy as np

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
arr = np.concatenate((arr1, arr2))
print(arr) 

#2D along axis1
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])
arr = np.concatenate((arr1, arr2), axis=0)
print(arr)
print(arr.shape)
arr = np.concatenate((arr1, arr2), axis=1)
print(arr)
print(arr.shape)

#stack one on top of another
arr = np.stack((arr1, arr2), axis=0)
print(arr)
print(arr.shape)
arr = np.stack((arr1, arr2), axis=1)
print(arr)
print(arr.shape)

#splitting 1-D arrays
print('\n splitting 1D array \n')
arr = np.array([1, 2, 3, 4, 5, 6])
newarr=np.split(arr,3);
print(newarr)
#access each array
print(newarr[0])
#split method will fail here, hence used array_split
newarr1 = np.array_split(arr, 4)
print(newarr1)

#splitting 2D arrays
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
newarr = np.array_split(arr, 3, axis=1)
print('\n splitting 2D array \n')
print(newarr[0]) 

#searching arrays; returns index where that element is
arr = np.array([1, 2, 3, 4, 5, 4, 4])
x = np.where(arr == 4)
print(x) 
#values are odd
x = np.where(arr%2 == 1)
print(x) 

#sorting; returns a copy leaving original array unchanged

arr = np.array([[3, 2, 4], [5, 0, 1]])
print(np.sort(arr)) 

#filtering based on a boolean index list (x in below example)
#filter only elements > 42
arr = np.array([41, 42, 43, 44])
x = [False, False, True, True]
newarr = arr[x]
print(newarr) 

#Another way
x = arr > 42
newarr = arr[x]
print(x)
print(newarr) 
