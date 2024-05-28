import numpy as np

arr = np.array([1, 2, 3, 4, 5, 6, 7])
#second element in the array, index starts at 0
print(arr[1]) 
#Slicing elements
print(arr[1:5]) #element at index 5 not printed 
#negative index means start from end
print(arr[-3:-1]) # element at -1 not printed
#stepping by 2
print(arr[1:5:2]) 
print(arr[::2])

#2D array
arr = np.array([[1,2,3,4,5], [6,7,8,9,10]])
print('The array is:')
print(arr)
#1 --> second row; 4 --> 5th element in that row, hence prints 10 
print('5th element on 2nd row: ', arr[1, 4]) 
#Slicing example
print('slice row 1,2 and columns 2-4')
print(arr[0:2, 1:4])  # arr[x,y] x=row index, y=column index

#returns array of items in the second column 
print('The items in the second column are:')  
print(arr[...,1]) 
print('\n')  

#returns array of items in the second row 
print('The items in the second row are:') 
print(arr[1,...]) 
print('\n')  

#3D array, prints the element 6
arr = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
print(arr[0, 1, 2]) 

#printing matrices of 3-D
for x in arr:
  print(x) 

#printing rows in the matrix
for x in arr:
  for y in x:
    print(y) 
    

#iterating through elements  
for x in arr:
  for y in x:
    for z in y:
      print(z)
      
#ndenumerate gives index information of every element
for idx, x in np.ndenumerate(arr):
  print(idx, x) 
  

 

'''2
[2 3 4 5]
[5 6]
[2 4]
[1 3 5 7]
The array is:
[[ 1  2  3  4  5]
 [ 6  7  8  9 10]]
5th element on 2nd row:  10
slice row 1,2 and columns 2-4
[[2 3 4]
 [7 8 9]]
The items in the second column are:
[2 7]


The items in the second row are:
[ 6  7  8  9 10]


6
[[1 2 3]
 [4 5 6]]
[[ 7  8  9]
 [10 11 12]]
[1 2 3]
[4 5 6]
[7 8 9]
[10 11 12]
1
2
3
4
5
6
7
8
9
10
11
12
(0, 0, 0) 1
(0, 0, 1) 2
(0, 0, 2) 3
(0, 1, 0) 4
(0, 1, 1) 5
(0, 1, 2) 6
(1, 0, 0) 7
(1, 0, 1) 8
(1, 0, 2) 9
(1, 1, 0) 10
(1, 1, 1) 11
(1, 1, 2) 12'''