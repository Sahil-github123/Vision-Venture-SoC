import numpy as np
print(np.__version__)

# Can create a ndarray object by using the array() method
#Below passes list as an argument, can use tuple also, see commented line
arr = np.array([1, 2, 3, 4, 5])
#arr = np.array((1, 2, 3, 4, 5))
print(arr)
print(type(arr))
print(arr.dtype) 

#create array as 4 byte integer
arr = np.array([1, 2, 3, 4], dtype='i4')
print(arr)
print(arr.dtype) 
arr = np.array([1, 2, 3], dtype = complex) 
print(arr)
print(arr.dtype) 

#convery from one type to another via astype() method
#data types
'''
    i - integer
    b - boolean
    u - unsigned integer
    f - float
    c - complex float
    m - timedelta
    M - datetime
    O - object
    S - string
    U - unicode string
    V - fixed chunk of memory for other type ( void )
'''
arr = np.array([1.1, 2.1, 3.1])
newarr = arr.astype('i')
print(newarr)
print(newarr.dtype) 

#convert list to array; can convert tuple also
x = [1,2,3] 
arr = np.asarray(x, dtype = float) 
print(arr)


#Dimensions of array
#0-D
arr = np.array(42)
print(arr.ndim) 
#1-D
arr = np.array([1, 2, 3, 4, 5])
print(arr.ndim) 
#2-D
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.ndim) 
#3-D
arr= np.array([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
print(arr.ndim) 

arr = np.array([1, 2, 3, 4], ndmin=5)
print(arr)
print('number of dimensions :', arr.ndim) 

#copy (new array owns the data) vs view (presents just a view of the original array)
arr = np.array([1, 2, 3, 4, 5])
x = arr.copy()
y = arr.view()
x[0]=10
y[1]=10
arr[2]=15
print(arr)
print(x)
print(y)
#base tells if the array owns the data or is just a view, view will return the org array
print(x.base)
print(y.base) 

'''1.22.4
[1 2 3 4 5]
<class 'numpy.ndarray'>
int64
[1 2 3 4]
int32
[1.+0.j 2.+0.j 3.+0.j]
complex128
[1 2 3]
int32
[1. 2. 3.]
0
1
2
3
[[[[[1 2 3 4]]]]]
number of dimensions : 5
[ 1 10 15  4  5]
[10  2  3  4  5]
[ 1 10 15  4  5]
None
[ 1 10 15  4  5]
'''
