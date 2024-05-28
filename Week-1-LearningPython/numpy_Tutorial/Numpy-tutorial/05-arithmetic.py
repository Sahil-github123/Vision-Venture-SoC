import numpy as np
#same shaped arrays, arithmetic is straightforward; element by corresponding element
arr = np.array([[1, 2, 3], [4, 5, 6]])

print(arr * arr)

print(arr-arr)

#** is power of 
print(arr ** 2)

arr1 = np.array([[0, 2, 4], [7, 2, 8]])
print(arr1 > arr)

#arrays with different shapes
a = np.array([[0.0,0.0,0.0],[10.0,10.0,10.0],[20.0,20.0,20.0],[30.0,30.0,30.0]]) 
b = np.array([1.0,2.0, 3.0])  
   
print('First array:')
print(a) 
print('\n')  
   
print('Second array:') 
print(b)
print('\n') 

#The smaller array is broadcast to the size of the larger array so that they have compatible shapes. Not all shapes will work for this
print('First Array + Second Array') 
print(a + b)

#dot multiplication
A = [[6, 7, 4],
      [8, 9, 1]]
      
B = [[1, 3],
      [5, 7],
 	[2, 3]]

print('dot multiplication')
print(np.dot(A,B))

#transpose
print('transpose')
print(np.transpose(A))

#sine function
a = np.array([0,30,45,60,90]) 
print('Sine of different angles:')
# Convert to radians by multiplying with pi/180 
print(np.sin(a*np.pi/180))
print('\n')  

#stats
a = np.array([[30,65,70],[80,95,10],[50,90,60]]) 

print('Our array is:') 
print(a) 
print('\n')  

print('Applying median() function:')
print(np.median(a)) 
print('\n') 

print('Applying median() function along axis 0:') 
print(np.median(a, axis = 0)) 
print('\n') 
 
print('Applying median() function along axis 1:') 
print(np.median(a, axis = 1))

