import numpy as np

#Special types of arrays
#array with ones on the main diagonal and zeros elsewhere.
arr = np.eye(3)
print(arr)

#array with zeros everywhere
arr = np.zeros((2,3))
print(arr)

#array with ones everywhere
arr = np.ones((3,4))
print(arr)

#an array with evenly spaced values via step within a specified interval.
arr = np.arange(10,20,2, dtype=float)
print(arr)
#evenly spaced values between the interval as specified.
arr = np.linspace(10,20,5) 
print(arr)

print('\n Random values \n')
#random with 3x2 matrix
arr=np.random.rand(3,2)
print(arr)
#random integers from 0 to 100, 2-D with size 3x5
x = np.random.randint(100, size=(3, 5))
print(x) 


'''[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
[[0. 0. 0.]
 [0. 0. 0.]]
[[1. 1. 1. 1.]
 [1. 1. 1. 1.]
 [1. 1. 1. 1.]]
[10. 12. 14. 16. 18.]
[10.  12.5 15.  17.5 20. ]

 Random values 

[[0.27661432 0.26747248]
 [0.31057947 0.82400715]
 [0.95069099 0.80710974]]
[[13 31 19 62 69]
 [18 62 19 49 36]
 [99 21 72 26 79]]'''