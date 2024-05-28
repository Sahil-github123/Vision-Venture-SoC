#functions defined using def keyword
#This function has 2 arguments

def my_function(fname, lname):
  print("Hello " + fname + " "+ lname)
  
my_function("Suraj", "Singh") 

# don't know how many arguments, use *
#function will receive a tuple of arguments,
def my_function(*kids):
  print("The youngest child is " + kids[-1])

my_function("Emil", "Tobias", "Linus", "Adam") 

#Another example:
def adder(*num):
    sum = 0   
    for n in num:
        sum = sum + n

    print("Sum:",sum)

adder(3,5)
adder(1,2,3,5,6)

#can pass list as argument as well
def my_function(food):
  for x in food:
    print(x)

fruits = ["apple", "banana", "cherry"]

my_function(fruits)

#Pass a dictionary of unkown size, use **
def intro(**data):
    print("\nData type of argument:",type(data))

    for key,value in data.items():
       print(key, " => ", value)

intro(Firstname="Sita", Lastname="Sharma", Age=22, Phone=1234567890)
intro(Firstname="John", Lastname="Wood", Email="johnwood@nomail.com", Country="Wakanda", Age=25, Phone=9876543210)

#return value
def my_function(x):
  return 5 * x

print(my_function(3))
print(my_function(5))

#a lambda function is a special type of function without the function name
#lambda arguments : expression
x = lambda a, b : a * b
print(x(5, 6)) 

# lambda that accepts one argument
greet_user = lambda name : print('Hey there,', name)
# lambda call
greet_user('Suraj')
#Notice that greet_user is a function; hence you could call it with argument Suraj
print(type(greet_user))

#The power of lambda is better shown when you use them as an anonymous function inside another function.
def myfunc(n):
  return lambda a : a * n
#mydoubler and mytriper are functions!
mydoubler = myfunc(2)
mytripler = myfunc(3)
print(mydoubler(11))
print(mytripler(11))

##############userinput###############
val = input("Enter your value: ")
print(val)

x, y = input("Enter two values: ").split()
print("Number of boys: ", x)
print("Number of girls: ", y)

'''
Hello Suraj Singh
The youngest child is Adam
Sum: 8
Sum: 17
apple
banana
cherry

Data type of argument: <class 'dict'>
Firstname  =>  Sita
Lastname  =>  Sharma
Age  =>  22
Phone  =>  1234567890

Data type of argument: <class 'dict'>
Firstname  =>  John
Lastname  =>  Wood
Email  =>  johnwood@nomail.com
Country  =>  Wakanda
Age  =>  25
Phone  =>  9876543210
15
25
30
Hey there, Suraj
<class 'function'>
22
33
Enter your value: 7
7
Enter two values: 4 3
Number of boys:  4
Number of girls:  3
'''

