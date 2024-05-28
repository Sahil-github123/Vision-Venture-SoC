########### Commenting ##############
#This is a comment
print("hello world!")

#This is a comment
#written in
#more than just one line

"""
This is a comment
written in
more than just one line
"""

################ Variables ###############
##Variables; notice no semicolon ending, just newline

x = 4       # x is of type int
y = "hello" # y is of type str
x = "world" # x is now of type str
z = 4.5 	# z is float
c = 2+1j   # c is complex
print(y,x)
print(c)

##get the type
print(type(x))
print(type(z))


## casting
x = str(3)    # x will be '3'
y = int(3)    # y will be 3
z = float(3)  # z will be 3.0 

##Multiple Values to multiple variables; notice that the space is given after fruit-name for clarity in printing
x, y, z = "Orange ", "Banana ", "Cherry "
print(x,y,z)
print(x + y + z)
## '+' meaning different for integers
x = 5
y = 10
print(x + y)

## global, ignore function definition for now, covered later
x = "awesome"

def myfunc():
  x = "fantastic"
  print("Python is " + x)

myfunc()

print("Python is " + x) 


x = "awesome"

def myfunc():
  global x
  x = "fantastic"

myfunc()

print("Python is " + x)


###############formatted printing via f strings##########
name = 'Ravi'
age = 23
print(f"Hello, My name is {name} and I'm {age} years old.")
x = 20.123
print(f"{x:.1f}")
a=12
b=12
c=2022
print(a,b,c,sep="-")
print('09','12','2016', sep='-', end='@')

#output
'''hello world!
hello world
(2+1j)
<class 'str'>
<class 'float'>
Orange  Banana  Cherry
Orange Banana Cherry
15
Python is fantastic
Python is awesome
Python is fantastic
Hello, My name is Ravi and I'm 23 years old.
20.1
12-12-2022
09-12-2016@'''
