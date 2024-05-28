#Python is an object oriented programming language.
# Almost everything in Python is an object, with its properties and methods. We have seen lot of methods used on string objects already!

# Class is an object constructor, blueprint for an object

#Built-in functions
#__init__ function is automatically called every time class is being used to create an object
# self parameter is a reference to the current instance of the class, and is used to access variables that belong to the class. It does not have to be named self , you can call it whatever you like, but it has to be the first parameter of any function in the class.

#The __str__() function controls what should be returned when the class object is represented as a string

#__function__ (replace function with some name, often are special functions within Python

class Person:
  def __init__(self, name, age):
    self.name = name
    self.age = age

  def __str__(self):
    return f"{self.name}({self.age})"
    
p1 = Person("John", 36)

print(p1.name)
print(p1.age) 
print(p1)

#using some other names (mysillyobject, abc) instead of self
#also notice another user-defined method
class Person:
  def __init__(mysillyobject, name, age):
    mysillyobject.name = name
    mysillyobject.age = age

  def myfunc(abc):
    print("Hello my age is ", abc.age)

p1 = Person("John", 36)
p1.myfunc() 
#can modify properties of an object
p1.age = 60
print(p1.age)


'''John
36
John(36)
Hello my age is  36
60
'''