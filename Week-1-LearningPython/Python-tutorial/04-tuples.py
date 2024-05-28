#tuples are written with round brackets
#tuples allow duplicate values
#tuples are ordered/indexed starting from index 0
#True below is a boolean data type
thistuple = ("apple", "banana", "cherry", "apple", 4.5, True, "cherry")
print(thistuple)
print(type(thistuple)) 
print(len(thistuple)) 
print(thistuple[1]) 
print(thistuple[:4])

#Tuples are unchangeable, or immutable; cannot use insert , append, remove methods etc
#This will throw an error, hence commenting, you can uncomment and see
#thistuple[1] = "mango"

#You can however convert the tuple into a list, change the list, and convert the list back into a tuple.
x = ("apple", "banana", "cherry")
y = list(x)
y[1] = "kiwi"
x = tuple(y)
print(x) 

#You are however allowed to add tuples to tuples
thistuple = ("apple", "banana", "cherry")
#tuple has only one item, you need to include a comma to identify it as a tuple
y = ("orange",)
thistuple += y
print(thistuple)

#can delete tuples
thistuple = ("apple", "banana", "cherry")
del thistuple

#output
'''
('apple', 'banana', 'cherry', 'apple', 4.5, True, 'cherry')
<class 'tuple'>
7
banana
('apple', 'banana', 'cherry', 'apple')
('apple', 'kiwi', 'cherry')
('apple', 'banana', 'cherry', 'orange')
'''