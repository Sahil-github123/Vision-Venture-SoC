#Lists use Square brackets
#List allow duplicate values
mylist = ["apple", "banana", "cherry", 32, "apple", 6, 2.5]
print(mylist)
print(len(mylist))

#List items are indexed, first index starts at 0
print(mylist[2])
#negative measns start from end
print(mylist[-1])
print(mylist[3:6])

#List items are mutable i.e. can be changed
#last index is not included, it changes for 1 and 2 i.e. banana and cherry
thislist = ["apple", "banana", "cherry"]
thislist[1:3] = ["mango", "watermelon"]
print(thislist)

#Can insert more items than specified by index
thislist = ["apple", "banana", "cherry"]
thislist[1:2] = ["blackcurrant", "watermelon"]
print(thislist) 

#can inser less items than specified
thislist = ["apple", "banana", "cherry"]
thislist[1:3] = ["watermelon"]
print(thislist) 

#can insert without replacement via insert method
thislist = ["apple", "banana", "cherry"]
thislist.insert(2, "watermelon")
thislist.append("orange")
print(thislist)

#can remove items as well
thislist.remove("orange")
#pop is used to remove from specific index
thislist.pop(1)
print(thislist)
#no argument, removes last element
thislist.pop()
#can use del also
thislist = ["apple", "banana", "cherry"]
del thislist[0]
print(thislist)
#removes entire list
del thislist

#List comprehension
#newlist = [expression for item in iterable if condition == True]
fruits = ["apple", "banana", "cherry", "kiwi", "mango"]
newlist = [x for x in fruits if "a" in x]
print(newlist) 
newlist = [x.upper() for x in fruits] 
print(newlist) 


#Join lists; + creates a new list; while extend, extends the first list
list1 = ["a", "b", "c"]
list2 = [1, 2, 3]

list3 = list1 + list2
print(list3) 

list1 = ["a", "b" , "c"]
list2 = [1, 2, 3]

list1.extend(list2)
print(list1) 


#Few other methods
#sort
thislist = ["orange", "mango", "kiwi", "pineapple", "banana"]
thislist.sort()
print(thislist)
#descending sort
thislist.sort(reverse = True)
print(thislist)
thislist = ["orange", "mango", "kiwi", "pineapple", "banana"]
#reverses current order
thislist.reverse()
print(thislist)

#copy list
thislist = ["apple", "banana", "cherry"]
mylist = thislist.copy()
print(mylist)

#output
'''
['apple', 'banana', 'cherry', 32, 'apple', 6, 2.5]
7
cherry
2.5
[32, 'apple', 6]
['apple', 'mango', 'watermelon']
['apple', 'blackcurrant', 'watermelon', 'cherry']
['apple', 'watermelon']
['apple', 'banana', 'watermelon', 'cherry', 'orange']
['apple', 'watermelon', 'cherry']
['banana', 'cherry']
['apple', 'banana', 'mango']
['APPLE', 'BANANA', 'CHERRY', 'KIWI', 'MANGO']
['a', 'b', 'c', 1, 2, 3]
['a', 'b', 'c', 1, 2, 3]
['banana', 'kiwi', 'mango', 'orange', 'pineapple']
['pineapple', 'orange', 'mango', 'kiwi', 'banana']
['banana', 'pineapple', 'kiwi', 'mango', 'orange']
['apple', 'banana', 'cherry']
'''
