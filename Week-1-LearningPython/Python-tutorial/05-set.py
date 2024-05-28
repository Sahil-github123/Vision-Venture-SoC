#set is written with curly brackets
#Duplicates ignored
thisset = {"apple", "banana", "cherry", "apple", True, 23, 23.0}
print(thisset)
print(len(thisset)) 
print(type(thisset))


#set is a collection which is unordered and unindexed.
#unordered means that the items in a set do not have a defined order.
#Set items can appear in a different order
print(thisset)
#You cannot access items in a set by referring to an index or a key but you can loop through via for loop or check if value present
print("banana" in thisset) 

#Cannot change items but can add add/remove items via add/update methods
thisset = {"apple", "banana", "cherry"}
thisset.add("orange")
print(thisset) 

#Can add elements of a set to a set
thisset = {"apple", "banana", "cherry"}
tropical = {"pineapple", "mango", "papaya"}
thisset.update(tropical)
print(thisset) 

#Can add elements of a list to a set as well
thisset = {"apple", "banana", "cherry"}
mylist = ["kiwi", "orange"]
thisset.update(mylist)
print(thisset) 

#remove items
#if item to removed does not exist, remove() will raise an error 
thisset.remove("banana")
#if item to removed does not exist, discard() will NOT raise an error
thisset.discard("mango")

#pop can remove only the last item; Sets are unordered, so when using the pop() method, you do not know which item that gets removed.
thisset = {"apple", "banana", "cherry"}
x = thisset.pop()
print(x)
print(thisset) 

#sets are like mathematical sets. you can do union, intersection etc. 
#Join sets
#union of two sets;  Both union() and update() will exclude any duplicate items.
#Note union returns a new set, while update updates the first set
set1 = {"a", "b" , "c", "a"}
set2 = {1, 2, 3, 2}
set3 = set1.union(set2)
print(set3) 
set1.update(set2)
print(set1) 

#interesection, much like union, returns a new set; where as intersection_update updates the first set 
x = {"apple", "banana", "cherry"}
y = {"google", "microsoft", "apple"}
z = x.intersection(y)
print(z) 
x = {"apple", "banana", "cherry"}
y = {"google", "microsoft", "apple"}
x.intersection_update(y)
print(x) 

# symmetric_difference: Return a set that contains all items from both sets, except items that are present in both; where as symm...update updates the first set 
x = {"apple", "banana", "cherry"}
y = {"google", "microsoft", "apple"}
z = x.symmetric_difference(y)
print(z)
x = {"apple", "banana", "cherry"}
y = {"google", "microsoft", "apple"}
x.symmetric_difference_update(y)
print(x) 

#output
'''
{True, 'cherry', 23, 'banana', 'apple'}
5
<class 'set'>
{True, 'cherry', 23, 'banana', 'apple'}
True
{'banana', 'orange', 'apple', 'cherry'}
{'cherry', 'papaya', 'banana', 'mango', 'apple', 'pineapple'}
{'kiwi', 'cherry', 'orange', 'apple', 'banana'}
banana
{'apple', 'cherry'}
{1, 2, 3, 'a', 'c', 'b'}
{1, 2, 3, 'a', 'c', 'b'}
{'apple'}
{'apple'}
{'banana', 'microsoft', 'cherry', 'google'}
{'cherry', 'microsoft', 'google', 'banana'}
'''