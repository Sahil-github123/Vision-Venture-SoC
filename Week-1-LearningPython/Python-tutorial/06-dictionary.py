#Dictionaries are used to store data values in key:value pairs.
# A dictionary is a collection which is ordered, changeable and do not allow duplicates.

#Dictionaries are written with curly brackets, and have keys and values:

thisdict ={
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
print(thisdict)
print(thisdict["brand"])
print(len(thisdict))
print(type(thisdict)) 

#As of Python version 3.7, dictionaries are ordered. In Python 3.6 and earlier, dictionaries are unordered.

#Duplicate values will overwrite existing values:
thisdict =	{
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964,
  "year": 2020
}
print(thisdict) 
#Can change values
thisdict["year"] = 2023
thisdict.update({"year": 2023}) 
print(thisdict)


#access items
x = thisdict["model"]
print(x)
x = thisdict.get("model")
print(x)
x = thisdict.keys() 
print(x)
x = thisdict.values() 
print(x)
x = thisdict.items() 
print(x)
if "model" in thisdict:
  print("Yes, 'model' is one of the keys in the thisdict dictionary") 


#can add more items, both work
thisdict["color"] = "white"
thisdict.update({"gear": "auto"}) 
print(thisdict)


#can remove items; pop removes the item, popitem removes last item
thisdict.pop("model")
thisdict.popitem()
print(thisdict) 

#can copy
mydict = thisdict.copy()
print(mydict)

#Can create nested dictionaries
child1 = {
  "name" : "Emil",
  "year" : 2004
}
child2 = {
  "name" : "Tobias",
  "year" : 2007
}
child3 = {
  "name" : "Linus",
  "year" : 2011
}

myfamily = {
  "child1" : child1,
  "child2" : child2,
  "child3" : child3
} 

print(myfamily)

'''
{'brand': 'Ford', 'model': 'Mustang', 'year': 1964}
Ford
3
<class 'dict'>
{'brand': 'Ford', 'model': 'Mustang', 'year': 2020}
{'brand': 'Ford', 'model': 'Mustang', 'year': 2023}
Mustang
Mustang
dict_keys(['brand', 'model', 'year'])
dict_values(['Ford', 'Mustang', 2023])
dict_items([('brand', 'Ford'), ('model', 'Mustang'), ('year', 2023)])
Yes, 'model' is one of the keys in the thisdict dictionary
{'brand': 'Ford', 'model': 'Mustang', 'year': 2023, 'color': 'white', 'gear': 'auto'}
{'brand': 'Ford', 'year': 2023, 'color': 'white'}
{'brand': 'Ford', 'year': 2023, 'color': 'white'}
{'child1': {'name': 'Emil', 'year': 2004}, 'child2': {'name': 'Tobias', 'year': 2007}, 'child3': {'name': 'Linus', 'year': 2011}} 
'''