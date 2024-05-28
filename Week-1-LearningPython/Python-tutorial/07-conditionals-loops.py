#Indentation in if statement very important
a = 200
b = 200
if b > a:
  print("b is greater than a")
elif a == b:
  print("a and b are equal")
else:
  print("a is greater than b")

#shorthand:One line if else statement, with 3 conditions:
a = 33
b = 33
print("a>b") if a > b else print("=") if a == b else print("a<b") 

#Can use logical and, or, not etc also
a = 200
b = 33
c = 500
if a > b or a > c:
  print("At least one of the conditions is True")
  
  
#while loop, indentation is important
i = 1
while i < 6:
  print(i)
  if i == 3:
    break
  i += 1 
 
#for loop
fruits = ["apple", "banana", "cherry"]
for x in fruits:
  if x == "banana":
    continue
  print(x)

#The range() function returns a sequence of numbers, starting from 0 by default, and increments by 1 (by default), and ends at a specified number.
for x in range(6):
  print(x)  

for x in range(2, 30, 3):
  print(x)
  
 #nested loops
adj = ["red", "big", "tasty"]
fruits = ["apple", "banana", "cherry"]

for x in adj:
  for y in fruits:
    print(x, y)  

'''
a and b are equal
=
At least one of the conditions is True
1
2
3
apple
cherry
{0}
{0, 1}
{0, 1, 2}
{0, 1, 2, 3}
{0, 1, 2, 3, 4}
{0, 1, 2, 3, 4, 5}
2
5
8
11
14
17
20
23
26
29
red apple
red banana
red cherry
big apple
big banana
big cherry
tasty apple
tasty banana
tasty cherry
PS C:\Users\hp\Documents\Academics\CS 104\Python>  c:; cd 'c:\Users\hp\Documents\Academics\CS 104\Python'; & 'C:\Python312\python.exe' 'c:\Users\hp\.vscode\extensions\ms-python.python-2023.10.1\pythonFiles\lib\python\debugpy\adapter/../..\debugpy\launcher' '64217' '--' 'C:\Users\hp\Documents\Academics\CS 104\Python\07-conditionals-loops.py' 
a and b are equal
=
At least one of the conditions is True
1
2
3
apple
cherry
0
1
2
3
4
5
2
5
8
11
14
17
20
23
26
29
red apple
red banana
red cherry
big apple
big banana
big cherry
tasty apple
tasty banana
tasty cherry
'''