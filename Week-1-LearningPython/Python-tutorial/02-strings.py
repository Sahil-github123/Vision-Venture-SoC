#string assignment; 
a = "hello"
#multiline string assignment
a = """Lorem ipsum dolor sit amet,
consectetur adipiscing elit,
sed do eiusmod tempor incididunt
ut labore et dolore magna aliqua."""

#strings as arrays
print(a)
print(a[1])
print(a[122])
print(len(a))
print(a[2:5])
print(a[:5])
print(a[120:])
#negative index means slice from the end of the string
print(a[-5:-1])

#Use of operators
#Is the word sit in the variable a?
print("sit" in a)
print("dolor" not in a)


#Built in methods on strings 
a = " hello world !!!   "
print(a.upper())
print(a.strip()) #removes spaces only at beginning and end
print(a.replace("l","i"))

#format string, this is another way, more recent is use of f-string as covered in 01-example.py
quantity = 3
itemno = 567
price = 49.95
myorder = "I want {} pieces of item {} for {} dollars."
print(myorder.format(quantity, itemno, price))

#output 
'''Lorem ipsum dolor sit amet,
consectetur adipiscing elit,
sed do eiusmod tempor incididunt
ut labore et dolore magna aliqua.
o
.
123
rem
Lorem
ua.
iqua
True
False
 HELLO WORLD !!!
hello world !!!
 heiio worid !!!
I want 3 pieces of item 567 for 49.95 dollars.'''