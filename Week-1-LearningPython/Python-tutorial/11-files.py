#read a file
f = open("file.txt","r")
#prints all content
print(f.read())
f.close()


f = open("file.txt","r")
#print first 8 characters
print(f.read(8)) 
f.close()

f = open("file.txt","r")
#prints first line
print(f.readline())
#prints second line
print(f.readline())
f.close()

f = open("file.txt", "r")
for x in f:
  print(x) 
f.close() 

#write to file, w overwrites the content, a appends the content
f = open("file1.txt", "a")
f.write("Now the file has some content!")
f.write("Now the file has more content!")
f.close()
f = open("file1.txt","r")
print(f.read())
f.close()



f = open("file1.txt", "w")
f.write("Now the file has overwritten content!")
f.close()
f = open("file1.txt","r")
print(f.read())
f.close()

'''
hello
world
Nice to meet you!

hello
wo
hello

world

hello

world

Nice to meet you!

Now the file has overwritten content!Now the file has some content!Now the file has more content!
Now the file has overwritten content!
'''