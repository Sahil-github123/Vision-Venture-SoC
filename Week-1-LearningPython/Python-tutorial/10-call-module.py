#A file containing a set of functions you want to include in your application (module.py)

def greeting(name):
  print("Hello, " + name) 
  
person1 = {
  "name": "John",
  "age": 36,
  "country": "Norway"
}



#imports user written module.py and give it a different name for use here
import module as m
#Can also import only the person1 dictionary from the module as below. Commented since above already imported entire module
#from mymodule import person1

a = m.person1["age"]
print(a) 

#these are built-in modules
import platform
x = platform.system()
print(x)
#dir lists all the defined names belonging to all modules, including user defined ones 
y = dir(platform)
print(y) 

#math module with lots of math functions
import math 
x = math.sqrt(81)
y = math.floor(7.7)
z = math.pi
print(x,y,z) 

#regular expression module
import re

txt = "The rain in Spain"
x = re.search("^The.*Spain$", txt) 
txt = "The rain in Spain"
print(x)
x = re.split("\s", txt)
print(x) 

# importing sys module that handles command line arguments; be sure to give "3" arguments when running this code
import sys
  
# storing the arguments
program = sys.argv[0]
#arg1 = sys.argv[1]
#arg2 = sys.argv[2]
#arg3 = sys.argv[3]
  
# displaying the program name
print("Program name : " + program)
  
# displaying the arguments
#print("arg1 : " + arg1)
#print("arg2 : " + arg2)
#print("arg3 : " + arg3)
print("Number of arguments : ", len(sys.argv))
print(sys.argv)

'''36
Windows
['_Processor', '_WIN32_CLIENT_RELEASES', '_WIN32_SERVER_RELEASES', '__builtins__', '__cached__', '__copyright__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', '__version__', '_comparable_version', '_component_re', '_default_architecture', '_follow_symlinks', '_get_machine_win32', '_ironpython26_sys_version_parser', '_ironpython_sys_version_parser', '_java_getprop', '_libc_search', '_mac_ver_xml', '_node', '_norm_version', '_os_release_cache', '_os_release_candidates', '_os_release_line', '_os_release_unescape', '_parse_os_release', '_platform', '_platform_cache', '_pypy_sys_version_parser', '_sys_version', '_sys_version_cache', '_sys_version_parser', '_syscmd_file', '_syscmd_ver', '_uname_cache', '_unknown_as_blank', '_ver_output', '_ver_stages', '_win32_ver', '_wmi', '_wmi_query', 'architecture', 'collections', 'freedesktop_os_release', 'functools', 'itertools', 'java_ver', 'libc_ver', 'mac_ver', 'machine', 'node', 'os', 'platform', 'processor', 'python_branch', 'python_build', 'python_compiler', 'python_implementation', 'python_revision', 'python_version', 'python_version_tuple', 're', 'release', 'sys', 'system', 'system_alias', 'uname', 'uname_result', 'version', 'win32_edition', 'win32_is_iot', 'win32_ver']
9.0 7 3.141592653589793
<re.Match object; span=(0, 17), match='The rain in Spain'>
['The', 'rain', 'in', 'Spain']
Program name : C:\Users\hp\Documents\Academics\CS 104\Python\10-call-module.py
PS C:\Users\hp\Documents\Academics\CS 104\Python>  c:; cd 'c:\Users\hp\Documents\Academics\CS 104\Python'; & 'C:\Python312\python.exe' 'c:\Users\hp\.vscode\extensions\ms-python.python-2023.10.1\pythonFiles\lib\python\debugpy\adapter/../..\debugpy\launcher' '52922' '--' 'C:\Users\hp\Documents\Academics\CS 104\Python\10-call-module.py' 
C:\Users\hp\Documents\Academics\CS 104\Python\10-call-module.py:31: SyntaxWarning: invalid escape sequence '\s'
  x = re.split("\s", txt)
36
Windows
['_Processor', '_WIN32_CLIENT_RELEASES', '_WIN32_SERVER_RELEASES', '__builtins__', '__cached__', '__copyright__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', '__version__', '_comparable_version', '_component_re', '_default_architecture', '_follow_symlinks', '_get_machine_win32', '_ironpython26_sys_version_parser', '_ironpython_sys_version_parser', '_java_getprop', '_libc_search', '_mac_ver_xml', '_node', '_norm_version', '_os_release_cache', '_os_release_candidates', '_os_release_line', '_os_release_unescape', '_parse_os_release', '_platform', '_platform_cache', '_pypy_sys_version_parser', '_sys_version', '_sys_version_cache', '_sys_version_parser', '_syscmd_file', '_syscmd_ver', '_uname_cache', '_unknown_as_blank', '_ver_output', '_ver_stages', '_win32_ver', '_wmi', '_wmi_query', 'architecture', 'collections', 'freedesktop_os_release', 'functools', 'itertools', 'java_ver', 'libc_ver', 'mac_ver', 'machine', 'node', 'os', 'platform', 'processor', 'python_branch', 'python_build', 'python_compiler', 'python_implementation', 'python_revision', 'python_version', 'python_version_tuple', 're', 'release', 'sys', 'system', 'system_alias', 'uname', 'uname_result', 'version', 'win32_edition', 'win32_is_iot', 'win32_ver']
9.0 7 3.141592653589793
<re.Match object; span=(0, 17), match='The rain in Spain'>
['The', 'rain', 'in', 'Spain']
Program name : C:\Users\hp\Documents\Academics\CS 104\Python\10-call-module.py
Number of arguments :  1
'''
