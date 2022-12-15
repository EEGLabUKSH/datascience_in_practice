#!/usr/bin/env python
# coding: utf-8

# # s01: Python Basics

# <br>
# <br>
# <img src="img/python.png" width="400px">
# <br>
# <br>

# ## Objectives
# 
#    - Get a *very* short introduction to Python types and syntax
#    - Be able to follow the rest of the examples in the course, even if you don't understand everything perfectly.
# 
#    We expect everyone to be able to know the following basic material
#    to follow the course (though it is not *everything* you need to
#    know about Python).
# 

# ## Scalars
# 
# Scalar types, that is, single elements of various types:

# In[1]:


i = 42       # integer
i = 2**77    # Integers have arbitrary precision
g = 3.14     # floating point number
c = 2 - 3j   # Complex number
b = True     # boolean
s = "Hello!" # String (Unicode)
q = b'Hello' # bytes (8-bit values)


# ## Collections
# 
# Collections are data structures capable of storing multiple values.

# In[2]:


l = [1, 2, 3]                      # list
l[1]                               # lists are indexed by int
print(l)
l[1] = True                        # list elements can be any type and changes now
print(l)
print('\n')

d = {"Janne": 123, "Richard": 456} # dictionary
d["Janne"]
print(d.keys()) # this gets all the keys from a dict
print(d.values()) # this extract all values from dict
print('\n')


# ## Control structures
# 
# Python has the usual control structures, that is conditional statements and loops. For example, the `if` statement:

# In[3]:


x = 2
if x == 3:
    print('x is 3')
elif x == 2:
    print('x is 2')
else:
    print('x is something else')


# While ``while` loops loop until some condition is met:

# In[4]:


x = 0
while x < 42:
    print('x is ', x)
    x += 5


# For `for` loops loop over some collection of values:

# In[5]:


xs = [1, 2, 3, 4]
for x in xs:
    print(x)


# You can nicely iterate over elements of a list using the `for` loop independet of the datatype it is holding:

# In[6]:


l = [1,'b',True,]

for e in l:
    print(e)


# A common need is to iterate over a collection, but at the same time also have an index number. For this there is the `enumerate` function:

# In[7]:


xs = [1, 'hello', 'world']
for ii, x in enumerate(xs):
    print(ii, x)


# ## Functions and classes
# 
# Python functions are defined by the `def` keyword. They take a number of arguments, and return a number of return values.

# In[8]:


def say_hello(name):
    """Say hello to the person given by the argument"""
    print('Hello', name)
    return 'Hello ' + name

str_out = say_hello("Anne")


# <div class="alert alert-info">
# There is a great tutorial on python functions by Clemens Brunner 
# <a href=https://cbrnr.quarto.pub/python-22w-04/ class="alert-link">website</a>.
# </div>

# Classes are defined by the `class` keyword:

# In[9]:


class Hello:
    def __init__(self, name):
        self._name = name
    def say(self):
        print('Hello', self._name)

h = Hello("Richard")
h.say()

