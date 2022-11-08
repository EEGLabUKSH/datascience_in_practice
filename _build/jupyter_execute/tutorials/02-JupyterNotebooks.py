#!/usr/bin/env python
# coding: utf-8

# # s01: Jupyter Notebooks
# 
# <br>
# <br>
# <img src="img/jupyter.png" width="200px">
# <br>
# <br>
# 
# This is a quick introduction to Jupyter notebooks.

# <div class="alert alert-success">
# Jupyter notebooks are a way to combine executable code, code outputs, and text into one connected file.
# </div>
# 
# <div class="alert alert-info">
# The official documentation from project Jupyter is available 
# <a href="https://jupyter-notebook.readthedocs.io/en/stable/" class="alert-link">here</a>
# and they also have some example notebooks 
# <a href="https://github.com/jupyter/notebook/tree/master/docs/source/examples/Notebook" class="alert-link">here</a>.
# </div>

# <div class="alert alert-danger">
# Create a Jupyter Notebook using the Anaconda Navigator and name it "s01_jupyter_introduction". (1 point)
# </div>
# 
# <div class="alert alert-danger">
# Close the Notebook and open it again. (0.5 point)
# </div>

# ## Menu Options & Shortcuts
# 
# To get a quick tour of the Jupyter user-interface, click on the 'Help' menu, then click 'User Interface Tour'.
# 
# There are also a large number of useful keyboard shortcuts. Click on the 'Help' menu, and then 'Keyboard Shortcuts' to see a list. 

# ## Cells

# <div class="alert alert-success">
# The main organizational structure of the notebook are 'cells'.
# </div>
# 
# Cells, can be markdown (text), like this one or code cells.

# ### Markdown cells

# For communicating information about our notes, markdown cells are helpful.
# They apply basic text formatting such as bold, italics, headings, links, and photos.
# To view the plain text in any of the cells in this section, double-click on any one of them. Run the cell to observe how the Markdown formatting appears.

# # This is a heading
# 
# ## This is a smaller heading
# 
# ### This is a really small heading

# We can italicize my text either like *this* or like _this_.

# We can embolden my text either like **this** or like __this__.

# Here is an unordered list of items:
# * This is an item
# * This is an item
# * This is an item

# Here is an ordered list of items:
# 1. This is my first item
# 2. This is my second item
# 3. This is my third item

# We can have a list of lists by using identation:
# * This is an item
# * This is an item
# 	* This is an item
# 	* This is an item
# * This is an item

# We can also combine ordered and unordered lists:
# 1. This is my first item
# 2. This is my second item
# 	* This is an item
# 	* This is an item
# 3. This is my third item

# We can make a link to this [useful markdown cheatsheet](https://www.markdownguide.org/cheat-sheet/) as such.

# ### Code Cells
# 
# Code cells are cells that contain python code, that can be executed. 
# Comments can also be written in code cells, indicated by '#'. 

# In[1]:


# In a code cell, comments can be typed
a = 1
b = 2


# In[2]:


# Cells can also have output, that gets printed out below the cell.
print(a + b)


# In[3]:


# Define a variable in code
my_string = 'hello world'


# In[4]:


# Print out a variable
print(my_string)


# In[5]:


# Operations that return objects get printed out as output
my_string.upper()


# In[6]:


# Define a list variable
my_list = ['a','b','c']


# In[7]:


# Print out our list variable
print(my_list)


# ## Accessing Documentation

# <div class="alert alert-success">
# Jupyter has useful shortcuts. Add a single '?' after a function or class get a window with the documentation, or a double '??' to pull up the source code. 
# </div>

# In[8]:


# Get information about a variable you've created
get_ipython().run_line_magic('pinfo', 'my_string')


# ## Names

# Names in Python are nothing more than names for certain objects (in other programming languages, names are usually called variables). For example, the assignment `a = 1` can be interpreted as giving the name `a` to object `1`.

# Names can contain letters and digits, but they must not start with a digit. In principle, lowercase letters should be used (Python is case-sensitive). Names may be of any length, but should be chosen as short as possible (but as long as necessary). In addition, the character _ (the underscore) may be used to make a name more readable, e.g. to separate parts of words.
# 
# Names should be chosen sensibly, i.e. they should document the usage or the content (i.e. short names like i, n and x are only sensible in exceptional cases). It is also reasonable to use English names.
# 
# Examples for valid names are:

# In[9]:


number_of_students_in_class = 23  # to long
NumberOfStudents = 24  # Words should be separated with _
n_students = 25  # good name (short, meaningful)
n = 25  # less good (too unspecific), but OK in some cases


# <div class="alert alert-danger">
# Create two variables. One variable with name "one" which holds the numeric value 1. The other variable should be named "two" and has to contain a string of value "two". (1 point)
# </div>

# ## Data types

# These are the main data types in python we are going to use
# 
# * bool (Logical)
# * numeric 
#     * int (Integers)
#     * float (Decimal numbers)
# * str (String/ Zeichenkette)
# * list (List of objects)
# 
# We can use the command `type()` to find out the data type of a variable.

# In[10]:


b = True
type(b)


# In[11]:


a = 17
type(a)


# In[12]:


a = 23.221
type(a)


# Due to the limited computational precision with which computers represent decimal numbers, rounding errors may occur (decimal numbers generally cannot be represented exactly). Example:

# In[13]:


0.1 + 0.2 == 0.3


# In[14]:


0.1 + 0.2


# In[15]:


s1 = "Python"
s2 = 'Python'
print(type(s1))
print(type(s2))


# In[16]:


k = [1, 2, 18.33, "Python", 44]
type(k)


# <div class="alert alert-danger">
# Create a list which contains at least one variable of types (bool,int,float,str). (1 point)
# </div>

# ## Autocomplete

# <div class="alert alert-success">
# Jupyter also has 
# <a href="https://en.wikipedia.org/wiki/Command-line_completion" class="alert-link">tab complete</a>
# capacities, which can autocomplete what you are typing, and/or be used to explore what code is available.  
# </div>

# In[17]:


# Move your cursor just after the period, press the first letter of the command you want to execute or tab for all avaliable commands, and a drop menu will appear showing all possible completions
np.


# In[ ]:


# Autocomplete does not have to be at a period. Move to the end of 'ra' and hit tab to see completion options. 
ra


# In[ ]:


# If there is only one option, tab-complete will auto-complete what you are typing
ran


# ## Kernel & Namespace
# 
# You do not need to run cells in order! This is useful for flexibly testing and developing code. 
# The numbers in the square brackets to the left of a cell show which cells have been run, and in what order.
# However, it can also be easy to lose track of what has already been declared / imported, leading to unexpected behaviour from running cells.
# 
# The kernel is what connects the notebook to your computer behind-the-scenes to execute the code. 
# It can be useful to clear and re-launch the kernel. You can do this from the 'kernel' drop down menu, at the top, optionally also clearing all ouputs.

# <div class="alert alert-info">
# For more useful information, check out Jupyter Notebooks 
# <a href="https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/" class="alert-link">tips & tricks</a>, and more information on how 
# <a href="http://jupyter.readthedocs.io/en/latest/architecture/how_jupyter_ipython_work.html" class="alert-link">notebooks work</a>.
# </div>
