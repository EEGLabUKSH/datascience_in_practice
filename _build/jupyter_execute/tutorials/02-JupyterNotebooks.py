#!/usr/bin/env python
# coding: utf-8

# # Jupyter Notebooks
# 
# <br>
# <br>
# <img src="https://raw.githubusercontent.com/COGS108/Tutorials/master/img/jupyter.png" width="200px">
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

# ## Menu Options & Shortcuts
# 
# To get a quick tour of the Jupyter user-interface, click on the 'Help' menu, then click 'User Interface Tour'.
# 
# There are also a large number of useful keyboard shortcuts. Click on the 'Help' menu, and then 'Keyboard Shortcuts' to see a list. 

# ## Cells

# <div class="alert alert-success">
# The main organizational structure of the notebook are 'cells'.
# </div>

# Cells, can be markdown (text), like this one or code cells (we'll get to those).

# ### Markdown cells

# Markdown cell are useful for communicating information about our notebooks.
# 
# They perform basic text formatting including italics, bold, headings, links and images.
# 
# Double-click on any of the cells in this section to see what the plain-text looks like. Run the cell to then see what the formatted Markdown text looks like.

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

# If we don't use the markdown syntax for links, it will just show the link itself as the link text: https://www.markdownguide.org/cheat-sheet/

# ### LaTeX-formatted text

# $$ P(A \mid B) = \frac{P(B \mid A) \, P(A)}{P(B)} $$

# ### Code Cells
# 
# Code cells are cells that contain code, that can be executed. 
# 
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


# Import numpy for examples
import numpy as np


# In[9]:


# Check the docs for a numpy array
get_ipython().run_line_magic('pinfo', 'np.array')


# In[10]:


# Check the full source code for numpy append function
get_ipython().run_line_magic('pinfo2', 'np.append')


# In[11]:


# Get information about a variable you've created
get_ipython().run_line_magic('pinfo', 'my_string')


# ## Autocomplete

# <div class="alert alert-success">
# Jupyter also has 
# <a href="https://en.wikipedia.org/wiki/Command-line_completion" class="alert-link">tab complete</a>
# capacities, which can autocomplete what you are typing, and/or be used to explore what code is available.  
# </div>

# In[12]:


# Move your cursor just after the period, press tab, and a drop menu will appear showing all possible completions
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
# 
# The numbers in the square brackets to the left of a cell show which cells have been run, and in what order.
# 
# However, it can also be easy to lose track of what has already been declared / imported, leading to unexpected behaviour from running cells.
# 
# The kernel is what connects the notebook to your computer behind-the-scenes to execute the code. 
# 
# It can be useful to clear and re-launch the kernel. You can do this from the 'kernel' drop down menu, at the top, optionally also clearing all ouputs.

# ## Magic Commands

# <div class="alert alert-success">
# 'Magic Commands' are a special (command-line like) syntax in IPython/Jupyter to run special functionality. They can run on lines and/or entire cells. 
# </div>
# 
# <div class="alert alert-info">
# The iPython <a href="http://ipython.readthedocs.io/en/stable/interactive/magics.html" class="alert-link">documentation</a> has more information on magic commands.
# </div>

# Magic commands are designed to succinctly solve various common problems in standard data analysis. Magic commands come in two flavors: line magics, which are denoted by a single % prefix and operate on a single line of input, and cell magics, which are denoted by a double %% prefix and operate on multiple lines of input.

# In[ ]:


# Access quick reference sheet for interactive Python (this opens a reference guide)
get_ipython().run_line_magic('quickref', '')


# In[ ]:


# Check a list of available magic commands
get_ipython().run_line_magic('lsmagic', '')


# In[ ]:


# Check the current working directory
get_ipython().run_line_magic('pwd', '')


# In[ ]:


# Check all currently defined variables
get_ipython().run_line_magic('who', '')


# In[ ]:


# Chcek all variables, with more information about them
get_ipython().run_line_magic('whos', '')


# In[ ]:


# Check code history
get_ipython().run_line_magic('hist', '')


# ### Line Magics
# 
# 
# Line magics use a single '%', and apply to a single line. 

# In[ ]:


# For example, we can time how long it takes to create a large list
get_ipython().run_line_magic('timeit', 'list(range(100000))')


# ### Cell Magics
# 
# Cell magics use a double '%%', and apply to the whole cell. 

# In[ ]:


get_ipython().run_cell_magic('timeit', '', '# For example, we could time a whole cell\na = list(range(100000))\nb = [n + 1 for n in a]')


# ### Running terminal commands
# 
# Another nice thing about notebooks is being able to run terminals commands

# In[ ]:


# You can run a terminal command by adding '!' to the start of the line
get_ipython().system('pwd')

# Note that in this case, '!pwd' is equivalent to line magic '%pwd'. 
# The '!' syntax is more general though, allowing you to run anything you want through command-line 


# In[ ]:


get_ipython().run_cell_magic('bash', '', '# Equivalently, (for bash) use the %%bash cell magic to run a cell as bash (command-line)\npwd')


# In[ ]:


# List files in directory
get_ipython().system('ls')


# In[ ]:


# Change current directory
get_ipython().system('cd .')


# <div class="alert alert-info">
# For more useful information, check out Jupyter Notebooks 
# <a href="https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/" class="alert-link">tips & tricks</a>, and more information on how 
# <a href="http://jupyter.readthedocs.io/en/latest/architecture/how_jupyter_ipython_work.html" class="alert-link">notebooks work</a>.
# </div>
