#!/usr/bin/env python
# coding: utf-8

# # Appendix: Data Wrangling

# <div class="alert alert-success">
# 'Data Wrangling' generally refers to transforming raw data into a useable form for your analyses of interest, including loading, aggregating and formating. 
# </div>

# In this notebook, we will focus on loading different types of data files. Other aspects of 'wrangling' such as combining different datasets will be covered in future tutorials, and are explored in the assignments.
# 
# Note: Throughout this notebook, we will be using `!` to run the shell command `cat` to print out the contents of example data files.

# ## Python I/O
# 
# Let's start with basic Python utilities for reading and loading data files. 

# <div class="alert alert-info">
# Official Python 
# <a href="https://docs.python.org/3/library/io.html" class="alert-link">documentation</a> 
# on input / output.
# </div>

# In[1]:


# Check out an example data file
get_ipython().system('cat files/data.txt')


# In[2]:


# First, explicitly open the file object for reading
file_obj = open('files/data.txt', 'r')

# You can then loop through the file object, grabbing each line of data
for line in file_obj:
    # Here we explicitly remove the new line marker at the end of each line (the '\n')
    print(line.strip('\n'))

# File objects then have to closed when you are finished with them
file_obj.close()


# Since opening and closing files basically always goes together, there is a shortcut to do both of them together, which is the `with` keyword. 
# 
# By using `with`, file objects will be opened, and then automatically closed at the end of the code block. 

# In[3]:


# Use 'with' keyword to open, read, and then close a file
with open('files/data.txt', 'r') as file_obj:
    for line in file_obj:
        print(line.strip('\n'))


# Using input / output functionality from standard library Python is a pretty 'low level' way to read data files. This strategy often takes a lot of work to organize and define the details of how files are organized and how to read them. For example, in the above simple example, we had to deal with the new line character explicitly. 
# 
# As long as you have reasonably well structured data files, using standardized file types, you can use higher-level functions that will take care of a lot of these details - loading data straight into `pandas` data objects, for example.

# ## File types
# 
# There are many different file types in which data may be stored. 
# 
# Here, we will start by examining CSV and JSON files. 

# ### CSV Files

# <div class="alert alert-success">
# 'Comma Separated Value' files store data, separated by comma's. Think of them like lists.
# </div>
# 
# <div class="alert alert-info">
# More information on CSV files from
# <a href="https://en.wikipedia.org/wiki/Comma-separated_values" class="alert-link">wikipedia</a>. 
# </div>

# In[4]:


# Let's have a look at a csv file (printed out in plain text)
get_ipython().system('cat files/data.csv')


# #### CSV Files with Python

# In[5]:


# Python has a module devoted to working with csv's
import csv


# In[6]:


# We can read through our file with the csv module
with open('files/data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        print(', '.join(row))


# #### CSV Files with Pandas

# In[7]:


# Pandas also has functions to directly load csv data
get_ipython().run_line_magic('pinfo', 'pd.read_csv')


# In[8]:


# Let's read in our csv file
pd.read_csv(open('files/data.csv'), header=None)


# As we can see, using `Pandas` save us from having to do more work (write more code) to use load the file. 

# ### JSON Files

# <div class="alert alert-success">
# JavaScript Object Notation files can store hierachical key/value pairings. Think of them like dictionaries.
# </div>
# 
# <div class="alert alert-info">
# More information on JSON files from
# <a href="https://en.wikipedia.org/wiki/JSON" class="alert-link">wikipedia</a>.
# </div>

# In[22]:


# Let's have a look at a json file (printed out in plain text)
get_ipython().system('cat files/data.json')


# In[23]:


# Think of json's as similar to dictionaries
d = {'firstName': 'John', 'age': '53'}
print(d)


# #### JSON Files with Python

# In[24]:


# Python also has a module for dealing with json
import json


# In[25]:


# Load a json file
with open('files/data.json') as dat_file:    
    dat = json.load(dat_file)


# In[26]:


# Check what data type this gets loaded as
print(type(dat))


# #### JSON Files with Pandas

# In[27]:


# Pandas also has support for reading in json files
get_ipython().run_line_magic('pinfo', 'pd.read_json')


# In[28]:


# You can read in json formatted strings with pandas
#  Note that here I am specifying to read it in as a pd.Series, as there is a single line of data
pd.read_json('{ "first": "Alan", "place": "Manchester"}', typ='series')


# In[29]:


# Read in our json file with pandas
pd.read_json(open('files/data.json'), typ='series')


# ## Conclusion
# 
# As a general guideline, for loading and wrangling data files, using standardized data files, and loading them with 'higher-level' tools such as `Pandas` makes it easier to work with data files. 
