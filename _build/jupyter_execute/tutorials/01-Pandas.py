#!/usr/bin/env python
# coding: utf-8

# # s01: Pandas
# 
# <br>
# <br>
# <img src="img/pandas_logo.png" width="200px">
# <br>
# <br>
# 
# This is a quick introduction to Pandas.
# 

# ## Objectives of this session:
#    - Learn simple and some more advanced usage of pandas dataframes
#    - Get a feeling for when pandas is useful and know where to find more information
#    - Understand enough of pandas to be able to read its documentation.

# Pandas is a Python package that provides high-performance and easy to use
# data structures and data analysis tools.
# This page provides a brief overview of pandas, but the open source community
# developing the pandas package has also created excellent documentation and training
# material, including:
# 
# - a  [Getting started guide](https://pandas.pydata.org/getting_started.html) - including tutorials and a 10 minute flash intro
# - a [10 minutes to pandas tutorial](https://pandas.pydata.org/docs/user_guide/10min.html#min)
# - thorough [Documentation](https://pandas.pydata.org/docs/)
# - a [cheatsheet](https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf)
# - a [cookbook](https://pandas.pydata.org/docs/user_guide/cookbook.html#cookbook).

# Let's get a flavor of what we can do with pandas. We will be working with an
# example dataset containing the passenger list from the penguins, which is often used in Kaggle competitions and data science tutorials. First step is to load pandas:

# In[1]:


import pandas as pd # pd is the standard abbreviation
import numpy as np


# We can download the data from seaborn directly reading into a `DataFrame`:

# In[2]:


import seaborn as sns
pengs = sns.load_dataset("penguins")


# We can now view the dataframe to get an idea of what it contains and print some summary statistics of its numerical data:

# In[3]:


# print the first 5 lines of the dataframe
pengs.head()


# ## What’s in a dataframe?

# Clearly, pandas dataframes allows us to do advanced analysis with very few commands, but it takes a while to get used to how dataframes work so let’s get back to basics.

# As we saw above, pandas dataframes are a powerful tool for working with tabular data. A pandas `pandas.DataFrame` is composed of rows and columns:
# 
# <br>
# <img src="img/02_pd_table_dataframe.svg" width="400px">
# <br>

# Lets get some detailed information about the numerical data of the `df`.

# In[4]:


# print summary statistics for each column
pengs.describe()


# Ok, so we have information on pengouins data. With the summary statistics we see that the body mass is 4201 g, maximum flipper length is 231 mm, etc.
# 
# Let’s say we’re interested in the mean boddy mass per species. With two one-liners, we can find the average body mass and plot corresponding histograms (`pandas.DataFrame.groupby()`, `pandas.DataFrame.hist()`):

# In[5]:


print(pengs.groupby("species")["body_mass_g"].mean())


# In[6]:


pengs.hist(column='body_mass_g', bins=25)


# `groupby` is a powerful method which splits a dataframe and aggregates data in groups. We start by creating a new column `child` to indicate whether a pengouin was a child or not, based on the existing `body_mass_g` column. For this example, let’s assume that you are a child when you weight less than 4500 g:

# In[7]:


pengs["child"] = pengs["body_mass_g"] < 4000


# Now we can test the if the flipper length is different for childs by grouping the data on `species` and then creating further sub-groups based on `child`:

# In[8]:


pengs.groupby(["species", "child"])["flipper_length_mm"].mean()


# Each column of a dataframe is a `pandas.Series` object - a dataframe is thus a collection of series:

# In[9]:


# print some information about the columns
pengs.info()


# Now we can already see what columns are present in the `df`. Any easier way jsut to display the column names is to use the function `df.columns`.

# In[10]:


pengs.columns


# Unlike a NumPy array, a dataframe can combine multiple data types, such as numbers and text, but the data in each column is of the same type. So we say a column is of type `int64` or of type `object`.
# 
# Let’s inspect one column of the penguin data (first downloading and reading the datafile into a dataframe if needed, see above):

# In[11]:


pengs["flipper_length_mm"]
pengs.flipper_length_mm          # same as above


# <div class="alert alert-success">
# A single column of a data frame is often refered to as a "Series". 
# <p></p>
# </div>

# Now let us adress not just a single column, but also include the rows as well. Adressing a row with numbers is what Pandas calls the `index`:

# In[12]:


pengs.index


# We saw above how to select a single column, but there are many ways of selecting (and setting) single or multiple rows, columns and values. We can refer to columns and rows either by number or by their name (`loc`, `iloc`, `at`, `iat`):

# In[13]:


pengs.loc[0,"island"]          # select single value by row and column
pengs.loc[:10,"bill_length_mm":"body_mass_g"]  # slice the dataframe by row and column *names*
pengs.iloc[0:2,3:6]                      # same slice as above by row and column *numbers*

pengs.at[0,"flipper_length_mm"] = 42      # set single value by row and column *name* (fast)
pengs.at[0,"species"]           # select single value by row and column *name* (fast)
pengs.iat[0,5]                           # select same value by row and column *number* (fast)

pengs["is_animal"] = True             # set a whole column


# Dataframes also support boolean indexing, just like we saw for `numpy` arrays:

# In[14]:


idx_big = pengs["body_mass_g"] > 4500 
pengs[idx_big]


# Using the boolean index `idx_big`, we can select specific row from an entire `df` based on values from a single column. 

# <div class="alert alert-danger">
# Task 1.8: How many pengouins per species live on which island? Store the answer in a new variable. (2 points).
# <p> </p>
# </div>

# <div class="alert alert-danger">
# Task 1.9: Using boolean indexing, compute the mean flipper length among pengouins over and under the average body mass. (2 points).
# <p> </p>
# </div>

# ## Tidy data

# The above analysis was rather straightforward thanks to the fact that the dataset is *tidy*.
# 
# In short, columns should be variables and rows should be measurements, and adding measurements (rows) should then not require any changes to code that reads the data.
# 
# What would untidy data look like? Here’s an example from some run time statistics from a 1500 m running event:

# In[15]:


runners = pd.DataFrame([
            {'Runner': 'Runner 1', 400: 64, 800: 128, 1200: 192, 1500: 240},
            {'Runner': 'Runner 2', 400: 80, 800: 160, 1200: 240, 1500: 300},
            {'Runner': 'Runner 3', 400: 96, 800: 192, 1200: 288, 1500: 360},
            ])


# In[16]:


runners.head()


# In[17]:


runners = pd.melt(runners, id_vars="Runner",
            value_vars=[400, 800, 1200, 1500],
            var_name="distance",
            value_name="time"
            )


# In[18]:


runners.head()


# In this form it's easier to **filter**, **group**, **join** and **aggregate** the data, and it's also easier to model relationships
# between variables.
# 
# The opposite of melting is to *pivot* data, which can be useful to view data in different ways as we'll see below.
# 
# For a detailed exposition of data tidying, have a look at [this article](http://vita.had.co.nz/papers/tidy-data.pdf).

# ## Creating dataframes from scratch

# We saw above how one can read in data into a dataframe using the read_csv() function. Pandas also understands multiple other formats, for example using read_excel, read_json, etc. (and corresponding methods to write to file: to_csv, to_excel, to_json, etc.)
# 
# But sometimes you would want to create a dataframe from scratch. Also this can be done in multiple ways, for example starting with a numpy array (see [`DataFrame` docs](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html#pandas.DataFrame) 

# Let's create a `df` for our study and fill it with random numbers which do not make **any** sense at all.

# In[19]:


import random
import string

n = 6

ids = [''.join(random.choices(string.ascii_uppercase + string.digits, k=5)) for i in range(0,n)] # this line generates IDs from Uppercase letters A-Z and Number 0-9 of length 5

nms_cols = ['age','height','MoCA']
rand_age = np.random.randint(18,99,n)
rand_height = np.random.randint(150,200,n)
rand_moca = np.random.randint(0,30,n)
df = pd.DataFrame(list(zip(rand_age, rand_height, rand_moca)), index=ids, columns=nms_cols)
df.head()

