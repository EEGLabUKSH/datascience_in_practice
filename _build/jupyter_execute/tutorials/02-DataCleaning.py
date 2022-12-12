#!/usr/bin/env python
# coding: utf-8

# # s02: Data Cleaning

# <div class="alert alert-success">
# 'Data Cleaning' is the process of finding and either removing or fixing 'bad data'.  
#     
# By 'bad data' we mean missing, corrupt and/or inaccurate data points. 
# </div>

# In[1]:


# Imports
import numpy as np
import pandas as pd


# ## Missing Values

# <div class="alert alert-success">
# Missing Values are simply data points that are missing.
# </div>

# Missing values can be indicated in several ways. 
# 
# Values may be literally empty, or encoded as a special value, such as the Python 'None', or 'NaN', a numpy object (short for 'not a number'). 
# 
# Sometimes missing values are indicated by an arbitrarily chosen value, for example being indicated by some impossible value, such as '-999'. 
# 
# Missing values usually need dealing with before any analysis.

# ### Python - None Type

# In[2]:


# Python has the special value 'None', which can encode a missing, or null value
data_none = None


# In[3]:


# None is actually it's own type
print(type(None))


# In[4]:


# Note that 'None' acts like a null type (as if the variable doesn't exist)
not data_none


# In[5]:


# Since None is a null type, basic operations will fail when None is in the data
data_lst = [1, 2, 3, None]
sum(data_lst) / len(data_lst)


# ### Numpy - NaN

# In[ ]:


# Numpy also has a special value for 'not a number' - NaN
data_nan = np.nan


# In[ ]:


# It's actually a special float value
type(data_nan)


# In[ ]:


# It doesn't evaluate as null (unlike None)
not data_nan


# In[ ]:


# Numpy has multiple ways to write NaN - but they are all actually the same.
np.nan is np.NaN is np.NAN


# In[ ]:


# NaN values won't fail (unlike None) but they will return undefined (NaN) answers
dat_a = np.array([1, 2, 3, np.nan])
print(np.mean(dat_a))


# In[ ]:


# You can tell numpy to do calculations, ignoring NaN values, but you have to explicitly tell it to do so
print(np.nanmean(dat_a))


# ### The 'Art' of Data Cleaning
# 
# Dealing with missing data is a decision point: what do you do?
# - Do you drop the observation?
#     - What if this entails dropping a lot of observations?
# - Do you keep it, but ignore it in any calculations?
#     - What if you end up with different N's in different calculcations?
# - Do you recode that data point?
#     - What do you recode it to?

# ### Impossible Values
# 
# Be wary that datasets may also encode missing data as a special value - for example using '-999' for missing age. 
# 
# These have to be dealt with, or they will skew your results.
# 
# Data cleaning includes checking for and dealing with impossible values. Impossible values can also occur due to encoding or data entry errors. 

# ## Data Cleaning in Pandas
# 
# Example problem: we have two separate files that collectively have id number, age, weight, and height for a group of people. 
# 
# Let's say that ultimately, we are interested in how age relates to height. We might want to ask, for example, if older people have a lower average height than younger people (is it really true that older people shrink!?)
# 
# Data Files:
# - messy_data.json, has id & height information
# - messy_data.csv, has id, age, & weight information

# In[ ]:


# Load in the json file
url = 'https://raw.githubusercontent.com/BioPsychKiel/datascience_in_practice/main/tutorials/files/messy_data.json'
df1 = pd.read_json(url)

# Since JSON files read in columns alphabetically, re-arrange columns
df1 = df1[['id', 'height']]


# In[ ]:


# Check out the data. We have a NaN value!
df1


# In[ ]:


# Let's use pandas to drop the NaN value(s)
#  Note the inplace argument: this performs the operation on the dataframe we call it on
#   without having having to return and re-save the dataframe to a new variable
df1.dropna(inplace=True)


# In[ ]:


# Check out the data after dropping NaNs
df1


# In[ ]:


# Read in the CSV data file
df2 = pd.read_csv('https://raw.githubusercontent.com/BioPsychKiel/datascience_in_practice/main/tutorials/files/messy_data.csv')


# In[ ]:


# Check out the data
df2


# Note that we have another NaN value! However, it is in the weight column, a feature we actually are not planning to use for our current analysis. If we drop NaN's from this dataframe, we are actually rejecting good data - since we will drop subject 1, who actually does have the age and height information we need. 

# In[ ]:


# So, since we don't need it, lets drop the weight column instead
df2.drop('weight', axis=1, inplace=True)


# In[ ]:


# Let's check if there are any NaN values in the age column (that we do need)
#  isnull() return booleans for each data point indicating whether it is NaN or not
#    We can sum across the boolean array to see how many NaN values we have
sum(df2['age'].isnull())


# There aren't any NaN values in the data column that we need! Let's proceed!

# In[ ]:


# Now lets merge our data together
#  Note that here we specify to use the 'id' column to combine the data
#    This means that data points will be combined based on them having the same id.
df = pd.merge(df1, df2, on='id')


# In[ ]:


# Check out our merged dataframe
df


# In[ ]:


# Check out basic descriptive statistics to see if things look reasonable
df.describe()


# So, it looks like our average age is about -300. That... doesn't seem right. 
# 
# At some point in data collection, missing age values seem to have been encoded as -999. We need to deal with these data. 

# In[ ]:


# Drop all rows with an impossible age
df = df[df['age'] > 0]


# In[ ]:


# So what is the actual average age?
df['age'].mean()


# In[ ]:


# Check out the cleaned data frame! It is now ready for doing real analysis with!
df


# Note that in this example the problematic or missing values were relatively easy to locate - since we could see all our data. In real datasets, we may have hundreds to thousands of rows and potentially dozens of columns. In those cases, searching manually for missing or problematic values will not work very well. Strategies and programmatic approaches for identifying and dealing any bad values are necessary for any data analysis project. 

# ### Data Cleaning Notes
# 
# This is really just the start of data cleaning - getting data into a fit shape for analysis can include a considerable amount of exploration and work to ensure high quality data goes into the analysis. 
# 
# Tips for data cleaning:
# - Read any documentation for the dataset you have
#     - Things like missing values might be arbitrarily encoded, but should (hopefully) be documented somewhere
# - Check that data types are as expected. If you are reading in mixed type data, make sure you end up with the correct encodings
#     - Having numbers read in as strings, for example, is a common way data wrangling can go wrong, and this can cause analysis errors
# - Visualize your data! Have a look that the distribution seems reasonable (more on this later)
# - Check basic statistics. df.describe() can give you a sense if the data is really skewed
# - Keep in mind how your data were collected
#     - If anything comes from humans entering information into forms, this might take a lot of cleaning
#         - Fixing data entry errors (typos)
#         - Dealing with inputs using different units / formats / conventions
#     - Cleaning this kind of data is likely to take more manual work (since mistakes are likely idiosyncratic)
#     
# Note that in many real cases, visually scanning through data tables to look for missing or bad data is likely intractable, and/or very inefficient. Looking at your data will likely entail looking at distributions and descriptive statistics, as opposed to raw data. 

# <div class="alert alert-info">
# Quartz has a useful
# <a href="https://github.com/Quartz/bad-data-guide" class="alert-link">Bad Data Guide</a>,
# and the 
# <a href="http://pandas.pydata.org/pandas-docs/stable/tutorials.html" class="alert-link">Pandas tutorials</a>
# have lots of relevant materials, including a chapter (#7) on data cleaning.
# </div>

# ## Tasks

# In[ ]:


url_goodreads = 'https://raw.githubusercontent.com/BioPsychKiel/datascience_in_practice/main/tutorials/files/goodreads.txt'


# In[ ]:


#Read the data into a dataframe
df = pd.read_csv(url_goodreads)

#Examine the first few rows of the dataframe
df.head()


# Oh dear. That does not quite seem to be right. We are missing the column names. We need to add these in! But what are they?
# 
# Here is a list of them in order:
# 
# ['rating', 'review_count', 'isbn', 'booktype','author_url', 'year', 'genre_urls', 'dir','rating_count', 'name']

# <div class="alert alert-danger">
# Task 2.3: Use the provided list to load the dataframe properly! Make sure to properly include the first line of the file as the first line of actual data (1 point).
# <p> </p>
# </div>

# <div class="alert alert-danger">
# Task 2.4: Remove all entries (rows) from the dataframe where no meaningful year is provided (1 point).
# <p> </p>
# </div>

# <div class="alert alert-danger">
# Task 2.5: What is the shortest name of any book in the dataset (1 point).
# <p> </p>
# </div>
