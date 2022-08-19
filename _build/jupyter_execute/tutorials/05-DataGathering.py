#!/usr/bin/env python
# coding: utf-8

# # Data Gathering

# <div class="alert alert-success">
# Data Gathering is the process of accessing data and collecting it together.
# </div>

# This notebook covers strategies for finding and gathering data.
# 
# If you want to start by working on data analyses (with provided data) you can move onto the next tutorials, and come back to this one later.
# 
# Data gathering can encompass many different strategies, including data collection, web scraping, accessing data from databases, and downloading data in bulk. Sometimes it even includes things like calling someone to ask if you can use some of their data, and asking them to send it over. 

# ## Where to get Data
# 
# There are lots of way to get data, and lots of places to get it from. Typically, most of this data will be accessed through the internet, in one way or another, especially when pursuing indepent research projects. 
# 
# ### Institutional Access
# 
# If you are working with data as part of an institution, such as a company of research lab, the institution will typically have data it needs analyzing, that it collects in various ways. Keep in mind that even people working inside institutions, with access to local data, will data still seek to find and incorporate external datasets. 
# 
# ### Data Repositories
# 
# **Data repositories** are databases from which you can download data. Some data repositories allow you to explore available datasets and download datasets in bulk. Others may also offer **APIs**, through which you can request specific data from particular databases.
# 
# ### Web Scraping
# 
# The web itself is full of unstructured data. **Web scraping** can be done to directly extract and collect data directly from websites.
# 
# ### Asking People for Data
# 
# Not all data is indexed or accessible on the web, at least not publicly. Sometimes finding data means figuring out if any data is available, figuring out where it might be, and then reaching out and asking people directly about data access. If there is some particular data you need, you can try to figure out who might have it, and get in touch to see if it might be available.

# ### Data Gathering Skills
# 
# Depending on your gathering method, you will likely have to do some combination of the following:
# 
# - Direct download data files from repositories
# - Query databases & use APIs to extract and collect data of interest
# - Ask people for data, and going to pick up data with a harddrive
# 
# Ultimately, the goal is collect and curate data files, hopefully structured, that you can read into Python.

# ## Definitions: Databases & Query Languages
# 
# Here, we will introduce some useful definitions you will likely encounter when exploring how to gather data. 
# 
# Other than these definitions, we will not cover databases & query languages more in these tutorials. 

# <div class="alert alert-success">
# A database is an organized collection of data. More formally, 'database' refers to a set of related data, and the way it is organized. 
# </div>

# <div class="alert alert-success">
# A query language is a language for operating with databases, such as retrieving, and sometimes modifying, information from databases.
# </div>

# <div class="alert alert-success">
# SQL (pronounced 'sequel') is a common query language used to interact with databases, and request data.
# </div>
# 
# <div class="alert alert-info">
# If you are interested, there is a useful introduction and tutorial to SQL
# <a href="http://www.sqlcourse.com/intro.html" class="alert-link">here</a>
# as well as some useful 'cheat sheets' 
# <a href="http://www.cheat-sheets.org/sites/sql.su/" class="alert-link">here</a>
# and
# <a href="http://www.sqltutorial.org/wp-content/uploads/2016/04/SQL-cheat-sheet.pdf" class="alert-link">here</a>.
# </div>

# ## Data Repositories

# <div class="alert alert-success">
# A Data Repository is basically just a place that data is stored. For our purposes, it is a place you can download data from. 
# </div>
# 
# <div class="alert alert-info">
# There is a curated list of good data source included in the 
# <a href="https://github.com/COGS108/Projects" class="alert-link">project materials</a>.
# </div>

# For our purposes, data repositories are places you can download data directly from, for example [data.gov](https://www.data.gov/).

# ## Application Program Interfaces (APIs)

# <div class="alert alert-success">
# APIs are basically a way for software to talk to software - it is an interface into an application / website / database designed for software.
# </div>
# 
# <div class="alert alert-info">
# For a simple explanation of APIs go
# <a href="https://medium.freecodecamp.com/what-is-an-api-in-english-please-b880a3214a82" class="alert-link">here</a>
# or for a much broader, more technical, overview try
# <a href="https://medium.com/@mattburgess/apis-a-basic-primer-f8250602597d" class="alert-link">here</a>.
# </div>
# 
# <div class="alert alert-info">
# This
# <a href="http://www.webopedia.com/TERM/A/API.html" class="alert-link">list</a>
# includes a collection of commonly used and available APIs. 
# </div>

# APIs offer a lot of functionality - you can send requests to the application to do all kinds of actions. In fact, any application interface that is designed to be used programmatically is an API, including, for example, interfaces for using packages of code. 
# 
# One of the many things that APIs do, and offer, is a way to query and access data from particular applications / databases. For example, there is a an API for Google maps that allows for programmatically querying the latitude & longitude positions of given addresses. 
# 
# The benefit of using APIs for data gathering purposes is that they typically return data in nicely structured formats, that are relatively easy to analyze.

# ### Launching URL Requests from Python
# 
# In order to use APIs, and for other approaches to collecting data, it may be useful to launch URL requests from Python.
# 
# Note that by `URL`, we just mean a file or application that can be reached by a web address. Python can be used to organize and launch URL requests, triggering actions and collecting any returned data. 
# 
# In practice, APIs are usually special URLs that return raw data, such as `json` or `XML` files. This is compared to URLs we are typically more used to that return web pages as `html`, which can be rendered for human viewers (html). The key difference is that APIs return structured data files, where as `html` files are typically unstructured (more on that later, with web scraping). 
# 
# If you with to use an API, try and find the documentation for to see how you send requests to access whatever data you want. 
# 
# #### API Example
# 
# For our example here, we will use the Github API. Note that the URL we use is `api.github.com`. This URL accesses the API, and will return structured data files, instead of the html that would be returned by the standard URL (github.com).

# In[1]:


import pandas as pd

# We will use the `requests` library to launch URL requests from Python
import requests


# In[2]:


# Request data from the Github API on a particular user
page = requests.get('https://api.github.com/users/tomdonoghue')


# In[3]:


# In this case, the content we get back is a json file
page.content


# In[4]:


# We can read in the json data with pandas
pd.read_json(page.content, typ='series')


# As we can see above, in a couple lines of code, we can collect a lot of structured data about a particular user.
# 
# If we wanted to do analyses of Github profiles and activity, we could use the Github API to collect information about a group of users, and then analyze and compare the collected data. 

# ## Web Scraping

# <div class="alert alert-success">
# Web scraping is when you (programmatically) extract data from websites.
# </div>
# 
# <div class="alert alert-info">
# <a href="https://en.wikipedia.org/wiki/Web_scraping" class="alert-link">Wikipedia</a>
# has a useful page on web scraping.
# </div>

# By web scraping, we typically mean something distinct from using the internet to access an API. Rather, web scraping refers to using code to systematically navigate the internet, and extract information of internet, from html or other available files. Note that in this case one is not interacting directly with a database, but simply exploring and collecting whatever is available on web pages.
# 
# Note that the following section uses the 'BeautifulSoup' module, which is not part of the standard anaconda distribution. 
# 
# If you do not have BeautifulSoup, and want to get it to run this section, you can uncomment the cell below, and run it, to install BeautifulSoup in your current Python environment. You only have to do this once.

# In[5]:


#import sys
#!conda install --yes --prefix {sys.prefix} beautifulsoup4


# In[6]:


# Import BeautifulSoup
from bs4 import BeautifulSoup


# In[7]:


# Set the URL for the page we wish to scrape
site_url = 'https://en.wikipedia.org/wiki/Data_science'

# Launch the URL request, to get the page
page = requests.get(site_url)


# In[8]:


# Print out the first 1000 characters of the scraped web page
page.content[0:1000]


# Note that the source of the scraped web-page is a messy pile of HTML. 
# 
# There is a lot of information in there, but with no clear organization. There is some structure in the page though, delineated by HTML tags, etc, we just need to use them to parse out the data. We can do that with BeautifulSoup, which takes in messy documents like this, and parses them based on a specified format. 

# In[9]:


# Parse the webpage with Beautiful Soup, using a html parser
soup = BeautifulSoup(page.content, 'html.parser')


# In[10]:


# With the parsed soup object, we can select particular segments of the web page

# Print out the page title
print('TITLE: \n')
print(soup.title)

# Print out the first p-tag
print('\nP-TAG:\n')
print(soup.find('p'))


# From the soup object, you can explore the page in a more organized way, and start to extract particular components of interest.
# 
# Note that it is still 'messy' in other ways, in that there might or might not be a systematic structure to how the page is laid out, and it still might take a lot of work to extract the particular information you want from it.

# ### APIs vs. Web Scraping
# 
# Web scraping is distinct from using an API, even though many APIs may be accessed over the internet. Web scraping is different in that you are (programmatically) navigating through the internet, and extracting data of interest. 
# 
# Note:
# Be aware that scraping data from websites (without using APIs) can often be an involved project itself. Web scraping itself can take a considerable amount of time and work to get the data you want. 
# 
# Be aware that data presented on websites may not be well structured, and may not be in an organized format that lends itself to easy collection and analysis.
# 
# If you try scraping websites, you should also check to make sure you are allowed to scrape the data, and follow the websites terms of service. 
