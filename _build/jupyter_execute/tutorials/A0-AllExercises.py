#!/usr/bin/env python
# coding: utf-8

# # Appendix: Exercise overview
# You can download [this](https://biopsychkiel.github.io/datascience_in_practice/_sources/tutorials/A0-AllExercises.ipynb) notebook as a template to fill out all exercises. A complete list of tasks has to be handed in by **28.02.2023, 23:59:59** via e-mail (j.welzel@neurologie.uni-kiel.de). There is a total of 37 points, you need 18 to pass (5 you already have by just sending an empty notebook). People who did the extra advent exercise have to score *14 points* to pass. 

# ## Chapter: Intro
# 
# Everybody get full score here, no need to put this in the final document.
# 
# ### 00 Introduction
# <div class="alert alert-danger">
# <ul>
# Task 0.1: Create an account at <a href="https://github.com/" class="alert-link">Github</a>. (1 point)
# </ul>
# </div>
# 
# <div class="alert alert-danger">
# <ul>
# Task 0.2: Download and install GitHub Desktop until the next session. (1 point)
# </ul>
# </div>
# 
# <div class="alert alert-danger">
# <ul>
# Task 0.3: Connect your Github account to your GitHub Desktop install. (1 point)
# </ul>
# </div>

# ## Chapter: Python
# 
# ### 01 JupyterNotebooks
# <div class="alert alert-danger">
# <ul>
# Task 1.1: Create a Jupyter Notebook using the Anaconda Navigator and name it "s01_jupyter_introduction". (1 point) (Of course everybody gets this point)
# </div>
# 
# <div class="alert alert-danger">
# <ul>
# Task 1.2: Close the Notebook and open it again. (1 point) (This one as well)
# </ul>
# </div>
# 
# <div class="alert alert-danger">
# <ul>
# Task 1.3: Create two variables. One variable with name "one" which holds the numeric value 1. The other variable should be named "two" and has to contain a string of value "two". (1 point)
# </ul>
# </div>
# 
# <div class="alert alert-danger">
# <ul>
# Task 1.4: Create a list which contains at least one variable of types (bool,int,float,str). (1 point)
# </ul>
# </div>
# 
# ### 01 Numpy
# <div class="alert alert-danger">
# <ul>
# Task 1.5: Create a random array with elements between 0 and 1. Then add 10 to all elements in the range [0.2, 0.7). (2 points)
# </ul>
# </div>
# 
# <div class="alert alert-danger">
# <ul>
# Task 1.6: 
# Create a (8, 8)-array called chess with a chessboard pattern (use the values 0 and 1 for this). There are many possible solutions, feel free to give several variants. For example, see the help for the function np.tile, or first create an array of all zeros and then insert ones at the appropriate places (e.g. by indexing appropriately). (3 points)
# </ul>
# </div>
# 
# ### 01 Plotting
# <div class="alert alert-danger">
# <ul>
# Task 1.7: This is a great exercise which is very close to real life. (3 points)
# 
# Your task is to select one visualization library (some need to be installed first - indoubt choose Matplotlib or Seaborn since they are part of Anaconda installation):
# <p>(i) <a href=https://matplotlib.org/stable/gallery/index.html class="alert-link">Matplotlib</a>: probably the most standard and most widely used</p>
# <p>(ii) <a href=https://seaborn.pydata.org/examples/index.html class="alert-link">Seaborn</a>: probably the most standard and most widely used</p>
# <p>(iii) <a href=https://yhat.github.io/ggpy/ class="alert-link">ggplot</a>: probably the most standard and most widely used</p>
# 
# - Browse the various example galleries (links above).
# - Select one example that simply interests you.
# - First try to reproduce this example in the Jupyter notebook.
# - Then try to print out the data that is used in this example just before the call of the plotting function to learn about its structure. Is it a pandas dataframe? Is it a NumPy array? Is it a dictionary? A list or a list of lists?
# </ul>
# </div>
# 
# ### 01 Pandas
# <div class="alert alert-danger">
# <ul>
# Task 1.8: How many pengouins per species live on which island? Store the answer in a new variable. (2 points).
# </ul>
# </div>
# 
# <div class="alert alert-danger">
# <ul>
# Task 1.9: Using boolean indexing, compute the mean flipper length among pengouins over and under the average body mass. (2 points).
# </ul>
# </div>

# ## Chapter: Data intro
# 
# ### 02 DataAnalysis
# <div class="alert alert-danger">
# <ul>
# Task 2.1: Import the numpy package as np and create a variable with a meaningful name. The variable should hold an 1darray with number as integers from 1 to 10. (2 points)
# </ul>
# </div>
# 
# <div class="alert alert-danger">
# <ul>
# Task 2.2: Using numpy create a 1darray of random integers (np.random.randint). Find the min, max, and mean of the array and print them in the command line using a formatted string. (2 points)
# </ul>
# </div>
# 
# ### 02 DataCleaning
# <div class="alert alert-danger">
# <ul>
# Task 2.3: Use the provided list to load the dataframe properly! Make sure to properly include the first line of the file as the first line of actual data (1 point).
# </ul>
# </div>
# 
# <div class="alert alert-danger">
# <ul>
# Task 2.4: Remove all entries (rows) from the dataframe where no meaningful year is provided (1 point).
# </ul>
# </div>

# ## Chapter: Stats Basics
# 
# ### 03 Testing Distributions
# <div class="alert alert-danger">
# <ul>
# Task 3.1: Write a function that plots two data vectors as histograms (e.g. `df["sepal_length"]` and `df["sepal_width"]`) and tests for normal distribution of the two. If one of the data vectors is not normally distributed, the function should not do any plotting, but print an error statement (2 points).
# </ul>
# </div>
# 
# <div class="alert alert-danger">
# <ul>
# Task 3.2: Write a function that automatically checks ALL numeric columns from the `df` for normal distribution and prints the result (1 point).
# </ul>
# </div>

# ## Chapter: Stats testing
# 
# ### 04 Statistical Comparisons
# <div class="alert alert-danger">
# <ul>
# Task 4.1: Write a function that takes distance and tip as inputs and tests for normal distribution of the two. Depending on the outcome of the test calculate the appropriate correltaion between the two. The function should return the correlation coefficient and the p-value as variables and print what type of correlation was used (2 points).
# </ul>
# </div>
# 
# <div class="alert alert-danger">
# <ul>
# Task 4.2: Make a scatterplot with distance on the x-axis and tip on the y-axis and label the plot accordingly (2 points).
# </ul>
# </div>

# ## Chapter: Applied Algorithms
# 
# ### 05.1 Clsutering
# <div class="alert alert-danger">
# <ul>
# Task 5.1: What are the default options for the sklearn KMeans function (number of clusters and maximal number of iterations). <b>No Code needed for this exercise</b> (1 point).
# </ul>
# </div>
# 
# <div class="alert alert-danger">
# <ul>
#     Task 5.2: Write a function that performs a KMeans classification between data vectors. It should take the vectors as inputs (data1, data2) and the number of cluster (n_clusters) as inputs. It should return the assigned labels and the mean error per cluster centroid. (1 point).
# </ul>
# </div>
# 
# ### 05.2 Classification
# <div class="alert alert-danger">
# <ul>
# Task 5.3: Load the amplitudes and labels in two variables using the pandas package, squeeze out the singelton dimension and transform them into a numpy array. (1 point).
# </ul>
# </div>
# 
# <div class="alert alert-danger">
# <ul>
#     Task 5.4: Using list comprehensions create a new variable which holds the cards names instead of the numerical labels (1='Ace of spades', 2 = 'Jack of clubs', .... ). The corresponsing card names are given below (1 point).
# </ul>
# </div>
# 
# Next we want to see if the participant did think about one card in particular (which was the task).
# 
# <div class="alert alert-danger">
# <ul>
#     Task 5.5.1: Create a SVM classifier with a 'poly' or 'linear' kernel (0.5 point). <br>
#     Task 5.5.2: Fit the amplitude data with the card labels (0.5 point).<br>
#     Task 5.5.3: Calculate predictions based on the amplitude data (0.5 point).<br>
#     Task 5.5.4: Show a classification result using your predictions and the card labels (0.5 point).<br>
# </ul>
# </div>
# 
# 
