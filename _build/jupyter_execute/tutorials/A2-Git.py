#!/usr/bin/env python
# coding: utf-8

# # Appendix: Version Control
# 
# This notebook explores version control, and using git & Github.

# <div class="alert alert-success">
# Version Control reflects a set of practices related to managing files, in particular managing different versions of files. 
# </div>
# 
# <div class="alert alert-info">
# You can read more about version control on
# <a href="https://en.wikipedia.org/wiki/Version_control" class="alert-link">Wikipedia</a>
#  or on the 
# <a href="https://git-scm.com/book/en/v2/Getting-Started-About-Version-Control" class="alert-link">git docs</a>.
# </div>

# <br>
# <br>
# <img src="https://raw.githubusercontent.com/COGS108/Tutorials/master/img/git.png" width="300px" class="center">
# <br>
# <br>

# <div class="alert alert-success">
# Git is a version control system: a tool to track changes in files, across multiple locations. 
# </div>
# 
# <div class="alert alert-info">
# Information for using git is available in the 
# <a href=https://git-scm.com/doc class="alert-link">official documentation</a>, 
# as well as in many external resources, collected in 
# <a href=https://github.com/openlists/GitResources class="alert-link">this list</a>.
# </div>

# <br>
# <br>
# <img src="https://raw.githubusercontent.com/COGS108/Tutorials/master/img/github.png" width="300px" class="center">
# <br>
# <br>

# <div class="alert alert-success">
# Github is company that offers version control services, using git.
#     
# Github is a hosting service, with git built in, or basically, a place on the internet to put code that is tracked with git. 
# </div>
# 
# <div class="alert alert-info">
# Click through to 
# <a href="https://github.com/" class="alert-link">Github</a>.
# </div>

# ## Using Git

# There are several ways to use git, including:
# - from the command line, typing in commands directly
# - using a graphical program to launch git commands
#     - This kind of program is referred to as a 'graphical user interface' (GUI).
#         - It basically just means you can click buttons to do things, rather than writing out commands
#         
# Either way the underlying commands, and code that gets executed, are the same. Behind the scenes, it all reduces to the same thing. 
# 
# You should use whichever approach you are most comfortable with. If you already know some command line programming, using git from command line can be useful, as generally slightly more functionality is available to you, with more specific control. Using a GUI is much more straight forward if you are not very familiar with the command line.

# ### Git Graphical User Interfaces (GUIs) e.g. GitHub Desktop

# <br>
# <br>
# <img src="https://raw.githubusercontent.com/COGS108/Tutorials/master/img/sourcetree.png" width="500px" class="center">
# <br>
# <br>

# If you plan to use a GUI, one option is SourceTree. 
# 
# <div class="alert alert-info">
# See the Source Tree
# <a href="https://confluence.atlassian.com/get-started-with-sourcetree" class="alert-link">documentation</a>.
# </div>

# ## Git repositories
# 
# A repository is simply a collection of files, like a folder on a computer. 
# 
# What git is mainly doing is keeping two (or more) versions of the same repository consistent.

# ### Local vs. Remote
# 
# Given a repository with multiple copies (or versions), we can refer to these copies:
# 
# - the 'local' copy, which is a copy of the repository on your computer (the one 'local' to you)
# - the 'remote' copy, which is a copy of the repository somewhere else, such as on Github
# 
# There is often a particular copy of the code that is referred to as the 'master', which just means it is the master version of the repository in question. Most typically, this is the copy of the code on Github - so there is one 'master' copy of the code on Github, and one or more people also have local copies of the code, with local updates. When local updates want to be shared, they can be sent to master, to update the main version of the code, for everyone.
# 
# Here we will consider the case of having two copies of the repository. What is described here can all be extended to multiple copies of the code, including multiple different remote copies of the repository.
# 
# The main function of git, as a version control system, is to automatically check all of the files in each copy of the repository, tracking any changes that happen. It then provides tools to synchronize between different copies when there are changes.

# ## Getting Code from Github
# 
# As a starting point, it if often the case that there is code available on Github, and you want to get a local copy. This allows you to access the code, to use, and potentially update it. Then, if you want, you can contribute your code updates back to the main version of the code up on Github. 
# 
# First, you need to get a local copy of the code. Git calls making a copy of a repository 'cloning'.
# 
# From command line, to clone a Github repository to your computer, use the clone command with the URL to the repository.
# - $ ``git clone 'repo_url'``
#     - For example, you can clone these materials with:
#         - ``git clone https://github.com/COGS108/Tutorials``

# ## Tracking & Propagating Changes
# 
# Once you have a local copy, connected to a remote repository, changes can go in two directions:
# 
# - sending changes you make locally to the remote
#     - this is called a `push`
# - updating your local copy with changes from the remote
#     - this is called a `pull`

# ### Tracking and Sending Changes from local -> remote
# 
# When working on local files, git has hierarchical 'levels' of how it is tracking files (or, of what it is paying attention to). These multiple levels are useful to organize groups of changes into well organized actions.
# 
# The repository can have many files in it, within which there are:
# - tracked files
#     - these are files that git is tracking
#     - any files that are not tracked by git are untracked, and won't be version controlled
# - staged files
#     - these are tracked files that have been indicated to be added to an update of the code
#     - staging allows for specifying which files to save updates to, and how to group files
# 
# The typical workflow is:
# - ``add``: select which changes, on which files, you would like git to add to staging area
#     - You can add multiple files together, each with their own changes
#     - If a file is untracked, this adds it to tracking
# - ``commit``: make a checkpoint, saving all files that have been added together
#     - These changes will be 'saved' together with a message (a commit log) about what the changes are
# - ``push``: send the changes to the remote
#     - These changes will now be available in the remote copy of the repository

# #### Example Workflow
# 
# First, in a local git repository, you make some changes to a file, or perhaps many files. 
# 
# You can now add files. This needs to be done for each file you changed, that we want to add to staging:
# - $ ``git add 'f_name'``
# 
# After you've added one or more files, you use a commit to save the status of these files:
# - $ ``git commit -m 'Commit message'``
# 
# The '-m' flag is an option to write your commit message directly with the command. If you don't add it, git will send you into a text editor for you to write a commit message there. You should always add an informative message about what you have changed / added.
# 
# Making small, incremental changes and commiting often, with detailed messages means that your Github log can serve as a history for your project. This is useful as a way to use version control to keep track of a projects history. This also allows you to step back to an older version of the code if everything suddenly breaks.
# 
# A committed change is still only stored in your local copy. To update the remote repostory, you have to push:
# - $ ``git push``
# 
# You do not have to push after each commit, you can push many commits at once. 
# 
# Once you have pushed, you should see your updates have been added to the remote repository, on Github.

# ### Getting Changes from remote -> local
# 
# Git refers to updating a local copy with changes from a remote copy as 'pulling'
# 
# On command line, if there are changes on the remote branch, use the 'git pull' to copy those changes to your local: copy.
# 
# - $ ``git pull``

# ## Git Cheatsheet
# 
# The most common git functions are:
# 
# - ``git status``
#     - Check the status of a git repository
# - ``git add 'file'``
#     - Add a file to staging area
# - ``git commit -m 'message'``
#     - Log a 'save point' of all changes in the staging area.
# - ``git push``
#     - Copy commits to remote
# - ``git diff 'file'``
#     - Check what has changed in file since last commit
# - ``git clone 'repo'``
#     - Create a local copy of a git repository
# - ``git pull``
#     - Update your local copy of a git repository from the remote
# - ``git log``
#     - Check the log of git commits

# ## Advanced Git: Branching & Merging
# 
# Git has many other functionalities, including [branching](https://git-scm.com/book/en/v1/Git-Branching/) and [merging](https://git-scm.com/docs/git-merge), that are worth exploring as you get used to using git, and work on larger projects.

# ## External Resources and Tutorials

# <div class="alert alert-info">
# There are many tutorials for working with Github, including an 
# <a href="https://try.github.io/levels/1/challenges/1" class="alert-link">interactive tutorial</a>, 
# and
# <a href="https://guides.github.com/activities/hello-world/" class="alert-link">Hello World</a>,
# made by Github, and many other guides and tutorials, for example those available from
# <a href="http://lifehacker.com/5983680/how-the-heck-do-i-use-github" class="alert-link">LifeHacker</a>, 
# <a href="https://www.atlassian.com/git/tutorials" class="alert-link">Atlassian</a>, and
# <a href="http://neuroplausible.com/github" class="alert-link">Neuroplausible</a>.
# For a list of many resources, see
# <a href=https://github.com/openlists/GitResources class="alert-link">this list</a>.
# </div>
