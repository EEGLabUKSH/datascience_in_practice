#!/usr/bin/env python
# coding: utf-8

# # Appendix: Adventcalender solutions

# Ab morgen geht in der WG der Adventskalender los. Leider gibt es noch keine Reihenfolge, wer wann ein Päckchen  öffnen darf. 
# 
# Natürlich kann seit diesem Jahr auch Python genutzt werden um das Problem zu lösen. Wer eine Lösung einer randomisierten Ziehung für die Mitbewohnenden (mitbewohner_innen = ["Charlotte", "Elli", "Malena", "Felix", "Vanessa", "Julius"]) - 01.12. bis 24.12.- bis nächste Woche zum Seminar einsendet, bekommt einen Joker für eine Woche Aufgaben. Alle sollten natürlich gleich viele Päckchen bekommen.

# ## Filenames examples

# In[1]:


good = 'this_is_a_good_file.ipynb' # all lower case and seperated by underscore
bad = 'This is a bad filename, by Julius.Welzel.ipynb' # includes whitespace and comma


# ## Example 1

# In[2]:


import numpy as np


# In[3]:


#variable erstellen, die zum Lösungsversuch mit np.random,choice passt
mitbewohner_innen = np.tile (["Charlotte", "Elli", "Malena", "Felix", "Vanessa", "Julius"], (4))
mitbewohner_innen


# In[4]:


# Kalendertürchen verteilen. Es wird ein Array ausgegeben, der die Reihenfolge der Türchenöffnenden angibt. 
verteilen = np.random.choice(mitbewohner_innen , size = 24, replace = True, p = [(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24),(1/24)])
verteilen


# In[5]:


# Kalendertürchen verteilen. Es wird ein Array ausgegeben, der die Reihenfolge der Türchenöffnenden angibt. 
n_tuerchen = 24
verteilen = np.random.choice(mitbewohner_innen , size = n_tuerchen, replace = True, p = [(1/n_tuerchen)] * n_tuerchen)
verteilen


# ## Example 2

# In[6]:


import numpy as np
import random

mitbewohner_innen = ["Charlotte", "Elli", "Malena", "Felix", "Vanessa", "Julius"]

datum = np.arange(1,25,1, dtype = int)
random.shuffle(datum)

x = 0
ind1 = 0
ind2 = 0

for i in mitbewohner_innen:
    ind2 = ind2 + 4
    list = datum[ind1:ind2]
    print(mitbewohner_innen[x])
    print(list)
    x = x+1
    ind1 = ind2


# In[7]:


new_test_list = list('christmas')


# In[ ]:


del list

import numpy as np
import random

mitbewohner_innen = ["Charlotte", "Elli", "Malena", "Felix", "Vanessa", "Julius"]

datum = np.arange(1,25,1, dtype = int)
random.shuffle(datum)

x = 0
ind1 = 0
ind2 = 0

for i in mitbewohner_innen:
    ind2 = ind2 + 4
    date_list = datum[ind1:ind2]
    print(mitbewohner_innen[x])
    print(date_list)
    x = x+1
    ind1 = ind2


# In[ ]:


new_test_list = list("x-mas")
print(new_test_list)


# ## Example 3

# In[ ]:


dates = np.arange(1, 25, 1, dtype = int)
names = list(np.repeat(["Charlotte", "Elli", "Malena", "Felix", "Vanessa", "Julius"], 4))
np.random.shuffle(names)
    
for i in range(len(dates)) :
    print(f"{dates[i]:>2}.12.2022: {names[i]}")


# ## One of many solutions

# In[ ]:


import pandas as pd
from random import shuffle

mitbewohner_innen = ["Charlotte", "Elli", "Malena", "Felix", "Vanessa", "Julius"]

n_days = 24
n_bewohner_innen = len(mitbewohner_innen)
n_draws = n_days // n_bewohner_innen
days = [i for i in range(1,n_days + 1)]
adv_cal = mitbewohner_innen * n_draws

shuffle(adv_cal)

pd_cal = pd.DataFrame(adv_cal, index=days, columns =['Adventskalender'])
print(pd_cal)

n_my_draws = pd_cal["Adventskalender"].str.count("Julius").sum()
print(f"\n Julius bekommt {n_my_draws} Päckchen")

