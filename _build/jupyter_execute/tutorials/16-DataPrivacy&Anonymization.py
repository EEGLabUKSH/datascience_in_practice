#!/usr/bin/env python
# coding: utf-8

# # s08: Data Privacy & Anonymization
# 
# A lot of data, perhaps the vast majority of data typically used in data science, is, directly or indirectly, about people. 
# 
# Individuals have privacy rights regarding who can know or share information about specifically identified individuals. This is true in particular about certain classes of sensitive information. For example, health-related information has special protections. Regardless of the data type, data privacy and security should also be a key concern when analyzing human data.

# ## Information Privacy

# <div class="alert alert-success">
# Information (or Data) Privacy refers to the legal, ethical, and practical issues of collecting, using and releasing data in which there is identifiable information about people included in the dataset. It also deals with when and how to deal with data privacy issues, and how to protect users' privacy.
# </div>
# 
# <div class="alert alert-info">
# <a href=https://en.wikipedia.org/wiki/Information_privacy class="alert-link">Wikipedia</a>
# has an overview of information privacy.
# </div>

# ## Anonymization

# <div class="alert alert-success">
# Data Anonymization is a type of information sanitization - that is the removal of sensitive information - for the purpose of privacy protection. It is a procedure to modify a data set such that the individuals it reflect are anonymous. Typically this means the removal or personally identifiable information from data sets such that the identify of individuals contained in the data set are anonymous.
# </div>
# 
# <div class="alert alert-info">
# <a href="https://en.wikipedia.org/wiki/Data_anonymization" class="alert-link">Wikipedia</a>
# also has an overview of data anonymization.
# </div>

# Data protection and anonymization are interdisciplinary components of data science and data practice. Data protection includes everything from considerations of the ethics & legalities of data use, to the practical and technical challenges of protecting and anonymizing data. 
# 
# Anonymizing data typically comes down to removing any personally identifiable data from a dataset, or, if this information must be kept, separating the identifiable data from sensitive information. 
# 
# Part of the difficulty of data anonymization is that while we can provably demonstrate that a given dataset is anonymized, this rests on particular assumptions. Most notably, datasets are only provably anonymized under the assumption that no extra external information is available to be used to attempt to de-identify it. In practice, this means that de-anonymization of data can often be done by combining multiple datasets. By using information from multiple information sources, one can often use processes of elimination to decode the individuals included in a particular dataset.

# # Regulations
# 
# There are many official guidelines, rules and standards for data privacy and user identity protection, although much of it is case specific. 
# 
# At the minimum, what is legally required in terms of data protection depends, amongst other things, on:
# - What the data is / contains, and who it is about, 
#     - Certain data types, and/or populations may have special protections, for example health-related information.
# - Who owns the data and in what capacity they are acting (company, university, etc.)
#     - For example, regulations for scientific research are different than those for companies
# - User agreements / consent procedures that were in place when the data were collected. 
#     - Individuals have a right to self-determination in terms of what their data is used for. Data should only be used for things that are covered by it's terms of use / terms of collection / consent procedures.
# - What the data is to be used for.
#     - Often a combination of the what and the who, there may be specific regulations about how you must deal with, and what you can do, based on the goal of having and using the data.
# - Where the data was collected and where it is stored, and who it is about.
#     - Different regions (countries, etc) often have different regulations.
# 
# Much of these regulations apply more directly to the collection, storage, and release of datasets (rather than analysis), but aspects also apply to the use of datasets, including publicly available datasets. Available datasets often have a user agreement for using the data, and, in particular, attempting to identify individuals from datasets may at a minimum break user agreements, and/or (depending on the nature of the data) be illegal based on consumer and research subject protection laws. 

# ## Research Standards

# <div class="alert alert-success">
# Data collected and used for research purposes has it's own set of guidelines and requirements regarding the treatment of human subjects, and the collection, storage, use, and dissemination of data. These regulations focus, among other things, on the right to self-determination of human subjects to consent to what data is collected, and how it is used. Data collected for research purposes must follow restrictions based on these consent procedures. 
# </div>
# 
# <div class="alert alert-info">
# Research protections under the
# <a href="https://en.wikipedia.org/wiki/Declaration_of_Helsinki" class="alert-link">Declaration of Helsinki</a>.
# </div>

# ## HIPAA - Protection for Health Related Information

# <div class="alert alert-success">
# The Health Insurance Portability and Accountability Act (HIPAA) is a US federal government regulation that standardizes and protects individuals medical records and health related data. It includes terms for how data must be stored, and how they can be used & shared.
# </div>
# 
# <div class="alert alert-info">
# The official US federal government HIPAA information
# <a href="https://www.hhs.gov/hipaa/" class="alert-link">guidelines</a>
# include an overview of HIPAA.
# </div>

# ## Safe Harbour Method

# <div class="alert alert-success">
# Safe Harbor is an official agreement regarding how to deal with datasets that have personal data. It describes specific guidelines on what information to remove from datasets in order to anonymize them. It is a single set of data protection requirements shared across many contexts and countries.
# </div>
# 
# <div class="alert alert-info">
# The 
# <a href="https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/" class="alert-link">official documentation</a>
# for Safe Harbour includes guidelines on how to anonymize data.
# </div>

# The Safe Harbor method requires that the following identifiers of the individuals be removed:
# - Names
# - Geographic Subdivisions smaller than a state**
# - Dates (such as birth dates, etc), and all ages above 90
# - Telephone Numbers
# - Vehicle Identification Numbers
# - Fax numbers
# - Device identifiers and serial numbers
# - Email addresses
# - Web Universal Resource Locators (URLs)
# - Social security numbers
# - Internet Protocol (IP) addresses
# - Medical record numbers
# - Biometric identifiers, including finger and voice prints
# - Health plan beneficiary numbers
# - Full-face photographs and any comparable images
# - Account numbers
# - Certificate/license numbers
# - Any other unique identifying number, characteristic, or code
# 
# ** The first three numbers of the zip code can be kept, provided that more than 20,000 people live in the region covered by all the zip codes that share the same initial three digits (the same geographic subdivision). 

# ### Unique Identifiers
# 
# The goal of Safe Harbor, and Data Anonymization in general, is to remove any unique information that could be used to identify you. 
# 
# This is perhaps most obvious for things like names. Other, perhaps less obvious specifications of Safe Harbour, are also based on the that this information being in a dataset creates a risk for identification of individuals contained in the dataset. 
# 
# For example, while it may be innocuous to talk about a 37 year old male who lives in Los Angeles (as there are many candidates, such that the specific individual is not revealed), it might actually be quite obvious who the person is when talking about a 37 year old male who lives in Potrero, California, a town of about 700 people. This is the same reason ages above 90 have to be removed - even in a fairly large area, say San Diego, it may be fairly obvious who the 98 year old female participant is. 
# 
# Basically - any information that makes you stand out is liable to identify you. Anonymization attempts to remove these kinds of indications from the data, such that individuals do not stand out in a way that lets someone figure out who they are.
# 
# This also underlies the difficulty in protecting data in the face of multiple data sources, since collecting observations together makes it much easier to start to pick out people more uniquely. It may still be relatively easy to identify the 37 year old male from LA if you also happen to know (or figure out) that he has a poodle, is 5'6", works at UCLA, and was at Griffith Park on Saturday, April 15th. All of this extra information may be relatively easy to figure out by combining publicly available, or easily obtainable, data.
