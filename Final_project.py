#!/usr/bin/env python
# coding: utf-8

# # CSE351 PROJECT

# ## by Anthony Yang, Joshua Sinanan, Sahil Sarna

# In[1]:


import pandas as pd 
import matplotlib
import sklearn
import numpy as np
import seaborn as sns
from scipy import stats
import copy
import json
import matplotlib.pyplot as plt
a4_dims = (11.7, 8.27)


# ### LOADING IN DATASETS

# In[2]:


moviesDataFrame = pd.read_csv('movie/tmdb_5000_movies.csv') # Movies Dataset
creditsDataFrame = pd.read_csv('movie/tmdb_5000_credits.csv') # Credits Dataset


# ### Making An Update Function For Central Tendencies

# In[3]:


# Budget CTs
def updateBudgetCTs():
    budgetMean = round(moviesDataFrame['budget'].mean(),2)
    budgetMedian = moviesDataFrame['budget'].median()
    budgetSD = round(moviesDataFrame['budget'].std(),2)
    print("Budget Mean:", budgetMean)
    print("Budget Median:", budgetMedian)
    print("Budget SD:", budgetSD)


# In[4]:


# Revenue CTs
def updateRevenueCTs():
    revenueMean = round(moviesDataFrame['revenue'].mean(),2)
    revenueMedian = moviesDataFrame['revenue'].median()
    revenueSD = round(moviesDataFrame['revenue'].std(),2)
    print("Revenue Mean:", revenueMean)
    print("Revenue Median:", revenueMedian)
    print("Revenue SD:", revenueSD)


# ## Data Cleaning

# ### Checking for Null Values

# In[5]:


# Checking for null values in the Movies Data Frame
moviesDataFrame.isnull().sum()


# Dropping 'homepage' feature as 64.4% of the values are missing and this feature is not useful to us at all

# In[6]:


moviesDataFrame.drop(columns='homepage', inplace=True)


# Checking Budget Central Tendencies 

# In[7]:


updateBudgetCTs()


# Checking No. of entries with Budget : $0

# In[8]:


moviesDataFrame['budget'].replace(0, np.nan, inplace=True)
moviesDataFrame.isnull().sum() 


# 1037 Entries with Budget $0 (21.6% of the data is missing budget)

# Checking No. of entries with Budget : $0-$10000

# In[9]:


moviesDataFrame['budget'].replace(range(0,10000), np.nan, inplace=True)
moviesDataFrame.isnull().sum() 


# 1072 Entries with Budget $0-$10000 (22.3% of the data is missing budget or just very low for the data set, given that the mean is $29M)

# Interpolating the values for the Budget $0-$10000 to fit these entries better with the dataset

# In[10]:


moviesDataFrame = moviesDataFrame.interpolate()


# Plotting Budget Graph

# In[11]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.displot(moviesDataFrame['budget'], bins=30)


# In[12]:


#Checking Budget
Q1 = moviesDataFrame['budget'].quantile(0.25)
Q3 = moviesDataFrame['budget'].quantile(0.75)
IQR = Q3 - Q1
whisk = 2 #whisker coeff
print("Q1 = ", Q1)
print("Q3 = ", Q3)
print("IQR = ", IQR)


# In[13]:


fig, ax = plt.subplots(figsize=a4_dims)
sns.boxplot(x=moviesDataFrame['budget'],whis=whisk,ax=ax)


# Checking for Number of Outliers

# In[14]:


((moviesDataFrame['budget'] < (Q1 - whisk * IQR)) | (moviesDataFrame['budget'] > (Q3 + whisk * IQR))).sum()


# Dropping these Outliers

# In[15]:


#Dropping outliers
moviesDataFrame = moviesDataFrame[~((moviesDataFrame['budget'] < (Q1 - whisk*IQR)) | (moviesDataFrame['budget'] > (Q3 + whisk*IQR)))]


# Budget Central Tendencies after dropping Outliers

# In[16]:


updateBudgetCTs()


# Checking for Revenue Central Tendencies

# In[17]:


updateRevenueCTs()


# Checking for number of revenue records in the range of $0-$10000

# In[18]:


moviesDataFrame['revenue'].replace(range(0,10000), np.nan, inplace=True)
moviesDataFrame.isnull().sum() 


# Fixing these incorrect revenue values using interpolation method

# In[19]:


moviesDataFrame = moviesDataFrame.interpolate()
moviesDataFrame.isnull().sum()


# Revenue Central Tendencies after fixing incorrect values

# In[20]:


updateRevenueCTs()


# ### Correlation Analysis

# Finding all correlations in the dataset using Pearson

# In[21]:


correlation = moviesDataFrame.corr()
# changing the position of revenue column to compare correnlation easier
correlation = correlation[['revenue', 'budget','id','popularity','runtime','vote_average','vote_count']]
correlation


# In[22]:


fig, ax = plt.subplots(figsize=a4_dims)
sns.heatmap(correlation,square=True,linewidth=1,cmap=("RdYlGn_r"),center=0)


# #### After Reviewing the chart above:
# - The Strongest Correlation to Revenue is <b>Vote Count</b>
# - The Weakest Correlation to Revenue is <b>Vote Average</b>
# 
# #### If we don't consider the votes then:
# - The Strongest Correlation to Revenue is <b>Popularity</b>
# - The Weakest Correlation to Revenue is <b>Runtime</b>
# 
# (We can ignore 'id' in both cases as it's not related to anything)

# ## Merging Both Datasets

# Checking for record at index '97' of Movies Dataset

# In[23]:


# Checking for data at same index
print(moviesDataFrame.loc[97])


# Checking for record at index '97' of Credits Dataset

# In[24]:


# Checking for data at same index
print(creditsDataFrame.loc[97])


# Since we can see that both the datasets have same movie and can be linked to each other through their 'id' we can merge them. 

# Merging the Datasets together

# In[25]:


# Changing column name 'movie_id' in creditsDataFrame to 'id' for merging
creditsDataFrame.rename(columns= {'movie_id' : 'id'}, inplace = True)

# Merging the two dataframes through common column 'id'
mergedDataFrame = pd.merge(moviesDataFrame, creditsDataFrame, on='id')
mergedDataFrame.head()


# ## Extracting List and Dictionaries

# In[26]:


## Getting Production Countries
countries = mergedDataFrame.production_countries
countryList = []

for country in countries:
    singleCountry = json.loads(country)
    singleList = []
    for countryName in singleCountry:
        singleList.append(countryName['name'])
    countryList.append(singleList)

mergedDataFrame['production_countries'] = countryList


# In[27]:


## Getting Production Companies
companies = mergedDataFrame.production_companies
companyList = []

for company in companies:
    singleCompany = json.loads(company)
    singleList = []
    for companyName in singleCompany:
        singleList.append(companyName['name'])
    companyList.append(singleList)

mergedDataFrame['production_companies'] = companyList


# In[28]:


## Getting Genres
genres = mergedDataFrame.genres
genreList = []

for genre in genres:
    singleGenre = json.loads(genre)
    singleList = []
    for genreName in singleGenre:
        singleList.append(genreName['name'])
    genreList.append(singleList)

mergedDataFrame['genres'] = genreList


# In[29]:


## Getting Spoken Languages
languages = mergedDataFrame.spoken_languages
languageList = []

for language in languages:
    singleLanguage = json.loads(language)
    singleList = []
    for languageName in singleLanguage:
        singleList.append(languageName['name'])
    languageList.append(singleList)

mergedDataFrame['spoken_languages'] = languageList


# In[30]:


## Getting Keywords
keywords = mergedDataFrame.keywords
keywordList = []

for keyword in keywords:
    singleKeyword = json.loads(keyword)
    singleList = []
    for keywordName in singleKeyword:
        singleList.append(keywordName['name'])
    keywordList.append(singleList)

mergedDataFrame['keywords'] = keywordList


# In[31]:


## Getting Crew 
crew = mergedDataFrame.crew
crewList = []

for members in crew:
    singleCrew = json.loads(members)
    singleList = []
    for crewName in singleCrew:
        singleList.append(crewName['name'])
    crewList.append(singleList)

mergedDataFrame['crew'] = crewList


# In[32]:


## Getting Cast 
cast = mergedDataFrame.cast
castList = []

for members in cast:
    singleCast = json.loads(members)
    singleList = []
    for castName in singleCast:
        singleList.append(castName['name'])
    castList.append(singleList)

mergedDataFrame['cast'] = castList


# The DataFrame after extracting data from list and dictionaries 

# In[33]:


mergedDataFrame


# ## Getting the count of movie release for different times

# In[34]:


## Count number of movies released by day of week, month, and year ##
import datetime
from datetime import date
import calendar
## function to find day of week given date (https://www.geeksforgeeks.org/python-program-to-find-day-of-the-week-for-a-given-date/) ##  
def findDayOfWeek(year, month, day):
    release = datetime.date(int(year), int(month), int(day))
    return release.strftime("%A")

## one movie does not have a release date associated with it (index 4553) ##
allReleaseDates = mergedDataFrame.release_date
releaseYears = []
releaseMonths = []
releaseDayOfWeek = []
probindex = 0
for date in allReleaseDates:
    if type(date) != str:
       continue
    probindex += 1
    date = date.split("-")
    releaseDayOfWeek.append(findDayOfWeek(date[0], date[1], date[2]))
    releaseYears.append(date[0])
    releaseMonths.append(date[1])

releaseYearsCount = {}
for elem in releaseYears:
    releaseYearsCount.setdefault(elem, 0)
    releaseYearsCount[elem] += 1

allYears = releaseYearsCount.keys()
print(releaseYearsCount)
print(len(releaseYearsCount))
print(sorted(allYears))


Sundays = releaseDayOfWeek.count("Sunday")
Mondays = releaseDayOfWeek.count("Monday")
Tuesdays = releaseDayOfWeek.count("Tuesday")
Wednesdays = releaseDayOfWeek.count("Wednesday")
Thursdays = releaseDayOfWeek.count("Thursday")
Fridays = releaseDayOfWeek.count("Friday")
Saturdays = releaseDayOfWeek.count("Saturday")


Januarys = releaseMonths.count("01")
Februarys = releaseMonths.count("02")
Marchs = releaseMonths.count("03")
Aprils = releaseMonths.count("04")
Mays = releaseMonths.count("05")
Junes = releaseMonths.count("06")
Julys = releaseMonths.count("07")
Augusts = releaseMonths.count("08")
Septemebers = releaseMonths.count("09")
Octobers = releaseMonths.count("10")
Novembers = releaseMonths.count("11")
Decembers = releaseMonths.count("12")

## TODO count up each for  and year use dicts ##


# In[35]:


mergedDataFrame.genres


# In[36]:


## plotting the count of movies for days of the week
fig_dow = plt.figure(figsize=(10,6))
ax = fig_dow.add_axes([0,0,1,1])
days = ['Sunday', 'Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
movieAmountDays= [Sundays, Mondays, Tuesdays, Wednesdays, Thursdays, Fridays, Saturdays]
ax.bar(days, movieAmountDays)
plt.xlabel('Days of the Week')
plt.ylabel('Number of Movies Released')
plt.show()


# In[37]:


## plotting the movie count for months of year
fig_months = plt.figure(figsize=(8,4))
ax = fig_months.add_axes([0,0,1,1])
months = ['January','February','March','April','May','June','July','August','September','October','November','December']
movieAmountMonths = [Januarys,Februarys,Marchs,Aprils,Mays,Junes,Julys,Augusts,Septemebers,Octobers,Novembers,Decembers]
ax.bar(months, movieAmountMonths)
plt.xticks(fontsize=9, rotation = 'vertical')
plt.xlabel('Months of the year')
plt.ylabel('Number of Movies released')
plt.show()


# In[38]:


## plotting number of movies released over the years(1916-2017)
fig_years = plt.figure(figsize=(8,4))
ax = fig_years.add_axes([0,0,1,1])
years = sorted(allYears)
movieAmountYears = []
sortedAllYears = sorted(allYears)
for i in sortedAllYears:
    movieAmountYears.append(releaseYearsCount.get(i))
ax.bar(years, movieAmountYears)
plt.xlabel('Number of Movies Released Over The Years')
plt.xticks("")
plt.show()


# ## Looking for trend in genre

# In[39]:


## Split into 4 intervals of ~25 years each to analyze genre shifts ## 

def getYear(date):
    if type(date) != str:
        return -1
    date = date.split("-")
    return int(date[0])

    
mergedDataFrame['releaseYear'] = mergedDataFrame['release_date'].apply(getYear)

intervalOne = mergedDataFrame[mergedDataFrame['releaseYear'] <= 1942] 
intervalTwo = mergedDataFrame[(mergedDataFrame['releaseYear'] > 1942) & (mergedDataFrame['releaseYear'] <= 1967)] 
intervalThree = mergedDataFrame[(mergedDataFrame['releaseYear'] > 1967) & (mergedDataFrame['releaseYear'] <= 1992)] 
intervalFour = mergedDataFrame[(mergedDataFrame['releaseYear'] > 1992) & (mergedDataFrame['releaseYear'] <= 2017)] 

intervalOneGenres = intervalOne.genres
intervalOneGenresCount = {}
for a, genres in intervalOneGenres.items():
    for genre in genres:
        intervalOneGenresCount.setdefault(genre, 0)
        intervalOneGenresCount[genre] += 1

intervalTwoGenres = intervalTwo.genres
intervalTwoGenresCount = {}
for a, genres in intervalTwoGenres.items():
    for genre in genres:
        intervalTwoGenresCount.setdefault(genre, 0)
        intervalTwoGenresCount[genre] += 1

intervalThreeGenres = intervalThree.genres
intervalThreeGenresCount = {}
for a, genres in intervalThreeGenres.items():
    for genre in genres:
        intervalThreeGenresCount.setdefault(genre, 0)
        intervalThreeGenresCount[genre] += 1

intervalFourGenres = intervalFour.genres
intervalFourGenresCount = {}
for a, genres in intervalFourGenres.items():
    for genre in genres:
        intervalFourGenresCount.setdefault(genre, 0)
        intervalFourGenresCount[genre] += 1




# In[40]:


# getting the keys for each seperate intervals
intervalFourKey = list(intervalFourGenresCount.keys())
intervalThreeKey = list(intervalThreeGenresCount.keys())
intervalTwoKey = list(intervalTwoGenresCount.keys())
intervalOneKey = list(intervalOneGenresCount.keys())


# In[41]:


fig_intervalOne = plt.figure()
ax = fig_intervalOne.add_axes([0,0,1,1])
genrelist1 = []
int1genreAmount = []
for i in intervalOneKey:
    int1genreAmount.append(intervalOneGenresCount.get(i))
    genrelist1.append(i)
plt.xlabel('Movie Genre before 1943')
ax.barh (genrelist1, int1genreAmount)
plt.show()


# In[42]:


fig_intervalTwo = plt.figure()
ax = fig_intervalTwo.add_axes([0,0,1,1])
genrelist2 = []
int2genreAmount = []
for i in intervalTwoKey:
    int2genreAmount.append(intervalTwoGenresCount.get(i))
    genrelist2.append(i)

ax.barh (genrelist2, int2genreAmount)
plt.xlabel('Movie Genre between 1943 and 1967')
plt.show()


# In[43]:


fig_intervalThree = plt.figure()
ax = fig_intervalThree.add_axes([0,0,1,1])
genrelist3 = []
int3genreAmount = []
for i in intervalThreeKey:
    int3genreAmount.append(intervalThreeGenresCount.get(i))
    genrelist3.append(i)

ax.barh (genrelist3, int3genreAmount)
plt.xlabel('Movie Genre between 1968 and 1992')
plt.show()


# In[44]:


fig_intervalFour = plt.figure()
ax = fig_intervalFour.add_axes([0,0,1,1])
genrelist4 = []
int4genreAmount = []
for i in intervalFourKey:
    int4genreAmount.append(intervalFourGenresCount.get(i))
    genrelist4.append(i)

ax.barh (genrelist4, int4genreAmount)
plt.xlabel('Movie Genre between 1993 and 2017')
plt.show()


# In[45]:


## Modeling Section ## 

## Analyzing correlation coefficients among all vars helps highlight what will make the best models ##

mergedDataFrame.corr()


# In[46]:





# In[47]:





# In[48]:





# In[49]:





# In[50]:





# In[51]:





# # Splitting Data and Modeling

# In[59]:



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge


modelingDataFrame = mergedDataFrame[['budget','popularity', 'vote_count', 'revenue']].copy()

## Since such a small amount of data is given, use 15% data for testing ##
training, testing = train_test_split(modelingDataFrame, test_size=.15)
trueVals = testing['revenue'].copy()
testing.drop(columns="revenue", inplace=True)

trainingFeats = training[['budget', 'popularity', 'vote_count']].copy()
trainingTargets = training[['revenue']].copy()

## https://m.youtube.com/watch?v=PIRQY6xmNZY --> for validation ##

## https://stackoverflow.com/questions/63333999/residual-standard-error-of-a-regression-in-python --> for RSE ##
def RSE(predicted, true):
    true = np.array(true)
    predicted = np.array(predicted)
    RSS = np.sum(np.square(true - predicted))
    return np.sqrt(RSS / (len(true) - 2))


LinReg = LinearRegression(normalize=True)
LinReg.fit(trainingFeats, trainingTargets)
print("LinReg Score, ", LinReg.score(trainingFeats, trainingTargets))
linRegPredictions = LinReg.predict(testing)
a = np.array(trueVals)
b = np.array(linRegPredictions)
print("LinReg RSE ", RSE(b,a))
print("LinReg % error ", mean_absolute_percentage_error(a,b))




lasso_reg = Lasso(alpha = .5)
lasso_reg.fit(trainingFeats, trainingTargets)
lasso_pred = lasso_reg.predict(testing)
print("Lasso ", lasso_reg.score(trainingFeats, trainingTargets))
print("RSE Lasso", RSE(lasso_pred, trueVals))
print("Lasso % error", mean_absolute_percentage_error(trueVals, lasso_pred))

ridge_reg = Ridge()
ridge_reg.fit(trainingFeats, trainingTargets)
ridge_pred = ridge_reg.predict(testing)
print("Ridge ", ridge_reg.score(trainingFeats, trainingTargets))
print("RSE Ridge ", RSE(ridge_pred, trueVals))
print("Ridge % error ", mean_absolute_percentage_error(trueVals, ridge_pred))


# In[80]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
# cv values indicated how many times to cross validate (use it on our best performing model based on RSE)
trainingValidate = modelingDataFrame[['budget','popularity', 'vote_count']].copy()
testValidate = modelingDataFrame[['revenue']].copy()

lasso_reg_valid = Lasso(alpha = .01)
cross_lasso = cross_validate(lasso_reg, trainingValidate, testValidate, cv=2)
print("After cross validation lasso: ",cross_lasso['test_score'].mean())

cross_LinReg = cross_validate(LinReg, trainingValidate, testValidate, cv=3)
print("After cross validation Linear Regression: ", cross_LinReg['test_score'].mean())

cross_ridge = cross_validate(ridge_reg, trainingValidate, testValidate, cv=4)
print("After cross validation Ridge regression: ", cross_ridge['test_score'].mean())



#print(cross_val_score(lasso_reg_valid, trainingValidate, testValidate, cv=10, scoring='accuracy').mean())


# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=bf8fb06f-cc38-44a5-a714-1815fdac330d' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
