#!/usr/bin/env python
# coding: utf-8

# # Fantasy Premier League Player Positions

# ### This project consists of looking into a game that is played by over 8.3 million people around the world and ways in which performance on a weekly basis can be maximised allowing people to attain the highest points possible. 

# ## How Does FPL work?
# 
# Fantasy Premier League is an online game where you are casted the role of a premier league manager. As a dedicated player of this game you must pick a squad of 15 players from the 20/21 premier league season which starts in August and ends in May. 
# 
# These players you have selected will then score points based on their performance in real-life live games. Each player depending on their position, contributions, impact will score varios points. Prices are given to players based on their form. Players performing well will rise in price and those that haven't performed will fall in value. 
# 
# - You have a starting budget of £100.0m to spend on a 15 man squad.
# - There are 38 Gameweeks (GW) in total.
# 
# 
# 

# ## How are points awarded?
# 
# 

# ## Other Rules & Important Information
# 
# ### Squad Size
# Your squad consists of 15 players:
# - 2 Goalkeepers
# - 5 Defenders
# - 5 Midfielders
# - 3 Forwards
# 
# ### Budget 
# The total value of an intitial squad should not exceed £100.0m.
# 
# ### Players Per Team
# **You are allowed to select only 3 players from a single Premier League team.** 

# # Aim
# 
# The aim of this project is to identify which players are out-of-posistion from the initial position they were allocated by the developers of the game. For example, those who watch football would know which player plays where on the pitch through eye-test without the need for data. 
# 
# A goalkeeper is classified by clean sheets, saves, penalty saves.
# A defender could be classified by clean sheets, tackles, interceptions, blocks, clearances and more.
# A midfielder could be classified by chances created, threat, assists, passes, interceptions, tackles and more.
# A forward could be classified by goals, assists and more.
# 
# However, taking these attributes into consideration has a player that was deemed for example a traditional midfielder, actually playing like a forward? Is a midfielder based off their statistics over the season actaully playing like a defender?
# 
# Correctly analysing these players will allow FPL players to make the right selections to maximise their GW points and rank higher in the leaderboards.
# 
# - Example: A midfielder returns more points for a goal than a forward. If this midfielder is predicted as a forward, fpl players can pick this player to get better returns
# 
# 

# # Methodology
# 
# In order to solve this problem, this project uses machine learning to train and then predict the final model with correct positions. This will consist of importing relevant packages, cleaning the dataset, data exploration- creating boxplots and charts to gain a general understanding of the statistics. The following part will involves comparing and contrasting various machine learning models. Once a model has been picked based off accuracy, the last step would be to implement this on a new dataset.  

# # 1. 

# In[36]:


import requests
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
 
# scipy
import scipy as sp

# scikit-learn
import sklearn as sk

# import other important functions and algorithms
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import svm
import seaborn as sns


# In[37]:


# Obtaining the relevant dataset from 19-20 premier league season. 

url_2019 = "/Users/jovan/Desktop/Every Player Data 2019-20 3.csv"


# In[38]:


df_key_stats_2019 = pd.read_csv(url_2019) # Getting the key variables that influences a players positions by the model
df_key_stats_2019.keys()


# In[39]:


# setting the correct headings for each section
df_key_stats_2019.columns = ["First_Name","Surname","Team","Pos","Start_Price","End_Price","Total_Points","Points_Per_Game","Transfers_In","Transfers_Out","Season_Value","Minutes","Goals","Assists","Clean_Sheets","Goals_Conceded","Own_Goals","Penalties_Saved","Penalties_Missed","Yellow_Cards","Red_Cards","Saves","Bonus_Points","BPS","Influence","Creativity","Threat","ICT_Index","Influence_Rank","Creativity_Rank","Threat_Rank","ICT_Index_Rank"]
df_key_stats_2019


# In[40]:


# in this project we are only looking at the factors that can influence a players position so can remove the unnecessary columns and pick the ones needed.

df_key_stats_2019 = df_key_stats_2019[["Pos","First_Name","Surname","Start_Price","End_Price","Points_Per_Game","Total_Points","Season_Value","Minutes","Goals","Assists","Clean_Sheets","Own_Goals","Penalties_Saved","Penalties_Missed","Yellow_Cards","Red_Cards","Bonus_Points","Influence","Creativity","Threat"]]
df_key_stats_2019 = df_key_stats_2019.iloc[1:]
df_key_stats_2019


# ***In the output we can see the observations that are needed and we are working with the entire dataset of the game with 666 players and required variables.*** 

# In[41]:


print(df_key_stats_2019.shape) # There are 666 players and 21 variables being used. (666 rows, 21 columns)


# In[42]:


df_key_stats_2019.isnull().values.any() # checking for any null values that amy create an error or impact results. No null values and there is data in all cells.


# In[43]:


dataTypeDict = dict(df_key_stats_2019.dtypes) # gives us what type of data this is
print(dataTypeDict)


# In[44]:


print(df_key_stats_2019.describe()) # descriptive statistics 


# - These statistics provides an overview of the premier league dataset. 

# In[45]:


print(df_key_stats_2019.groupby('Pos').size()) # the output variable 


# In[46]:


print(df_key_stats_2019.groupby('Goals').size()) # the output variable 


# In[47]:


# Boxplot of Total_Points
df_key_stats_2019["Goals"].plot(kind = "box", subplots = False,sharex = False, sharey = False)
plt.ylabel("Threat")
plt.show()


# In[48]:


# Boxplot of Total_Points
df_key_stats_2019["Total_Points"].plot(kind = "box", subplots = False,sharex = False, sharey = False)
plt.ylabel("Points")
plt.show()


# In[49]:


# Boxplot of Minutes
df_key_stats_2019["Minutes"].plot(kind = "box", subplots = False,sharex = False, sharey = False)
plt.ylabel("Minutes")
plt.ylabel("Minutes")
plt.show()


# In[50]:


df_key_stats_2019[["Minutes"]].hist()
plt.show()


# In[51]:


# weed out players who do not play even 600 minutes
df_key_stats_2019.sort_values('Minutes', inplace=True, ascending=True) # sort on minutes
df_key_stats_2019
count = 0
for index, row in df_key_stats_2019.iterrows():
    if(row["Minutes"] < 600):
        count += 1
        continue
    else:
        break
print(count)
df_key_stats_2019 = df_key_stats_2019[285:]


# In[52]:


# better data to work with
df_key_stats_2019[["Minutes"]].hist()
plt.show()


# In[53]:


# Box plot check again
df_key_stats_2019["Total_Points"].plot(kind = "box", subplots = False,sharex = False, sharey = False)
plt.show()


# In[54]:


# Box plot check again
df_key_stats_2019["Minutes"].plot(kind = "box", subplots = False,sharex = False, sharey = False)
plt.show()


# In[55]:


# boxplot of starting price and end price
df_key_stats_2019[["Start_Price","End_Price"]].plot(kind = "box", subplots = False,sharex = False, sharey = False)
plt.show()


# In[56]:


df_key_stats_2019["Goals"].hist()
plt.xlabel("Goals")
plt.ylabel("Count")
plt.title("Histogram of Goals for Players Who Play at Least 600 Minutes (2019-2020)")
plt.show()


# In[57]:


df_key_stats_2019["Assists"].hist()
plt.xlabel("Assists")
plt.ylabel("Count")
plt.title("Histogram of Assists for Players Who Play at Least 600 Minutes (2019-2020)")
plt.show()


# In[58]:


df_key_stats_2019["Total_Points"].hist()
plt.xlabel("Total_Points")
plt.ylabel("Count")
plt.title("Histogram of Total_Points for Players Who Play at Least 600 Minutes (2019-2020)")
plt.show()


# In[59]:


df_key_stats_2019["Threat"].hist()
plt.xlabel("Threat")
plt.ylabel("Count")
plt.title("Histogram of Threat Score for Players Who Play at Least 600 Minutes (2019-2020)")
plt.show()


# In[60]:


# quick scatter matrix view to see the relationship between variables
scatter_matrix(df_key_stats_2019[["Goals","Assists","Influence","Creativity","Threat","Total_Points"]])
plt.xticks(rotation = 90)
plt.show()


# In[61]:


# split-out validation dataset (testing data)
array = df_key_stats_2019.values
X = array[:,4:] # row, columns of the features that I want
y = array[:,0] # output variable: player position
X_train,X_validation,Y_train,Y_validation = train_test_split(X,y,test_size = 0.2, random_state = 1) # random state means seed


# In[62]:


# Algorithms
all_models = []
# liblinear has both L1 and L2 regularization
#     - Ridge Regression and Lasso Regression -> avoid overfitting and feature selection
# ovr means it's a binary problem for each label (one vs rest)
all_models.append(("LogReg",LogisticRegression(solver = "liblinear", multi_class="ovr")))

# maximizes seperation using our chosen features
#     - maximizing distance between means and minimizing variance 
#     - reduces features down 
#     - minimizes the scatter
all_models.append(("LDA",LinearDiscriminantAnalysis()))

# classification algorithm that memorizes observations to classify new data
#     - new data is classified by observing the "nearest neighbours"
all_models.append(("KNN", KNeighborsClassifier()))

# tree-like diagram where each leaf node is the outcome that is used to classify new data
all_models.append(("CART",DecisionTreeClassifier()))

# uses the Gaussian distribution to classify new data
all_models.append(("NB",GaussianNB()))

# uses a threshold that is the midpoint between different classifications
all_models.append(("SVM-g",SVC(gamma = "auto")))

# evaluate each model
results = []
names = []
# go through each model and perform cross validation to compare different machine learning algorithms
#     - kind of like foiling through the n_splits and training and testing for each algorithm
for name, model in all_models:
  kfold = StratifiedKFold(n_splits = 10, random_state = 1, shuffle = True)
  cv_results = cross_val_score(model, X_train,Y_train,cv=kfold, scoring = "accuracy") # evaluate score using cross validation
  results.append(cv_results)
  names.append(name)
  print("%s:%f(%f)"%(name,cv_results.mean(),cv_results.std())) # take the mean and standard deviation


# In[63]:


# compare algorithms
plt.boxplot(results,labels = names)
plt.title("Algorithm Comparison")
plt.show()
# best algorithm is Logistic Regression in terms of the mean


# In[64]:


# best model - Logistic Regression
model = LogisticRegression(solver = "liblinear", multi_class="ovr",random_state = 1) 
model.fit(X_train,Y_train)
predictions = model.predict(X_validation)


# In[65]:


print(accuracy_score(Y_validation,predictions))
print(confusion_matrix(Y_validation,predictions))
print(classification_report(Y_validation,predictions))


# In[66]:


importance=model.coef_[0] # Coefficient of the features
print(model.get_params())
print(importance)
print(df_key_stats_2019.columns[4:])
# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()


# In[67]:


lda = LinearDiscriminantAnalysis()
lda_model = lda.fit(X_train,Y_train)
predictions = lda.predict(X_validation)


# In[68]:


print(accuracy_score(Y_validation,predictions))
print(confusion_matrix(Y_validation,predictions))
print(classification_report(Y_validation,predictions))


# In[69]:


# make predictions decision tree
clf = DecisionTreeClassifier(random_state = 1)
clf.fit(X_train,Y_train)
predictions = clf.predict(X_validation)


# In[70]:


print(accuracy_score(Y_validation,predictions))
print(confusion_matrix(Y_validation,predictions))
print(classification_report(Y_validation,predictions))


# In[71]:


# Find the most important features using Decision Tree
important_features = pd.DataFrame({'feature':df_key_stats_2019.columns[4:],'importance':np.round(clf.feature_importances_,3)})
important_features = important_features.sort_values('importance',ascending=False)
 
f, ax = plt.subplots(figsize=(12, 14))
g = sns.barplot(x='importance', y='feature', data=important_features,
                color="red", saturation=.5, label="Total")
g.set(xlabel='Feature Importance', ylabel='Feature', title='Importance of Features For Position using Decision Tree')
plt.show()


# In[72]:


# Get 2020 to 2021 data and use this model to predict their positions
url_2021 = 'https://fantasy.premierleague.com/api/bootstrap-static/'
req = requests.get(url_2021)
json = req.json()
json.keys()


# In[73]:


df_elements = pd.DataFrame(json['elements'])
df_elements_types = pd.DataFrame(json['element_types'])
df_teams = pd.DataFrame(json['teams'])


# In[74]:


df_elements.columns


# In[75]:


pd.set_option('mode.chained_assignment', None) # ignore warning
df_key_stats_2021 = df_elements[["team","element_type","first_name","second_name","now_cost","points_per_game","total_points","value_season","minutes","goals_scored","assists","clean_sheets","own_goals","penalties_saved","penalties_missed","yellow_cards","red_cards","bonus","influence","creativity","threat"]]
df_key_stats_2021['position'] = df_key_stats_2021.element_type.map(df_elements_types.set_index('id').singular_name) # position of player
df_key_stats_2021['team'] = df_key_stats_2021.team.map(df_teams.set_index('id').name) # team of player
df_key_stats_2021['now_cost'] = df_key_stats_2021['now_cost'].map(lambda x: x/10) # cost of player
df_key_stats_2021['value'] = df_key_stats_2021.apply(lambda row: round(row.total_points/row.now_cost,2), axis = 1) # total points/cost of player
def change_name(x):
    if(x == "Midfielder"):
        return "MID"
    elif(x == "Forward"):
        return "FWD"
    elif(x == "Goalkeeper"):
        return "GKP"
    else:
        return "DEF"
        
df_key_stats_2021["position"] = [change_name(x) for x in df_key_stats_2021["position"]]
df_key_stats_2021


# In[76]:


df_key_stats_2021 = df_key_stats_2021[["position","first_name","second_name","now_cost","points_per_game","total_points","value_season","minutes","goals_scored","assists","clean_sheets","own_goals","penalties_saved","penalties_missed","yellow_cards","red_cards","bonus","influence","creativity","threat"]]
df_key_stats_2021


# In[77]:


df_key_stats_2021.sort_values('minutes', inplace=True, ascending=True) # sort on minutes
df_key_stats_2021 = df_key_stats_2021.reset_index(drop=True)
df_key_stats_2021


# In[78]:


df_key_stats_2021[["minutes"]].hist()


# In[79]:


# remove players who have less than 300 minutes of playing time
count = 0
for index, row in df_key_stats_2021.iterrows():
    if(row["minutes"] < 300):
        count += 1
        continue
    else:
        break
print(count)
df_key_stats_2021 = df_key_stats_2021[302:]
df_key_stats_2021 = df_key_stats_2021.reset_index(drop=True)


# In[80]:


dataTypeDict = dict(df_key_stats_2021.dtypes)
print(dataTypeDict)


# In[81]:


df_key_stats_2021[["influence","creativity","threat"]] = df_key_stats_2021[["influence","creativity","threat"]].apply(lambda col:pd.to_numeric(col, errors='coerce'))


# In[83]:


df_key_stats_2021
dataTypeDict = dict(df_key_stats_2021.dtypes)
print(dataTypeDict)


# In[85]:


df_key_stats_2021[["minutes"]].hist()
plt.xlabel("minutes")
plt.ylabel("count")


# In[86]:


print(df_key_stats_2021.shape) # 339 rows and 20 columns


# In[87]:


df_key_stats_2021.isnull().values.any()


# In[88]:


print(df_key_stats_2021.describe())


# In[89]:


df_key_stats_2021["goals_scored"].hist()
plt.xlabel("Goals")
plt.ylabel("Count")
plt.title("Histogram of Goals for Players Who Play at Least 600 Minutes (2020-2021)")
plt.show()


# In[90]:


df_key_stats_2021["total_points"].hist()
plt.xlabel("Total_Points")
plt.ylabel("Count")
plt.title("Histogram of Total_Points for Players Who Play at Least 600 Minutes (2020-2021)")
plt.show()


# In[91]:


# quick scatter matrix view to see the relationship between variables
scatter_matrix(df_key_stats_2021[["goals_scored","assists","influence","creativity","threat","total_points"]])
plt.xticks(rotation = 90)
plt.show()


# In[92]:


array = df_key_stats_2021.values
new_input_X = array[:,3:]
print(new_input_X)
true = array[:,0] # true position of the player
predictions = model.predict(new_input_X)
print(new_input_X)
print(new_input_X, predictions)


# In[93]:


print(accuracy_score(true, predictions))
print(confusion_matrix(true, predictions))
print(classification_report(true, predictions))


# In[94]:


df_key_stats_2021["predicted"] = predictions
df_key_stats_2021 = df_key_stats_2021[["predicted","position","first_name","second_name","now_cost","points_per_game","total_points","value_season","minutes","goals_scored","assists","clean_sheets","own_goals","penalties_saved","penalties_missed","yellow_cards","red_cards","bonus","influence","creativity","threat"]]
df_key_stats_2021 = df_key_stats_2021.sort_values('total_points', inplace=False, ascending=False)
df_key_stats_2021.head(50)

