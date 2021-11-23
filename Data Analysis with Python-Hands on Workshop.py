#!/usr/bin/env python
# coding: utf-8

# # Data Analysis With Python 
# 
# Data analysis is a process of inspecting, cleansing, transforming and modeling data with the goal of discovering useful information, informing conclusions and supporting decision-making. Data analysis has multiple facets and approaches, encompassing diverse techniques under a variety of names, and is used in different business, science, and social science domains. In today's business world, data analysis plays a role in making decisions more scientific and helping businesses operate more effectively.
# 
#       -- Pandas Fundamentals
#       -- Data Visualization using data viz libraries
# 

# In[2]:


#import all the libraries 
#Pandas :  for data analysis
#numpy : for Scientific Computing.
#matplotlib and seaborn : for data visualization
#scikit-learn : ML library for classical ML algorithms
#math :for mathematical functions


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_palette('husl')
import warnings
import math
get_ipython().run_line_magic('matplotlib', 'inline')











'''
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
#from lightgbm import LGBMClassifier
from sklearn.metrics import  accuracy_score
import warnings
warnings.filterwarnings('ignore')
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error,r2_score

import xgboost as xgb
#import lightgbm as  lgb
from xgboost.sklearn import XGBClassifier
from catboost import CatBoostClassifier

from sklearn.preprocessing import StandardScaler, LabelBinarizer
# auxiliary function
from sklearn.preprocessing import LabelEncoder

'''


# In[3]:


#Read data from csv
iris_data= pd.read_csv('/Users/priyeshkucchu/Desktop/IRIS 2.csv',engine='python')


# In[37]:


#Show top 5 rows

iris_data.head(5)
iris_data.tail(10)


# In[5]:


#Get detailed information of data 
#checking if there is any inconsistency in the dataset
#as we see there are no null values in the dataset, so the data can be processed
iris_data.info()


# In[8]:


#No of columns in the data
#0=rows, 1=columns
iris_data.shape[1]


# In[9]:


#No of rows in the data

iris_data.shape[0]


# In[41]:


iris_data.shape


# In[10]:


iris_data.species.unique()


# In[11]:


iris_data.tail(5)


# In[14]:


iris_data.count(axis=0)


# In[15]:


iris_data.isnull()


# In[42]:


iris_data.describe()


# In[43]:


iris_data["species"].value_counts()


# In[16]:


iris_data.dropna()


# In[23]:


def missing_values(x):
    return sum(x.isnull())

print("Missing values in each column:")
print(iris_data.apply(missing_values,axis=0))


# In[24]:


type(iris_data)


# # Complete Data Visualization

# In[201]:


import matplotlib.pyplot as plt
import seaborn as sns 

#Read csv

iris_data= pd.read_csv('/Users/priyeshkucchu/Desktop/IRIS 2.csv',engine='python')




#gives us a General Idea about the dataset.

iris_data.describe().plot(kind="area",fontsize=20,figsize=(20,8),table=False,colormap="rainbow")

plt.xlabel("Statistics")
plt.ylabel('Value')
plt.title("Statistics of IRIS dataset")


# In[103]:


#Here the frequency of the observation is plotted.
#In this case we are plotting the frequency of the three species in the Iris Dataset
#Count Plot shows the number of occurrences of an item based on a certain type of category.

sns.countplot("species",data=iris_data)
#plt.show()


# In[104]:


'''
Then you can use the method plt.pie() to create a plot. The slices will be ordered and plotted counter-clockwise.
We can see that there are 50 samples each of all the Iris Species in the data set.

'''
iris_data["species"].value_counts().plot.pie(explode=[0.1,0.1,0.1], autopct='%1.1f%%',shadow=True, figsize=(10,8))
plt.show()
#We can see that there are 50 samples each of all the Iris Species in the data set.


# In[105]:


#Jointplot is seaborn library specific and can be used to quickly visualize and analyze the relationship between 
#two variables and describe their individual distributions on the same plot.
#You can draw a plot of two variables with bivariate and univariate graphs.

#Draw a scatterplot with marginal histograms
figure=sns.jointplot(x="sepal_length",y="sepal_width",data=iris_data,color="orange")


# In[106]:


#Replace the scatterplots and histograms with density estimates 

figure=sns.jointplot(x="sepal_length",y="sepal_width",data=iris_data,color="green",kind="kde")


# In[107]:


# Add regression 

figure=sns.jointplot(x="sepal_length",y="sepal_width",data=iris_data,color="blue",kind="reg")


# In[108]:


# Replace the scatterplot with a joint histogram using hexagonal bins

figure=sns.jointplot(x="sepal_length",y="sepal_width",data=iris_data,color="pink",kind="hex")


# In[109]:


#Draw a scatterplot, then add a joint density estimate

figure=(sns.jointplot(x="sepal_length",y="sepal_width",data=iris_data,color="purple").plot_joint(sns.kdeplot,zorder=0,n_levels=6))


# In[202]:


#Facetgrid : Multi-plot grid for plotting conditional relationships.
#It is used as a Multi-plot grid for plotting conditional relationships.

sns.FacetGrid(iris_data,hue="species",height=7).map(plt.scatter,"sepal_length","sepal_width").add_legend()


# In[111]:


#Boxplot : give a statical summary of the features being plotted.Top line represent the max value,top edge 
#of box is third Quartile, middle edge represents the median,bottom edge represents the first quartile value.
#The bottom most line respresent the minimum value of the feature.The height of the box 
#is called as Interquartile range.The black dots on the plot represent the outlier values in the data.

fig=plt.gcf()
fig.set_size_inches(11,8)
fig=sns.boxplot(x="species", y="petal_width", data=iris_data, hue="species",                order=["Iris-versicolor", "Iris-setosa",,"Iris-virginica"],                linewidth=2.5,orient='v',dodge=False )


# In[112]:


#Draw a categorical scatterplot with non-overlapping points.
'''
The Swarm plot is used whenever you want to draw a categorical scatterplot with non-overlapping points. 
This gives a better representation of the distribution of values, but it does not scale well to large 
numbers of observations. 
This style of the plot is sometimes called a “beeswarm”.
'''
sns.swarmplot(x="species", y="petal_width", data=iris_data, color=".25")


# In[113]:


#Draw boxplot by species

iris_data.boxplot(by="species",figsize=(10,8))


# In[114]:


#Strip Plot : Draw a scatterplot where one variable is categorical.

#A strip plot can be drawn on its own, but it is also a good complement to 
#a box or violin plot in cases where you want to show all 
#observations along with some representation of the underlying distribution.

fig=plt.gcf()
fig.set_size_inches(11,8)
fig=sns.stripplot(x="species",y="petal_width",data=iris_data,color="blue",hue="species",order=["Iris-setosa",                "Iris-versicolor","Iris-virginica"],jitter=True,edgecolor="black",linewidth=1,size=6,orient='v'                ,palette="Set2")


# In[115]:


#Combine Stripplot and boxplot

fig=plt.gcf()
fig.set_size_inches(11,8)
fig=sns.boxplot(x="species",y="petal_width",data=iris_data)
fig=sns.stripplot(x="species",y="petal_width",data=iris_data,jitter=True, edgecolor="black",hue="species"                 ,linewidth=1.0)


# In[116]:


#Violin Plot It is used to visualize the distribution of data and its probability distribution.
#This chart is a combination of a Box Plot and a Density Plot that is rotated and placed on
#each side, to show the distribution shape of the data. The thick black bar in the centre 
#represents the interquartile range, the thin black line extended from it represents the
#95% confidence intervals, and the white dot is the median.Box Plots are limited in their display of the data, as 
#their visual simplicity tends to hide significant details about how values in the data are distributed.

fig=plt.gcf()
fig.set_size_inches(11,8)
fig=sns.violinplot(x="species",y="petal_width",data=iris_data,hue="species",saturation=0.8,palette="Set3")


# In[117]:


#plot subplot for different columns in the data set

plt.figure(figsize=(20,10))
plt.subplot(2,2,1)
sns.violinplot(x="species",y="sepal_length",data=iris_data,hue="species",saturation=0.8,palette="summer")
plt.subplot(2,2,2)
sns.violinplot(x="species",y="sepal_width",data=iris_data,hue="species",saturation=0.8,palette="summer")
plt.subplot(2,2,3)
sns.violinplot(x="species",y="petal_length",data=iris_data,hue="species",saturation=0.8,palette="summer")
plt.subplot(2,2,4)
sns.violinplot(x="species",y="petal_width",data=iris_data,hue="species",saturation=0.8,palette="summer")


