#!/usr/bin/env python
# coding: utf-8

# # Task 2 - Prediction using Unsupervised Machine Learning 
# 
# ## Author - Amal Madhu 
# ## The Sparks Foundation GRIP Internship

# In[14]:


#Impoting libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets 
import seabon as sns 


# In[7]:


#loading the dataset 

iris = datasets.load_iris()
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)

#printing the first 5 rows
iris_df.head()


# In[8]:


iris_df.describe()


# In[11]:


# checking for missing values 

iris_df.isnull().sum()


# In[15]:


#Visualizing using pair plot

sns.pairplot(iris_df)


# In[10]:


#Fininding the optimum number of clusters for k-means classification

x = iris_df.iloc[:, [0, 1, 2, 3]].values

from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# In[13]:


#Plotting the result

plt.plot(range(1,11), wcss)
plt.title("The elbow method")
plt.xlabel("Number of cluster")
plt.ylabel("WCSS")
plt.show()


# In[ ]:


#You can clearly see why it is called 'The elbow method' from the above graph, the optimum clusters is where the elbow occurs. This is when the within cluster sum of squares (WCSS) doesn't decrease significantly with every iteration. From this we choose the number of clusters as ** '3**'.


# In[16]:


#applying kmeans to the dataset, creating the kmeans classifier 

kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# In[ ]:





# In[26]:


#Visualising the clusters- On the first two columns

plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0,1],
           s = 100, c = 'red', label = "Iris-setosa")
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1,1],
           s = 100, c = 'blue', label = "Iris-versicolour")
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2,1],
           s = 100, c = 'green', label = "Irid-virginica")
plt.legend()


# In[27]:


#Plotting with centroids of the clusters

plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0,1],
           s = 100, c = 'red', label = "Iris-setosa")
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1,1],
           s = 100, c = 'blue', label = "Iris-versicolour")
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2,1],
           s = 100, c = 'green', label = "Irid-virginica")

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
           s = 100, c = "yellow", label = 'Centroids')
plt.legend()


# In[ ]:


#Successfully completed the task 

