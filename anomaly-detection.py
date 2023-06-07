#!/usr/bin/env python
# coding: utf-8

# # Anomaly Detection (Outlier Detection)

# This notebook contains 3 effective outlier detection techniques explained and applied to a UCI Machine Learning Repository's [Wholesale Customers dataset](https://archive.ics.uci.edu/ml/datasets/wholesale+customers)

# ### [Sklearn's Isolation Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html) 

# Returns the anomaly score of each sample using the IsolationForest algorithm
# 
# The IsolationForest ‘isolates’ observations by randomly selecting a feature and then randomly selecting a split value between the maximum and minimum values of the selected feature.
# 
# Since recursive partitioning can be represented by a tree structure, the number of splittings required to isolate a sample is equivalent to the path length from the root node to the terminating node.
# 
# This path length, averaged over a forest of such random trees, is a measure of normality and our decision function.
# 
# Random partitioning produces noticeably shorter paths for anomalies. Hence, when a forest of random trees collectively produce shorter path lengths for particular samples, they are highly likely to be anomalies.

# In[13]:


import pandas as pd
df = pd.read_csv('Wholesale Customers data.csv')


# In[14]:


df.head()


# In[19]:


import matplotlib.pyplot as plt
plt.scatter(df.iloc[:,3], df.iloc[:,5])


# In[20]:


from sklearn.ensemble import IsolationForest


# In[21]:


clf = IsolationForest(contamination = 0.2)
clf.fit(df)
predictions = clf.predict(df)


# In[22]:


predictions


# The predictions marked as '-1' are outliers.

# In[25]:


import numpy as np
index = np.where(predictions < 0)
index


# In[26]:


x = df.values


# In[27]:


plt.scatter(df.iloc[:,2], df.iloc[:,4])
plt.scatter(x[index,2], x[index,4], edgecolors = "r")


# ### [DBSCAN Clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)

# Perform DBSCAN clustering from vector array or distance matrix.
# 
# DBSCAN - Density-Based Spatial Clustering of Applications with Noise. Finds core samples of high density and expands clusters from them. Good for data which contains clusters of similar density.

# In[1]:


# DBSCAN
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


X, y = make_circles(n_samples = 750, factor = 0.3, noise = 0.1)


# In[4]:


X


# In[5]:


plt.scatter(X[:,0], X[:,1], c=y)


# In[6]:


from sklearn.cluster import DBSCAN


# In[7]:


dbscan = DBSCAN(eps = 0.1)


# In[8]:


dbscan.fit_predict(X)


# In[9]:


dbscan.labels_


# In[10]:


plt.scatter(X[:,0],X[:,1], c= dbscan.labels_)


# In[12]:


plt.scatter(X[:,0],X[:,1], c = y)


# 
# ### [Local Outlier Factor Anomaly Detection](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html)

# Unsupervised Outlier Detection using the Local Outlier Factor (LOF).
# 
# The anomaly score of each sample is called the Local Outlier Factor. It measures the local deviation of the density of a given sample with respect to its neighbors. It is local in that the anomaly score depends on how isolated the object is with respect to the surrounding neighborhood. More precisely, locality is given by k-nearest neighbors, whose distance is used to estimate the local density. By comparing the local density of a sample to the local densities of its neighbors, one can identify samples that have a substantially lower density than their neighbors. These are considered outliers.

# In[28]:


from sklearn.neighbors import LocalOutlierFactor


# In[30]:


local = LocalOutlierFactor(n_neighbors = 3)


# In[42]:


predictions = local.fit_predict(df)


# In[43]:


predictions


# In[44]:


index = np.where(predictions < 0)
index


# In[48]:


x = df.values


# In[49]:


plt.scatter(df.iloc[:,2], df.iloc[:,4])
# plt.scatter(x[index,2], x[index, 4])
plt.scatter(x[index,2], x[index,4], edgecolors = "r")


# In[ ]:




