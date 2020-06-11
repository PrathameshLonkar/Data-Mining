#!/usr/bin/env python
# coding: utf-8

# In[305]:


import numpy as np
import pandas as pd
from sklearn import decomposition, datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn import metrics 
from scipy.spatial.distance import cdist


# In[306]:


def transform(vector,df2,index):
    arr=[]
    count=0
    sum=0
    for(j,val) in enumerate(vector):
        #print("in",j,val)
        if val=="?":
            continue
        else:
            count=count+1
            value=float(val)
            sum=sum+value;
            
    mean=sum/count
    print("Mean is: ",mean)
    df2[index].replace('?',mean,inplace=True)
    #print(df2)
    return arr


# In[307]:


df = pd.read_csv("C:/Users/DELL/Desktop/DM/Assignment 2/assignment2/water-treatment.data",header=None,sep=",")
df1=pd.DataFrame(df)
count=0
result=[]
for index,column in df1.iteritems():
    #print("out",index,column)
    if index==0:
        continue
    else:
        arr1=transform(column,df1,index)
        result.append(arr1)

#result   
print(df1)


# In[308]:


df4=pd.DataFrame(df1)
del df4[0]
df4= df4.astype('float64')
df3=(df4-df4.min())/(df4.max()-df4.min())
print(df3)


# In[309]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

sil_result = []
kmax = 10

for k in range(2, kmax+1):
  kmeans = KMeans(n_clusters = k).fit(df3)
  labels = kmeans.labels_
  sil_result.append(silhouette_score(df3, labels, metric = 'euclidean',random_state=None))


# In[310]:


distortions=[]

for k in range(2,10): 
     
    kmeanModel = KMeans(n_clusters=k).fit(df3) 
    kmeanModel.fit(df3)     
    
    distortions.append(sum(np.min(cdist(df3, kmeanModel.cluster_centers_, 
                      'euclidean'),axis=1)) / df3.shape[0]) 
    
plt.plot(range(2,10), distortions, 'bx-') 
plt.xlabel('Values of K') 
plt.ylabel('Distortion') 
plt.title('The Elbow Method using Distortion') 
plt.show() 
    


# In[311]:


plt.plot(range(2,11),sil_result)
print(sil_result)


# In[312]:


kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=100, n_init=10, random_state=0)
predicted1 = kmeans.fit(df3)
predicted_result = predicted1.predict(df3)
df5= kmeans.transform(df3)
#plt.scatter(pred_y[0],pred_y[1])
vals = df3.values
#ax1 = df3.plot.scatter(x=1,y=1,c='DarkBlue')
#df3.plot.scatter(x=mini,y=mini,c='DarkBlue')
#plt.scatter(pred_y[0],pred_y[1])
#plt.scatter(df5[:, 0],df5[:, 1])
plt.scatter(df3[1],df3[2], c= kmeans.labels_, s=10, alpha=0.5)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=10, c='red')
plt.show()

for (index,value) in enumerate(predicted_result):
    print(index+1,":",value+1)
file = open('C:/Users/DELL/Desktop/DM/Assignment 2/kmeans_output.ascii', 'w')
for i in range(0,527):
    file.write("%d %d \n" %(i+1,predicted_result[i]+1))
file.close()


# In[313]:


pca = PCA().fit(df3)
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)')
plt.title('urban waste water treatment plant Dataset')
plt.show()


# In[314]:


explained_variance = pca.explained_variance_


# In[315]:


explained_variance
Total_variance=0
for index,value in enumerate(explained_variance):
    Total_variance+=value

print(Total_variance*100,"%")


# In[316]:


explained_variance


# In[317]:


pca = PCA(n_components=25)
dataset = pca.fit_transform(df3)
print(dataset)
data =pd.DataFrame(dataset)


# In[318]:


kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=100, n_init=10, random_state=0)
predicted_values = kmeans.fit_predict(dataset)
#plt.scatter(pred_y[0],pred_y[1])
#vals = dataset.values
#ax1 = df3.plot.scatter(x=1,y=1,c='DarkBlue')
#df3.plot.scatter(x=mini,y=mini,c='DarkBlue')
#plt.scatter(pred_y[0],pred_y[1])
plt.scatter(data[0],data[1], c= kmeans.labels_, s=10, alpha=0.5)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=10, c='red')
plt.show()
for (index,value) in enumerate(predicted_values):
    print(index+1,":",value+1)
    
f = open('C:/Users/DELL/Desktop/DM/Assignment 2/pca_output.ascii', 'w')
for i in range(0,527):
    f.write("%d %d \n" %(i+1,predicted_values[i]+1))
f.close()


# In[ ]:




