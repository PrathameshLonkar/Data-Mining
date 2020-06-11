#!/usr/bin/env python
# coding: utf-8

# In[341]:


import os
import numpy as np 
import pandas as pd 
from numpy.random import seed
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense
from keras.models import Model


# In[342]:


d = pd.read_csv("water-treatment.data",header=None,sep=",")
d1=pd.DataFrame(d)
count=0
result=[]

for index,column in d1.iteritems():
    
    if index==0:
        continue
    else:
        arr=[]
        count=0
        sum=0
        for(j,val) in enumerate(column):
            
            if val=="?":
                continue
            else:
                count=count+1
                value=float(val)
                sum=sum+value;
            
        mean=sum/count
        print("Mean is: ",mean)
        d1[index].replace('?',mean,inplace=True)
print(d1)


# In[343]:


d4=pd.DataFrame(d1)
del d1[0]
d2= d1.astype('float64')
d3=(d2-d2.min())/(d2.max()-d2.min())
print(d2)
X_train, X_test = train_test_split(d3, test_size=0.2)


# In[344]:


train_scaled = minmax_scale(X_train, axis = 0)
test_scaled = minmax_scale(X_test, axis = 0)


# In[345]:


ncol = train_scaled.shape[1]
train_scaled.shape[1]


# In[346]:


encoding_dim = 25


# In[350]:


input_dim = Input(shape = (38, ))

# Encoder Layers
enco1 = Dense(25, activation = 'relu')(input_dim)
enco2 = Dense(27, activation = 'relu')(enco1)
enco3 = Dense(25, activation = 'relu')(enco2)
enco4 = Dense(22, activation = 'relu')(enco3)
enco5 = Dense(20, activation = 'relu')(enco4)
enco6 = Dense(17, activation = 'relu')(enco5)
enco7 = Dense(15, activation = 'relu')(enco6)
enco8 = Dense(encoding_dim, activation = 'relu')(enco7)


# Decoder Layers
deco1 = Dense(25, activation = 'relu')(enco8)
deco2 = Dense(50, activation = 'relu')(deco1)
deco3 = Dense(75, activation = 'relu')(deco2)
deco4 = Dense(10, activation = 'relu')(deco3)
deco5 = Dense(12, activation = 'relu')(deco4)
deco6 = Dense(15, activation = 'relu')(deco5)
deco7 = Dense(17, activation = 'relu')(deco6)
deco8 = Dense(38, activation = 'sigmoid')(deco7)
encoder = Model(input_dim, enco1)
# Combine Encoder and Deocder layers
autoencoder = Model(inputs = input_dim, outputs = deco8)

# Compile the Model
autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')


# In[351]:


autoencoder.summary()


# In[353]:


autoencoder.fit(d3,d3,epochs=5,batch_size=200,shuffle=True)


# In[354]:


encoded_data = encoder.predict(train_scaled)
encoded_data.shape


# In[ ]:




