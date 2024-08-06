#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[3]:


df=pd.read_csv('D:\Downloads/mail_data.csv')


# In[4]:


df


# In[5]:


data=df.where((pd.notnull(df)), '')


# In[6]:


data.head(10)


# In[7]:


data.info()


# In[8]:


data.shape


# In[26]:


data.loc[data['Category'] == 'spam', 'Category',]=0
data.loc[data['Category'] == 'ham', 'Category',]=1


# In[27]:


X = data['Message']

Y = data['Category']


# In[28]:


X


# In[29]:


Y


# In[31]:


print(X.shape)
print(X_train.shape)
print(X_test.shape)


# In[32]:


feature_extraction = TfidfVectorizer(min_df = 1, stop_words = 'english', lowercase=True)


X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


# In[33]:


X_train


# In[34]:


print(X_train_features)


# In[35]:


model=LogisticRegression()


# In[36]:


model.fit(X_train_features,Y_train)


# In[37]:


prediction_on_training_data=model.predict(X_train_features)
accuracy_on_training_data=accuracy_score(Y_train,prediction_on_training_data)


# In[38]:


print("Accuracy on training data :",accuracy_on_training_data)


# In[39]:


prediction_on_test_data=model.predict(X_test_features)
accuracy_on_test_data=accuracy_score(Y_test,prediction_on_test_data)


# In[40]:


print("Accuracy on test data :",accuracy_on_test_data)


# In[ ]:


def plot_word_cloud(data, typ):
    email_corpus = " ".join(data['text'])
 
    plt.figure(figsize=(7, 7))
 
    wc = WordCloud(background_color='black',
                   max_words=100,
                   width=800,
                   height=400,
                   collocations=False).generate(email_corpus)
 
    plt.imshow(wc, interpolation='bilinear')
    plt.title(f'WordCloud for {typ} emails', fontsize=15)
    plt.axis('off')
    plt.show()
 
plot_word_cloud(balanced_data[balanced_data['spam'] == 0], typ='Non-Spam')
plot_word_cloud(balanced_data[balanced_data['spam'] == 1], typ='Spam')


# In[42]:


input_your_mail=["Even my brother is not like to speak with me. They treat me like aids patent."]
input_data_features=feature_extraction.transform(input_your_mail)

prediction=model.predict(input_data_features)

print(prediction)

if(prediction[0]==1):
    print("Ham Mail")
else:
    print("Spam mail")


# In[ ]:




