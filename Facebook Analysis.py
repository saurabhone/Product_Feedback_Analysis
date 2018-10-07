#!/usr/bin/env python
# coding: utf-8

# # <font color ='Red'> PRODUCT FEEDBACK ANALYSIS </font>
# 

# #### Social Media is playing an important role in marketing of a product. This social media can also be used to know the customer's opinions on improvising the product's features, it's quality and many other things.
# #### Sentimental Analysis can roughly give us an idea about the future demand of the product which can help us in increasing the revenue and cutting down on manufacturing costs.
# #### In the Jupyter Notebook, we will do the sentimental analysis on a car model and for analysis purpose we will take the comments as the data from it's Facebook post.
# #### We will use Python libraries along with Facebook Graph API to achieve our goal.

# In[10]:


from IPython.display import Image
Image("/Users/saurabhkarambalkar/Desktop/Picture.jpg", width=1000)


# # <font color ='Red'>Business Aspects</font>
# <li> Reduce the Manufactuing units and cost of the product
# <li> Improve the quality of  product by taking the feedback from the customers 
# 

# # <font color ='Red'>Assumptions</font>
# <li> The assumptions for the following analysis is that all the reviews provided by the customers are authentic and unbiased

# # <font color = 'Red'> Limitations </font>
# - The API provides limited access to the data due to which the models don't have much data to be trained upon
# - If all the reviews given by the customers turn out to be facts, sentiment analysis could not be carried out on the data 

# ## Facebook

# In[5]:


import pandas as pd
import numpy as np
import facebook


# In[71]:


graph = facebook.GraphAPI(access_token="EAACcYJ4a3rQBAJhxgkoqEy6aNGmj5QdTMLsOxqXpHR2eomKHE0DeYvAf7syzc7HK7KYfYzAOSRZAqxhVuPvXguZALFa8TpWVoyFlyybn0d6asCK34cRZA6aFeC6LyVfTmYkjtbEK6H81bzDWUsBoelImOZBJXzRlhBMauZBOZAg2inW6QIqfTjY21EVQZChJZB8ZD", version="2.6")


# In[105]:


#### Get comments from post
post = graph.get_object(id='10155966298429003', fields='comments')
print(post)


# In[107]:


data = post['comments']['data']
len(data)


# In[119]:


messages=[]
for i in data:
    messages.append(i['message'])
    #messages[].append(i['id'])
messages


# In[240]:


Facebook_Messages=pd.DataFrame()
Facebook_Messages['Messages']=messages
Facebook_Messages


# In[218]:


import json
from pandas.io.json import json_normalize


# In[219]:


df1=pd.DataFrame(messages)
df1


# In[220]:


import textblob as tb
from textblob import TextBlob


# In[221]:


comments=df1[0]
comments


# In[224]:


cm1=comments[8]
blob=TextBlob(cm1)
blob.sentiment


# In[159]:


import matplotlib.pyplot as plot
get_ipython().run_line_magic('matplotlib', 'inline')
polarity=[]
subj=[]
for t in commennts:
    tx=TextBlob(t)
    polarity.append(tx.sentiment.polarity)
    subj.append(tx.sentiment.subjectivity)


# In[160]:


poltweet= pd.DataFrame({'polarity':polarity,'subjectivity':subj})   
poltweet.plot(title='Polarity and Subjectivity')


# In[161]:


list=[]
list= comments
wordstring = list
wordstring


# # Twitter

# In[168]:


import numpy as np
import pandas as pd


# In[169]:


import twitter
from twitter import Twitter
from twitter import OAuth
from twitter import TwitterHTTPError
from twitter import TwitterStream


# In[170]:


ck= 'hq1ikoFrNoXH32vVwH4tdYewd'
cs= 'UpsUIWyTXNtOoKGCuQZw7dQ9LO2lzx5vxnw069g4gg9BGfWl3Z'
at='753638455-fQJkPQDV4aDafISatU7ZjRBjf4UMb2ufYyKSI2bU'   
ats='Cjym1sTksiwAKp7Zeh78ngdqZYUI29r981Dw4eFqMCJ4e'


# In[172]:


oauth= OAuth(at,ats,ck,cs)
twit_api=Twitter(auth=oauth)
t_loc= twit_api.trends.available()
t_loc
ts= TwitterStream(auth=oauth)


# In[173]:


iterator = ts.statuses.filter(track="Volkswagen",language="en")


# In[174]:


b=[]
for t in iterator:
    print(t)
    b.append(t)
    if len(b)==25:
        break


# In[175]:


len(b)


# In[176]:


import json
from pandas.io.json import json_normalize


# In[177]:


df=json_normalize(b)
df.head()


# In[178]:


# Textblob
import textblob as tb
from textblob import TextBlob


# In[179]:


get_ipython().system('python -m textblob.download_corpora')


# In[180]:


tweettext=df['text']


# In[189]:


tx=tweettext[16]
blob=TextBlob(tx)
tx


# In[190]:


blob.sentiment


# In[183]:


# Plot Sentiments
import matplotlib.pyplot as plot
get_ipython().run_line_magic('matplotlib', 'inline')
polarity=[]
subj=[]
for t in tweettext:
    tx=TextBlob(t)
    polarity.append(tx.sentiment.polarity)
    subj.append(tx.sentiment.subjectivity)

    
poltweet= pd.DataFrame({'polarity':polarity,'subjectivity':subj})   
poltweet.plot(title='Polarity and Subjectivity')


# In[191]:


list=[]
list= df['text']
wordstring = list[0]
wordstring


# In[192]:


n=1
while n < 25:
    wordstring += list[n]
    n=n+1


# In[193]:


wordstring
wordlist = wordstring.split()


# In[194]:


tweettext=df['text']


# In[195]:


blob=TextBlob(wordstring)
blob


# In[203]:


df_twit = pd.DataFrame()
df_twit['data']=tweettext
df_twit


# In[211]:


import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# In[213]:


corpus = []
for i in range(0,24):
    review = re.sub('[^a-zA-Z]', ' ', df_twit['data'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()  # Taking roots of different versions of the same word; 
                          # Not to have too many words in the end; 
                          # To regroup same versions of the words;
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
corpus


# In[196]:


blob.sentiment


# In[197]:


# Plots
import matplotlib.pyplot as plot
get_ipython().run_line_magic('matplotlib', 'inline')
polarity=[]
subj=[]
for t in tweettext:
    tx=TextBlob(t)
    polarity.append(tx.sentiment.polarity)
    subj.append(tx.sentiment.subjectivity)
    
poltweet= pd.DataFrame({'polarity':polarity,'subjectivity':subj})   
poltweet.plot(title='Polarity and Subjectivity')


# In[245]:


combined_data=pd.DataFrame()
combined_data=pd.concat([df_twit['data'],Facebook_Messages['Messages']],ignore_index=True)
combined_data


# In[278]:


combined_data['Messages']=combined_data[0]


# In[253]:


combined_data


# In[261]:


tweettext=combined_data


# In[275]:


tx=tweettext[1]
blob=TextBlob(tx)
tx


# In[276]:


blob.sentiment


# In[268]:


# Plots
import matplotlib.pyplot as plot
get_ipython().run_line_magic('matplotlib', 'inline')
polarity=[]
subj=[]
for t in tweettext:
    tx=TextBlob(t)
    polarity.append(tx.sentiment.polarity)
    subj.append(tx.sentiment.subjectivity)
    
poltweet= pd.DataFrame({'polarity':polarity,'subjectivity':subj})   
poltweet.plot(title='Polarity and Subjectivity')


# # <font color='red'>Conclusion and Future Scope</font>

# 
# #### We are planning on sending the feedback from the above analysis to the owners of the companies, using which they can improve their service, inturn maximizing their profits.
