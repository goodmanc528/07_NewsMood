
Analysis
1. Most of the compound sentiment scores for all of the media organizations covered in this assigment were distributed between +0.5 and -1.0.
2. Looking at the scatter plot chart there appears to more negative compound scores than positive.
3. The bar chart enforces the other two observations showing that all five media organizations have an overall negative compound score.   


```python
%matplotlib inline
import os
import pandas as pd
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import tweepy
import datetime as dt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
from config import consumer_key, consumer_secret, access_token, access_token_secret
```


```python
#Set target variables
targets = ['@BBCWorld','@CBSNews','@CNN', '@FoxNews', '@nytimes']

# #Build dictionaries to store Compound Scores and Tweets Ago information
cpdBBC = {}
cpdCBS = {}
cpdCNN = {}
cpdFOX = {}
cpdNYT = {}

key1 = 'Compound Score'
key2 = 'Tweets Ago'
key3 = 'Polarity'
key4 = 'Retweets'

date = dt.datetime.today().strftime("%Y/%m/%d")

#Store analysis dictionaries in a list
newsList = [cpdBBC, cpdCBS, cpdCNN, cpdFOX, cpdNYT]

#Twitter Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
#obtain 100 tweets from each target, 5 pages X 20 tweets = 100 tweets
counter = 0
for target in targets:
    #newsList[counter] = {}
#     print(type(counter))
#     print(newsList[counter])
    newsList[counter][key1] = []
    newsList[counter][key2] = []
    newsList[counter][key3] = []
    newsList[counter][key4] = []
    
    for i in range(1):
        allTweets = api.user_timeline(target, count=100, result_type='recent')
        #loop through tweets found and perform VADER analysis on each tweet.
        #Dump results in list for each news outlet
        tweetCounter = 0
        for tweet in allTweets:
            results = analyzer.polarity_scores(tweet['text'])
            cpdResults = results['compound']
            polarity = results['pos']-results['neg']
            retweet = tweet['retweet_count']
            newsList[counter][key1].append(cpdResults)
            newsList[counter][key2].append(tweetCounter)
            newsList[counter][key3].append(polarity)
            newsList[counter][key4].append(retweet)
            tweetCounter += 1
#     #Printing output to verify data looks appropriate
#     print(f'---------{target}----------')
#     print(f'Number of tweets: {len(newsList[counter])}.')
#     print(newsList[counter])
    counter += 1
```


```python
#convert dictionaries containing Compoung Scores and Tweets Ago to dataframe to create plots
dfBBC = pd.DataFrame(cpdBBC)
dfCBS = pd.DataFrame(cpdCBS)
dfCNN = pd.DataFrame(cpdCNN)
dfFOX = pd.DataFrame(cpdFOX)
dfNYT = pd.DataFrame(cpdNYT)
#compute overall compound score of the entire set captured.
avgBBC = dfBBC[key1].mean()
avgCBS = dfCBS[key1].mean()
avgCNN = dfCNN[key1].mean()
avgFOX = dfFOX[key1].mean()
avgNYT = dfNYT[key1].mean()
```


```python
fig, ax = plt.subplots(figsize=(9,7))
ax.scatter(key2, key1, c='violet',alpha=1, s=125, edgecolors='darkviolet', linewidths=2, data=dfBBC, label='BBC')
ax.scatter(key2, key1, c='khaki',alpha=1, s=125, edgecolors='darkgoldenrod', linewidths=2, data=dfCBS, label='CBS')
ax.scatter(key2, key1, c='lightcoral',alpha=1, s=125, edgecolors='firebrick', linewidths=2, data=dfCNN, label='CNN')
ax.scatter(key2, key1, c='palegreen',alpha=1,s=125, edgecolors='darkgreen', linewidths=2, data=dfFOX, label='FOX')
ax.scatter(key2, key1, c='lightblue',alpha=1,s=125, edgecolors='dodgerblue', linewidths=2, data=dfNYT, label='New York Times')
ax.set_xlim(105, -5)
ax.set_ybound(-1.05, 1.05)
ax.set_yticks([-1, -0.5, 0, 0.5, 1.0])
ax.set_title(f'Sentiment Analysis on Media Tweets {date}')
ax.grid(alpha=0.3, color='k', fillstyle='full')
ax.set_xlabel('Tweets Ago')
ax.set_ylabel('Compound Score')

leg = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, fontsize='large' )
for handle in leg.legendHandles:
    handle.set_sizes([150.0])
fig.savefig('SentimentAnalysis-1.png', bbox_inches='tight')
plt.show()
```


![png](output_5_0.png)



    <matplotlib.figure.Figure at 0x20590fbe400>



```python
secondPlot = pd.Series(
    [avgBBC, avgCBS, avgCNN, avgFOX, avgNYT],
    index = ['BBC','CBS','CNN','FOX','New York Times']
)

#Set descriptions:
plt.title(f'Tweet Sentiment by Media {date}')
plt.ylabel('Avg Compound Score')
plt.xlabel('Media')
plt.axhline(0, c='k')
#Plot the data:
my_colors = ['violet','khaki','lightcoral','palegreen','lightblue']
secondPlot.plot(
    figsize=(9,6),    
    kind='bar', 
    color=my_colors
)
plt.savefig('SentimentAnalysis-2.png', bbox_inches='tight')
plt.show()
```


![png](output_6_0.png)

