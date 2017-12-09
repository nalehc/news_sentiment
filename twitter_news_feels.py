#Dependencies
import matplotlib.pyplot as plt
import pandas as pd
import tweepy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

#API Key Entry
consumer_key = input("Please enter twitter consumer key")
consumer_secret = input("Please enter twitter secret consumer key")
access_token = input("Please enter twitter token ID")
access_token_secret = input("Please enter twitter secret token")
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())

accounts = ["@BBC", "@CBS", "@CNN", "@FoxNews", "@nytimes"]

def tweet_frame(query):
    sentiments = []
    oldest_id = ''
    public_tweets = api.search(query, count=100,
                                   result_type="recent", max_id=oldest_id)
    for tweet in public_tweets['statuses']:
        text = tweet['text']
        scores = analyzer.polarity_scores(text)
        sentiments.append(scores)
        oldest_id = tweet['id_str']
    sentiments_df = pd.DataFrame(sentiments)
    print(sentiments_df.head())
    sentiments_df.to_csv(query + ".csv")
    plt.figure(figsize=(8, 5))
    sentiments_df['compound'].plot(marker='o', linewidth=0)
    plt.xlabel('Tweet number')
    plt.ylabel('Compound score')
    avg = sentiments_df['compound'].mean()
    plt.hlines(avg, 0, len(sentiments), linewidth=1, linestyle='dotted',
               color='red')
    title = ("VADER sentiment analysis on " + query)
    plt.title(title)
    plt.savefig(title + ".png")

for query in accounts:
    tweet_frame(query)








