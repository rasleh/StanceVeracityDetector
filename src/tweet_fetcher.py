import os
from collections import deque
import json
from datetime import date
import configparser
import tweepy

current_path = os.path.abspath(__file__)
ini_path = os.path.join

# TODO: Make fetcher use absolute paths
# TODO: Implement command-line client
# Performs authentication necessary to access the Twitter API, using the credentials given in twitter.ini
def authenticate():
    config = configparser.ConfigParser()
    config.read('../twitter.ini')

    consumer_key = config.get('Twitter', 'consumer_key')
    consumer_secret = config.get('Twitter', 'consumer_secret')
    access_key = config.get('Twitter', 'access_key')
    access_secret = config.get('Twitter', 'access_secret')

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    return tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


def navigate_to_source(tweet_id):
    tweet = api.get_status(tweet_id)
    username = tweet.user.name
    parent = tweet.in_reply_to_status_id_str
    while parent:
        print('Given tweet is not a root node - navigating to root node\n')
        tweet_id = parent
        tweet_status = api.get_status(tweet_id)
        parent = tweet_status.in_reply_to_status_id_str
        username = tweet_status.user.name
    return tweet_id, username


def add_sdqc_placeholders(tweet_item):
    tweet_item['SourceSDQC'] = "Underspecified"
    tweet_item['SDQC_Parent'] = "Underspecified"
    tweet_item['SDQC_Submission'] = "Underspecified"


def identify_comments(tweet_id, username):
    children = []
    # Lookup a given user, extract all recent replies to user, and check if replies are to a specific tweet
    for result in tweepy.Cursor(api.search, q='to:' + username, result_type='recent', timeout=999999, tweet_mode='extended').items():
        if hasattr(result, 'in_reply_to_status_id_str'):
            if result.in_reply_to_status_id_str == tweet_id:
                # Mark tweets for further investigation, and add tweet id to list of comments
                tweets_of_interest.append(result)
                children.append(result.id_str)

    # Add ids for all commenting tweets to json of parent tweet
    collected_tweets[tweet_id]['children'] = children


# TODO: Remove the overwrite stuff, write it into the two primary methods as updates to a json object -> Update the
#  "Children" array with any new posts.
def write_to_file(source_tweet_id):
    with open('tweet_data.txt', 'r+', encoding="UTF-8") as db_file:
        in_db = False
        data = []
        if db_file.read(1):
            db_file.seek(0)
            data = db_file.readlines()
            for i in range(len(data)):
                if data[i].split('\t')[0] == source_tweet_id:
                    data[i] = source_tweet_id + '\t' + str(collected_tweets)
                    in_db = True

            if not in_db:
                data[-1] = data[-1] + '\n'

        if not in_db:
            data.append(source_tweet_id + '\t' + str(collected_tweets))

        db_file.seek(0)
        print(data)
        for i in range(len(data)):
            db_file.write(data[i])


def retrieve_conversation_thread(tweet_id, write_out=False):
    source_tweet_id, source_username = navigate_to_source(tweet_id)

    # Collect source tweet, add to collected tweets and fill SDQC-related fields with placeholders
    source_tweet_item = api.get_status(source_tweet_id, tweet_mode='extended')._json
    print("Scraping from source tweet {}\n{}\n\nCollected tweets:"
          .format(source_tweet_id, source_tweet_item['full_text'].replace('\n', ' ')))
    add_sdqc_placeholders(source_tweet_item)
    source_tweet_item['full_text'] = source_tweet_item['full_text'].replace('\n', ' ')
    collected_tweets[str(source_tweet_id)] = source_tweet_item

    # Identify tweets commenting on source
    identify_comments(source_tweet_id, source_username)

    # Iterate over tweets identified in comment section, collect them, and search for deeper comments
    while tweets_of_interest.__len__() != 0:
        item_of_interest = tweets_of_interest.popleft()
        add_sdqc_placeholders(item_of_interest._json)
        item_of_interest._json['full_text'] = item_of_interest._json['full_text'].replace('\n', ' ')
        collected_tweets[item_of_interest.id_str] = item_of_interest._json
        print(item_of_interest.id_str+"\t"+item_of_interest.full_text)
        identify_comments(item_of_interest.id_str, item_of_interest.user.screen_name)
    if write_out:
        # Save tweets in JSON format
        write_to_file(source_tweet_id)

    return collected_tweets


tweets_of_interest = deque()
collected_tweets = {}
api = authenticate()


#if __name__ == '__main__':

    # retrieve_conversation_thread("1168771907036033024", "oestergaard")
    # retrieve_conversation_thread("1168845054569566208", True)
    # retrieve_conversation_thread("1169544969784320000", "Kristianthdahl")
