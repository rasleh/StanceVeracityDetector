import os
from collections import deque
import json
from datetime import datetime
import configparser
import tweepy

current_path = os.path.abspath(__file__)
ini_path = os.path.join(current_path, '../../twitter.ini')


# TODO: Make fetcher use absolute paths
# TODO: Implement command-line client
# Performs authentication necessary to access the Twitter API, using the credentials given in twitter.ini
def authenticate():
    config = configparser.ConfigParser()
    config.read(ini_path)

    consumer_key = config.get('Twitter', 'consumer_key')
    consumer_secret = config.get('Twitter', 'consumer_secret')
    access_key = config.get('Twitter', 'access_key')
    access_secret = config.get('Twitter', 'access_secret')

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    return tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


def navigate_to_source(tweet):
    # Given tweet is an ID
    if type(tweet) is str:
        tweet_object = api.get_status(tweet, tweet_mode='extended')
        tweet_id = tweet
        all_tweets[tweet] = tweet_object
    # Given tweet is a tweet object
    else:
        tweet_object = tweet
        tweet_id = tweet.id_str
    username = tweet_object.user.screen_name
    parent = tweet_object.in_reply_to_status_id_str
    while parent:
        print('Given tweet is not a root node - navigating to root node\n')
        tweet_id = parent
        if tweet_id in all_tweets:
            tweet_object = all_tweets[tweet_id]
        else:
            tweet_object = api.get_status(tweet_id, tweet_mode='extended')
            all_tweets[tweet_id] = tweet_object
        parent = tweet_object.in_reply_to_status_id_str
        username = tweet_object.user.screen_name
    return tweet_id, username


def add_sdqc_placeholders(tweet_item):
    tweet_item['SourceSDQC'] = "Underspecified"
    tweet_item['SDQC_Parent'] = "Underspecified"
    tweet_item['SDQC_Submission'] = "Underspecified"


def identify_comments(tweet_id, username):
    children = []
    # Lookup a given user, extract all recent replies to user, and check if replies are to a specific tweet
    for result in tweepy.Cursor(api.search, q='to:' + username, since_id=tweet_id, result_type='recent', timeout=999999, tweet_mode='extended').items():
        if hasattr(result, 'in_reply_to_status_id_str'):
            if result.in_reply_to_status_id_str == tweet_id:
                # Mark tweets for further investigation, and add tweet id to list of comments
                all_tweets[result.id_str] = result
                tweets_of_interest.append(result.id_str)
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
        for i in range(len(data)):
            db_file.write(data[i])


def retrieve_conversation_thread(tweet_id, write_out=False):
    start_time = datetime.now()
    source_tweet_id, source_username = navigate_to_source(tweet_id)

    # Collect source tweet, add to collected tweets and fill SDQC-related fields with placeholders
    source_tweet_item = all_tweets[source_tweet_id]._json
    source_tweet_item['full_text'] = source_tweet_item['full_text'].replace('\n', ' ')
    print("Scraping from source tweet {}\n{}\n"
          .format(source_tweet_id, source_tweet_item['full_text']))
    if write_out:
        print("Collected tweets:")

    add_sdqc_placeholders(source_tweet_item)
    collected_tweets[source_tweet_id] = source_tweet_item

    # Identify tweets commenting on source
    identify_comments(source_tweet_id, source_username)
    # Iterate over tweets identified in comment section, collect them, and search for deeper comments
    while tweets_of_interest.__len__() != 0:
        item_of_interest_id = tweets_of_interest.popleft()
        item_of_interest = all_tweets[item_of_interest_id]
        add_sdqc_placeholders(item_of_interest._json)
        item_of_interest._json['full_text'] = item_of_interest._json['full_text'].replace('\n', ' ')
        collected_tweets[item_of_interest_id] = item_of_interest._json
        if write_out:
            print(item_of_interest_id+"\t"+item_of_interest.full_text)
        identify_comments(item_of_interest_id, item_of_interest.user.screen_name)
    if write_out:
        # Save tweets in JSON format
        write_to_file(source_tweet_id)
    print('Time elapsed for scraping: {}\n\n'.format(datetime.now()-start_time))
    return source_tweet_id, collected_tweets


def popular_search():
    cursor = 0
    for tweet in tweepy.Cursor(api.search, result_type='latest', lang='da', geocode='56.013377,10.362431,200km', count='100').items():
        cursor += 1
        if tweet.retweet_count > 20:
            popular_tweets[tweet.id_str] = tweet
        all_tweets[tweet.id_str] = tweet
        if cursor % 1000 is 0:
            print('Scraped {}/{} tweets, and {} popular tweets'.format(len(all_tweets), cursor, len(popular_tweets)))

        if cursor >= 1000:
            break

    for tweet_id, tweet in popular_tweets.items():
        source_id, username = navigate_to_source(tweet_id)
        if source_id is not tweet_id:
            del popular_tweets[tweet_id]
            if source_id not in popular_tweets:
                if source_id in all_tweets:
                    popular_tweets[source_id] = all_tweets[source_id]
                else:
                    popular_tweets[source_id] = api.get_status(source_id)

    print('Number of source tweets: {}'.format(len(popular_tweets)))


popular_tweets = {}
all_tweets = {}

tweets_of_interest = deque()
collected_tweets = {}
api = authenticate()

retrieve_conversation_thread('1194734268649549825')