import configparser
import os
import tweepy
import json
import datetime

from tweepy import TweepError

current_path = os.path.abspath(__file__)
ini_path = os.path.join(current_path, '../../data/datasets/twitter/twitter.ini')
out_path = os.path.join(current_path, '../../data/datasets/twitter/claims_data/')


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


def navigate_to_highest_available_node(tweet):
    # Given tweet is a tweet object
    tweet_object = tweet
    tweet_id = tweet.id_str
    username = tweet_object.user.screen_name
    parent = tweet_object.in_reply_to_status_id_str
    while parent:
        try:
            tweet_object = api.get_status(parent, tweet_mode='extended')
        except TweepError:
            return tweet_id, username
        tweet_id = parent
        parent = tweet_object.in_reply_to_status_id_str
        username = tweet_object.user.screen_name
    return tweet_id, username


if __name__ == '__main__':
    print('Collecting claim data')
    api = authenticate()
    data = {}
    counter = 0
    for tweet in tweepy.Cursor(api.search, q='min_replies:5', result_type='latest', lang='da', geocode='56.013377,10.362431,200km', count='100').items():
        source_id, _ = navigate_to_highest_available_node(tweet)
        source_tweet = api.get_status(source_id, tweet_mode='extended')
        data[source_id] = source_tweet._json
        counter += 1
        if counter % 5 is 0:
            print('Scraped {} popular tweets. Latest tweet: {}'.format(len(data), tweet.created_at))
    counter = 0
    with open(os.path.join(out_path, 'claims_raw.json'), mode='r', encoding='utf-8') as claim_file:
        db_data = json.load(claim_file)

    with open(os.path.join(out_path, 'claims_raw.json'), mode='w', encoding='utf-8') as claim_file:
        for tweet in data:
            if tweet.id_str not in db_data:
                db_data[tweet.id_str] = tweet
                counter += 1
        json.dump(data, claim_file)
        print('Added {} new datapoints to claims_raw.json'.format(counter))
