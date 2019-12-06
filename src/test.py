import argparse
import json
import os
import sys
from pathlib import Path

from src import tweet_fetcher
from src import veracity
from src import preprocess_stance
from src.models.hmm_veracity import HMM
from src.models.lstm_stance import StanceLSTM

current_path = os.path.abspath(__file__)
veracity_hmm_model_no_timestamps = os.path.join(current_path, Path('../../pretrained_models/hmm_1_branch.joblib'))
#veracity_hmm_model_no_timestamps = os.path.join(current_path, Path('../../pretrained_models/hmm_branch_0.52_nts_truecast.joblib'))
stance_lstm_model = os.path.join(current_path, Path('../../pretrained_models/stance_lstm_3_200_1_50_0.36.joblib'))
potential_rumour_path = os.path.join(current_path, Path('../potential_rumours.txt'))
default_raw_twitter_path = os.path.join(current_path, Path('../../data/datasets/twitter/raw/'))

features = dict(text=False, lexicon=False, sentiment=False, pos=False, wembs=False, lstm_wembs=True)


def live_veracity_twitter(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-smp', '--stance_model_path', default=stance_lstm_model,
                        help='Path to pre-trained stance detection model')
    parser.add_argument('-vmp', '--veracity_model_path', default=veracity_hmm_model_no_timestamps,
                        help='Path to pre-trained veracity prediction model')
    parser.add_argument('-ts', '--timestamps', default=False,
                        help='Include normalized timestamps of comments as features?')

    args = parser.parse_args(argv)

    with open(potential_rumour_path, 'w', encoding='UTF-8') as rumour_file:
        for source_tweet_id, collected_tweets in tweet_fetcher.popular_search():
            data = [veracity.generate_tweet_tree(collected_tweets[source_tweet_id], collected_tweets)]
            dataset, feature_vectors = preprocess_stance.preprocess('twitter', data, text=features['text'],
                                              lexicon=features['lexicon'],
                                              sentiment=features['sentiment'], pos=features['pos'],
                                              wembs=features['wembs'], lstm_wembs=features['lstm_wembs'])
            for result in veracity.predict_veracity(args, dataset, feature_vectors):
                rumour_file.write(result)
                rumour_file.flush()


def get_raw_twitter_data_size(file_name):
    with open(os.path.join(default_raw_twitter_path, file_name), 'r', encoding='UTF-8') as data_file:
        tweet_count = 0
        for line in data_file:
            tweets = json.loads(line.split('\t')[1])
            for tweet_id, tweet in tweets.items():
                print(tweet['full_text'].replace('\n', ' '))
            tweet_count += len(tweets)
        print(tweet_count)


get_raw_twitter_data_size('2019-12-05.txt')
