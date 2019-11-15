import argparse
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
stance_lstm_model = os.path.join(current_path, Path('../../pretrained_models/stance_lstm_3_200_1_50_0.36.joblib'))

features = dict(text=False, lexicon=False, sentiment=False, pos=False, wembs=False, lstm_wembs=True)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-smp', '--stance_model_path', default=stance_lstm_model,
                        help='Path to pre-trained stance detection model')
    parser.add_argument('-vmp', '--veracity_model_path', default=veracity_hmm_model_no_timestamps,
                        help='Path to pre-trained veracity prediction model')
    parser.add_argument('-ts', '--timestamps', default=False,
                        help='Include normalized timestamps of comments as features?')

    args = parser.parse_args(argv)

    for source_tweet_id, collected_tweets in tweet_fetcher.popular_search():
        data = [veracity.generate_tweet_tree(collected_tweets[source_tweet_id], collected_tweets)]
        dataset, feature_vectors = preprocess_stance.preprocess('twitter', data, text=features['text'],
                                          lexicon=features['lexicon'],
                                          sentiment=features['sentiment'], pos=features['pos'],
                                          wembs=features['wembs'], lstm_wembs=features['lstm_wembs'])
        veracity.predict_veracity(args, dataset, feature_vectors)


if __name__ == "__main__":
    main(sys.argv[1:])
