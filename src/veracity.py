import argparse
import datetime
import os
import sys
from pathlib import Path

import numpy as np
import torch
from src import tweet_fetcher
from src.models.hmm_veracity import HMM
from src.models.lstm_stance import StanceLSTM
from src.preprocess_stance import preprocess
from src.data_loader import generate_tweet_tree

from joblib import load

current_path = os.path.abspath(__file__)
stance_lstm_model = os.path.join(current_path, Path('../../pretrained_models/stance_lstm_3_200_1_50_0.36.joblib'))
veracity_hmm_model_timestamps = os.path.join(current_path, Path('../../pretrained_models/hmm_branch_0.56_ts.joblib'))
veracity_hmm_model_no_timestamps = os.path.join(current_path, Path('../../pretrained_models/hmm_1_branch.joblib'))
twitter_data_path = os.path.join(current_path, Path('../../data/datasets/twitter/raw/loekke_oestergaard.txt'))
dast_data_path = os.path.join(current_path, Path('../../data/datasets/dast/raw/dataset/'))


"""
Script for loading two machine learning models; one for stance detection and one for veracity determination, predicting 
stance for a dataset of comments in a branch structure and, based on those stance predictions, determining the veracity 
of the comment at the root of each branch.

Methods
find_early_late(branch_features, dataset)
    goes through each datapoint in a dataset, finding the earliest and latest creation date for data points
predict_stance(feature_vector, lstm_clf)
    predicts the stance of a datapoint given the datapoint's feature vector, using a given classifier
"""


def find_early_late(branch_features, dataset):
    """
    Finds the earliest and latest creation dates for datapoints in a dataset

    :param branch_features: an array of features, with the ID of the comment at index [0]
    :param dataset: a dataset of comments of the DataSet class
    :return: the earliest and latest creation times of datapoints in the dataset
    """
    latest = datetime.datetime.min
    earliest = datetime.datetime.max

    for vector in branch_features:
        created = dataset.annotations[vector[0]].created
        if created < earliest:
            earliest = created
        if created > latest:
            latest = created

    return earliest, latest


def predict_stance(feature_vector, clf):
    """
    Predicts the stance in a given feature vector, using the classifier passed as argument, returning these as an array

    :param feature_vector: an array of features
    :param clf: a stance detection classifier
    :return: array of class predictions
    """
    # Exclude first two parts of vector; text ID and text label
    vector = feature_vector[2:]
    embs = []
    for emb in vector:
        if clf and type(emb) is list:
            # Flatten vector further to allow use of LSTM model
            for obj in emb:
                embs.extend(obj)
        else:
            embs.extend(emb)

    # Get model prediction
    label_scores = clf(torch.tensor(embs))
    predicted = [torch.argmax(label_scores.data, dim=1).item()]
    return predicted


def predict_veracity(args, dataset, feature_vectors):
    predictions = []
    num_to_stance = {0: 'Supporting', 1: 'Denying', 2: 'Querying', 3: 'Commenting'}

    hmm_clf = load(args.veracity_model_path)
    lstm_clf = load(args.stance_model_path)

    pointer = 0
    for source in dataset.submissions:
        print("Predicting veracity for following source tweet based on branches: {}\n{}\n\n"
              .format(source.source.id, source.source.text))

        for branch in source.branches:
            branch_features = feature_vectors[pointer]
            pointer += 1

            # Cases where features could not be extracted for a given branch - e.g. when only using word embeddings,
            # but text contains nothing to embed
            if len(branch_features) is 0:
                continue

            if args.timestamps:
                earliest, latest = find_early_late(branch_features, dataset)

            veracity_features = []

            # Flatten vector to allow prediction
            for vector in branch_features:
                predicted = predict_stance(vector, lstm_clf)
                comment_id = vector[0]

                if args.timestamps:
                    if len(branch_features) is 1:
                        predicted.append(0)
                    else:
                        created = dataset.annotations[comment_id].created
                        predicted.append((created - earliest) / (latest - earliest))

                if len(predicted) is not 0:
                    veracity_features.append(predicted)

            print("Stances in branch of length {}:".format(len(branch)))
            for i in range(len(veracity_features)):
                print("ID: {},\t\tLabel: {},\t\tPost: {}".format(branch[i].id, num_to_stance[veracity_features[i][0]],
                                                                 branch[i].text))

            veracity_features = np.array(veracity_features).reshape(-1, len(veracity_features))

            rumour_veracity = hmm_clf.predict([[veracity_features]])[0]
            # predictions.append("{}\t{}\t{}".format(source.source.text, rumour_veracity, veracity_features))
            if rumour_veracity:
                print("Source veracity: True, based on stances in comment branch\n")
            else:
                print("Source veracity: False, based on stances in comment branch\n")

            branch_labels = [num_to_stance[x] for x in veracity_features.ravel()]
            yield "Veracity: {}\tBranch stance labels: {}\tRumour text: {}\n".\
                format(rumour_veracity, branch_labels, source.source.text)


def veracity_stored(args, features):
    if args.data_path is None:
        if args.data_type is 'twitter':
            args.data_path = twitter_data_path
        elif args.data_type is 'dast':
            args.data_path = dast_data_path
        else:
            print('Defined data type not recognized')
            return

    dataset, feature_vectors = preprocess(args.data_type, args.data_path, text=features['text'],
                                          lexicon=features['lexicon'],
                                          sentiment=features['sentiment'], pos=features['pos'],
                                          wembs=features['wembs'], lstm_wembs=features['lstm_wembs'])
    return dataset, feature_vectors


def veracity_new(args, features):
    args.data_type = 'twitter'
    new_data = tweet_fetcher.retrieve_conversation_thread(args.id)
    source_tweet_id = new_data[0]
    raw_data = new_data[1]
    data = [generate_tweet_tree(raw_data[source_tweet_id], raw_data)]
    dataset, feature_vectors = preprocess(args.data_type, data, text=features['text'],
                                          lexicon=features['lexicon'],
                                          sentiment=features['sentiment'], pos=features['pos'],
                                          wembs=features['wembs'], lstm_wembs=features['lstm_wembs'])

    return dataset, feature_vectors


def main(argv):
    """
    Script main method, which loads two machine learning models, performing stance detection and subsequent veracity
    determination for a dataset of branches using these models, and printing the results.

    See project README for more in-depth description of command-line interfaces.

    :param argv: user-specified arguments parsed from command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-smp', '--stance_model_path', default=stance_lstm_model,
                               help='Path to pre-trained stance detection model')
    parser.add_argument('-vmp', '--veracity_model_path', default=None,
                               help='Path to pre-trained veracity prediction model')
    parser.add_argument('-ts', '--timestamps', default=True,
                        help='Include normalized timestamps of comments as features?')

    subparsers = parser.add_subparsers(help='Choose whether to use new or sored data for veracity preciction')

    # Create parser for using new data for veracity prediction
    new_parser = subparsers.add_parser('new', help='Using new data for veracity prediction')
    new_parser.add_argument('id', help='The ID of a tweet from the conversation, for which veracity will be determined')
    new_parser.set_defaults(func=veracity_new)

    # Create parser for using stored data for veracity prediction
    stored_parser = subparsers.add_parser('stored', help='Using stored data for veracity prediction. Defauilts are'
                                                         'supplied for all parameters')
    stored_parser.add_argument('-dt', '--data_type', default='twitter',
                        help='Type of data used for veracity prediction, either \'twitter\' or \'dast\'')
    stored_parser.add_argument('-dp', '--data_path', default=None, help='Path to data')
    stored_parser.set_defaults(func=veracity_stored)

    args = parser.parse_args(argv)

    if args.veracity_model_path is None:
        if args.timestamps:
            args.veracity_model_path = veracity_hmm_model_timestamps
        else:
            args.veracity_model_path = veracity_hmm_model_no_timestamps

    features = dict(text=False, lexicon=False, sentiment=False, pos=False, wembs=False, lstm_wembs=True)

    dataset, feature_vectors = args.func(args, features)

    predict_veracity(args, dataset, feature_vectors)


if __name__ == "__main__":
    main(sys.argv[1:])
