import argparse
import datetime
import os
import sys
from pathlib import Path

import numpy as np
import torch

from src.models.hmm_veracity import HMM
from src.models.lstm_stance import StanceLSTM
from src.preprocess_stance import preprocess

from joblib import load

current_path = os.path.abspath(__file__)
stance_lstm_model = os.path.join(current_path, Path('../../pretrained_models/stance_lstm_3_200_1_50_0.36.joblib'))
veracity_hmm_model_timestamps = os.path.join(current_path, Path('../../pretrained_models/hmm_branch_0.56_ts.joblib'))
veracity_hmm_model_no_timestamps = os.path.join(current_path, Path('../../pretrained_models/hmm_1_branch.joblib'))
twitter_data_path = os.path.join(current_path, Path('../../data/datasets/twitter/raw/loekke_oestergaard.txt'))
dast_data_path = os.path.join(current_path, Path('../../data/datasets/dast/raw/dataset/'))


def find_early_late(branch_features, dataset):
    latest = datetime.datetime.min
    earliest = datetime.datetime.max

    for vector in branch_features:
        created = dataset.annotations[vector[0]].created
        if created < earliest:
            earliest = created
        if created > latest:
            latest = created

    return earliest, latest


def predict_stance(feature_vector, lstm_clf):
    # Exclude first two parts of vector; text ID and text label
    vector = feature_vector[2:]
    embs = []
    for emb in vector:
        if lstm_clf and type(emb) is list:
            # Flatten vector further to allow use of LSTM model
            for obj in emb:
                embs.extend(obj)
        else:
            embs.extend(emb)

    # Get model prediction
    label_scores = lstm_clf(torch.tensor(embs))
    predicted = [torch.argmax(label_scores.data, dim=1).item()]
    return predicted


def main(argv):
    parser = argparse.ArgumentParser(description='Performing veracity prediction on new data, using pre-trained models.'
                                                 'Defaults supplied for all parameters.')
    parser.add_argument('-smp', '--stance_model_path', default=stance_lstm_model,
                        help='Path to pre-trained stance detection model')
    parser.add_argument('-vmp', '--veracity_model_path', default=None,
                        help='Path to pre-trained veracity prediction model')
    parser.add_argument('-dt', '--data_type', default='twitter',
                        help='Type of data used for veracity prediction, either \'twitter\' or \'dast\'')
    parser.add_argument('-ts', '--timestamps', default=True,
                        help='Include normalized timestamps of comments as features?')
    parser.add_argument('-dp', '--data_path', default=None, help='Path to data')

    args = parser.parse_args(argv)

    if args.data_path is None:
        if args.data_type is 'twitter':
            args.data_path = twitter_data_path
        elif args.data_type is 'dast':
            args.data_path = dast_data_path
        else:
            print('Defined data type not recognized')
            return

    if args.veracity_model_path is None:
        if args.timestamps:
            args.veracity_model_path = veracity_hmm_model_timestamps
        else:
            args.veracity_model_path = veracity_hmm_model_no_timestamps

    features = dict(text=False, lexicon=False, sentiment=False, pos=False, wembs=False, lstm_wembs=True)

    num_to_stance = {
        0: 'Supporting',
        1: 'Denying',
        2: 'Querying',
        3: 'Commenting'
    }

    hmm_clf = load(args.veracity_model_path)
    lstm_clf = load(args.stance_model_path)

    # TODO: Evaluate whether veracity should make use of pre-processed data, and data-loader class
    dataset, feature_vectors = preprocess(args.data_type, args.data_path, text=features['text'], lexicon=features['lexicon'],
                                          sentiment=features['sentiment'], pos=features['pos'],
                                          wembs=features['wembs'], lstm_wembs=features['lstm_wembs'])

    pointer = 0
    for source in dataset.submissions:
        print('ID is: ', source.source.id)

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
                        predicted.append((created-earliest)/(latest-earliest))

                if len(predicted) is not 0:
                    veracity_features.append(predicted)

            print("Stances in branch of length {}:".format(len(branch)))
            for i in range(len(veracity_features)):
                print("ID: {},\t\tLabel: {},\t\tPost: {}".format(branch[i].id, num_to_stance[veracity_features[i][0]], branch[i].text))

            veracity_features = np.array(veracity_features).reshape(-1, len(veracity_features))

            rumour_veracity = hmm_clf.predict([[veracity_features]])[0]

            if rumour_veracity:
                print("Veracity: True,\t\tSource id: {}\t\t Source post: {}\n".format(source.source.id, source.source.text))
            else:
                print("Veracity: False,\t\tSource id: {}\t\t Source post: {}\n".format(source.source.id, source.source.text))


if __name__ == "__main__":
    main(sys.argv[1:])
