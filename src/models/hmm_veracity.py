import argparse
import os
import sys
from pathlib import Path
from random import shuffle

import joblib
import numpy as np
import sklearn.metrics as sk
from hmmlearn import hmm
from sklearn.base import BaseEstimator

from src import data_loader

current_path = os.path.abspath(__file__)
dast_veracity_path_timestamps = os.path.join(current_path, Path('../../../data/datasets/dast/preprocessed/veracity/preprocessed_timestamps.csv'))
dast_veracity_path_no_timestamps = os.path.join(current_path, Path('../../../data/datasets/dast/preprocessed/veracity/preprocessed_no_timestamps.csv'))

# lambda which flattens list of lists
flatten = lambda l: [item for sublist in l for item in sublist]


class HMM(BaseEstimator):
    """Single spaced hidden markov model classifier."""

    def __init__(self, components):
        self.components = components
        self.models = dict()

    def fit(self, data):
        classes = dict()

        feature_count = len(data[1][1][0])

        # partition data in labels
        for datapoint in data:
            if datapoint[0] not in classes:
                classes[datapoint[0]] = []
            classes[datapoint[0]].append(datapoint[1])

        # Make and fit model for each label
        for veracity_label, sdqc_labels in classes.items():
            lengths = [len(x) for x in sdqc_labels]
            thread_flat = np.array(flatten(sdqc_labels)).reshape(-1, feature_count)
            self.models[veracity_label] = hmm.GaussianHMM(n_components=self.components).fit(thread_flat,
                                                                                            lengths=lengths)
        return self

    def predict(self, data):
        predicts = []
        for branch in data:
            branch_length = len(branch)
            branch_labels = np.array(branch).reshape(-1, len(branch[0]))
            strongest_label = -1
            best_probability = None
            for label, model in self.models.items():
                probability = model.score(branch_labels, lengths=[branch_length])
                if best_probability is None or probability > best_probability:
                    strongest_label = label
                    best_probability = probability

            predicts.append(strongest_label)

        return predicts

    def test(self, data, unverified_cast):
        feature_vectors = [x[1] for x in data]
        predicted_labels = self.predict(feature_vectors)
        actual_labels = [x[0] for x in data]
        if unverified_cast is not 'none':
            c_matrix = sk.confusion_matrix(actual_labels, predicted_labels, labels=[0, 1])
        else:
            c_matrix = sk.confusion_matrix(actual_labels, predicted_labels, labels=[0, 1, 2])
        print("Confusion matrix:")
        print(c_matrix)
        cm = c_matrix.astype('float') / c_matrix.sum(axis=1)[:, np.newaxis]
        class_acc = cm.diagonal()
        acc = sk.accuracy_score(actual_labels, predicted_labels)
        f1_macro = sk.f1_score(actual_labels, predicted_labels, average='macro')
        f1_micro = sk.f1_score(actual_labels, predicted_labels, average='micro')
        print("Class acc:", class_acc)
        print("Accuracy: %.5f" % acc)
        print("F1-macro:", f1_macro)
        print("F1-micro:", f1_micro)
        return class_acc, acc, f1_macro, f1_micro


def split_test_train(data, test_partition):
    shuffle(data)
    test_data = data[:int(len(data) * test_partition)]
    train_data = data[int(len(data) * test_partition):]
    return test_data, train_data


def run_benchmark(components, data_path, unverified_cast):
    data = data_loader.load_veracity(data_path, unverified_cast)
    test_data, train_data = split_test_train(data, 0.2)
    model = HMM(components)
    model.fit(train_data)
    class_acc, acc, f1_macro, f1_micro = model.test(test_data, unverified_cast)
    return model, f1_macro


def main(argv):
    parser = argparse.ArgumentParser(description='Training and testing HMM model for veracity prediction')
    parser.add_argument('-uc', '--unverified_cast', default='false',
                        help='Whether, and how, unverified rumours are cast, either \'true\', \'false\' or \'none\'')
    parser.add_argument('-sm', '--save_model', default=True, help='Whether the model is to be saved to a joblib file')
    parser.add_argument('-mn', '--model_name', help='Name for model joblib file, will be generated if not given')
    parser.add_argument('-ts', '--timestamps', default=True,
                        help='Include normalized timestamps of comments as features?')
    parser.add_argument('-dp', '--data_path', default=None,
                        help='Path to data file relative to hmm_veracity.py script, DAST dataset is used as default')

    args = parser.parse_args(argv)

    if args.data_path is None:
        if args.timestamps:
            args.data_path = dast_veracity_path_timestamps
        else:
            args.data_path = dast_veracity_path_no_timestamps

    if args.unverified_cast not in ['true', 'false', 'none']:
        print(
            'Please specify whether and how unverified rumours should be cast, either \'true\', \'false\' or \'none\'')
        return

    model, f1_macro = run_benchmark(2, args.data_path, args.unverified_cast)

    if args.save_model:
        if not args.model_name:
            if args.timestamps:
                model_name = 'hmm_branch_{}_ts.joblib'.format(round(f1_macro, 2))
            else:
                model_name = 'hmm_branch_{}.joblib'.format(round(f1_macro, 2))
        else:
            model_name = args.model_name

        joblib.dump(model, os.path.join(current_path, Path('../../../pretrained_models/{}'.format(model_name))))


if __name__ == "__main__":
    main(sys.argv[1:])
