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
dast_veracity_path_timestamps = os.path.join(current_path, Path('../../../data/datasets/dast/preprocessed/veracity/timestamps.csv'))
dast_veracity_path_no_timestamps = os.path.join(current_path, Path('../../../data/datasets/dast/preprocessed/veracity/no_timestamps.csv'))

# lambda which flattens list of lists
flatten = lambda l: [item for sublist in l for item in sublist]


# TODO: Refactor and split file; one class-file containing only HMM logic, one script-file containing e.g. benchmarking,
#  saving features and command-line client
class HMM(BaseEstimator):
    """Single spaced hidden markov model classifier.

    Attributes
    models : dict
        a dictionary connecting class labels to an HMM model calculating probability for that class
    components : int
        number of states in the model

    Methods
    fit(data)
        Initializes an HMM for each class label in data, and estimates model parameters to optimally identify correct
        classes
    predict(data)
        Predicts the label of data points in the given data using a pre-trained HMM for each class label
    test(data)
        Predicts class labels using the predict(function), compares these with actual class labels and prints and
        returns the results
    """

    def __init__(self, components):
        self.components = components
        self.models = dict()

    def fit(self, data):
        """
        Estimates model parameters by initializing a Gaussian HMM for each class label and fitting data for that model

        :param data: matrix with the dimensions [number of datapoints][2][1 or 2]
        In the first matrix dimension, each datapoint will be stored. In the second dimension, at index 0, the veracity
        label of a given rumour will be stored. At index 1, the features will be stored. The third dimension will be of
        size 1 or 2, depending on whether only SDQC labels are used for the prediction, or timestamps are also included
        as features.
        :return: the HMM model, with sub-models fitted for each data label
        """
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
        """
        Finds most likely labels, using the pre-trained models found in self.models

        :param data: matrix with the dimensions [number of datapoints][1 or 2]
        In the first matrix dimension, each datapoint will be stored. The second dimension will be of
        size 1 or 2, depending on whether only SDQC labels are used for the prediction, or timestamps are also included
        as features.
        :return: an array of class predictions
        """
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
        """
        Finds model predictions using the predict() function, and compares results with the actual labels, printing
        results while also returning these as output. Needs indication of how Unverified rumours have been handled;
        cast as either True or False, or left as unverified to turn the task into 3-class prediction.

        :param data: matrix with the dimensions [number of datapoints][2][1 or 2]
        In the first matrix dimension, each datapoint will be stored. In the second dimension, at index 0, the veracity
        label of a given rumour will be stored. At index 1, the features will be stored. The third dimension will be of
        size 1 or 2, depending on whether only SDQC labels are used for the prediction, or timestamps are also included
        as features.
        :param unverified_cast: how unverified rumours have been handled; is 'none' if they have not been cast as
        another class, or alternatively 'true' or 'false'
        :return:
        """
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
    """
    Splits a dataset into train and test partitions based on user input

    :param data: an array of datapoints
    :param test_partition: how much of the data should be partitioned for the test split, all other data is used for
    training
    :return: two arrays containing datapoints
    """
    shuffle(data)
    test_data = data[:int(len(data) * test_partition)]
    train_data = data[int(len(data) * test_partition):]
    return test_data, train_data


def main(argv):
    """
    Client for initializing, training and testing a HMM model for veracity determination, and saving this model to a
    joblib file. Default values are supplied for all arguments.

    See project README for more in-depth description of command-line interfaces.

    :param argv: user-specified arguments parsed from command line.
    """

    parser = argparse.ArgumentParser(description='Training and testing HMM model for veracity prediction')
    parser.add_argument('-uc', '--unverified_cast', default='false',
                        help='Whether, and how, unverified rumours are cast, either \'true\', \'false\' or \'none\'')
    parser.add_argument('-sm', '--save_model', default=True, help='Whether the model is to be saved to a joblib file')
    parser.add_argument('-mn', '--model_name', help='Name for model joblib file, will be generated if not given')
    parser.add_argument('-ts', '--timestamps', default=False,
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

    data = data_loader.load_veracity(args.data_path, args.unverified_cast)
    test_data, train_data = split_test_train(data, 0.2)
    model = HMM(2)
    model.fit(train_data)
    class_acc, acc, f1_macro, f1_micro = model.test(test_data, args.unverified_cast)

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
