import argparse
import collections
import os
import sys
from pathlib import Path
from random import shuffle

import joblib
import numpy as np
import sklearn.metrics as sk
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

from src import data_loader

current_path = os.path.abspath(__file__)
full_dast_path = os.path.join(current_path,
                              Path('../../../data/datasets/dast/preprocessed/stance/pure_lstm_subset.csv'))
test_dast_path = os.path.join(current_path,
                              Path('../../../data/datasets/dast/preprocessed/stance/pure_lstm_5g.csv'))
benchmark_path = os.path.join(current_path, Path('../../../benchmarking/'))

# TODO: Implement full support for bi-directionality
# TODO: Refactor and split file; one class-file containing only LSTM logic, one script-file containing e.g. benchmarking,
#  saving features and command-line client
class StanceLSTM(nn.Module):
    """
    LSTM architecture with variable dimensions in the form of number of LSTM layers and dimensions, ReLU layers and
    dimensions, and the option of making the model bi-directional.
    Inspired by https://discuss.pytorch.org/t/example-of-many-to-one-lstm/1728/4 and
    https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

    Attributes
    lstm_layers : int
        the number of LSTM layers to be used in the model
    lstm_dim : int
        the number of dimensions within each LSTM layer
    hidden_layers : int
        the number of linear hidden layers with ReLU activation functions
    hidden_dim : int
        the number of dimensions in each hidden layer
    emb_dim : int
        the size of the embeddings used in the data
    bi_directional : boolean
        whether the model should be bi-directional; bi-directionality is still not fully supported
    dropout = boolean
        whether the model should apply dropout
    lstm : nn.LSTM object
        an nn.LSTM object initialized using user input, containing a user specified number of layers and dimensions
    dense_layers : array
        an array of the hidden layers and, if so specified by the user, followed by a dropout layer
    hidden2label : Sequential object
        a torch object sequentializing the contents of dense_layers
    hidden : torch.tensor object
        the hidden state of the LSTM model, initialized by the init_hidden() function

    Methods
    init_hidden()
        initializes the hidden state of the model as torch tensors containing zeroes
    forward()
        performs a forward pass over the model layers
    """

    def __init__(self, lstm_layers, lstm_dim, hidden_layers, hidden_dim,
                 emb_dim, bi_directional, dropout=True):
        super(StanceLSTM, self).__init__()
        self.dropout = dropout
        num_labels = 4

        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers

        # LSTM layers
        self.lstm = nn.LSTM(emb_dim, lstm_dim, lstm_layers, bidirectional=bi_directional)

        # Linear layer
        dense_layers = collections.OrderedDict()
        dense_layers["lin0"] = torch.nn.Linear(lstm_dim, hidden_dim)
        dense_layers["rec0"] = torch.nn.ReLU()
        for i in range(hidden_layers - 1):
            dense_layers["lin%d" % (i + 1)] = torch.nn.Linear(hidden_dim, hidden_dim)
            dense_layers["rec%d" % (i + 1)] = torch.nn.ReLU()
        if dropout:
            dense_layers["drop"] = torch.nn.Dropout(p=0.5)
        dense_layers["lin%d" % hidden_layers] = torch.nn.Linear(hidden_dim, num_labels)
        self.hidden2label = torch.nn.Sequential(dense_layers)

        # initialize state
        self.hidden = self.init_hidden()

    def init_hidden(self):
        """Initializes an empty hidden state with axes semantics (hidden_layers, minibatch_size, hidden_dim)"""
        return torch.zeros(self.lstm_layers, 1, self.hidden_dim), torch.zeros(self.lstm_layers, 1, self.hidden_dim)

    def forward(self, text):
        """A forward pass over the full model, running the text through deep learning layers followed by the hidden layers,
        returning class probabilities as 'label_scores'"""
        lstm_out, self.hidden = self.lstm(text.view(len(text), 1, 300))
        label_space = self.hidden2label(lstm_out[-1])
        label_scores = f.log_softmax(label_space, dim=1)
        return label_scores


def train(data, model, loss_function, optimizer, epochs):
    """
    Train a given Stance_LSTM model for a number of epochs, using a given loss function and optimizer. The function takes
    data in list form, each element containing text id, a label and a feature vector. The feature vector contains two
    elements; word embeddings for a source text in a conversation branch, followed by the text for which stance is to be
    determined, towards the source text. Both on the format [text length][embedding size]

    :param data: array of training data, at each index containing a tuple; (ID, actual label, [feature vector])
    :param model: a StanceLSTM object to be trained
    :param loss_function: the applied loss function, expected to be of a torch.nn class
    :param optimizer: the applied optimizer, expected to be of a torch.optim class
    :param epochs: the number of epochs for which the model should be trained
    """
    epoch_loss = 0.0
    for epoch in range(epochs):
        for text_id, label, feature_vector in data:
            # Clear out gradients and hidden layers
            model.zero_grad()
            model.hidden = model.init_hidden()

            # Build tensor for submission source, and condition model on this
            source_tensor = torch.tensor(feature_vector[1])
            model(source_tensor)

            # Build tensors for quotes and labels
            target = torch.tensor([label])
            inputs = torch.tensor(feature_vector[0])

            # Extract class probability distribution for label data
            label_scores = model(inputs)

            # Calculate loss using the defined loss function, back-propagate loss and optimize model based on loss
            loss = loss_function(label_scores, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print('Epoch %d, loss: %.5f' % (epoch + 1, epoch_loss / 1000))
        epoch_loss = 0


def test(data, model):
    """
    Tests a pre-trained model. The function takes data in list form, each element containing text id, a label and a
    feature vector. The feature vector contains two  elements; word embeddings for a source text in a conversation branch,
    followed by the text for which stance is to be determined, towards the source text. Both on the format
    [text length][embedding size]

    :param data: array of test data, at each index containing a tuple; (ID, actual label, [feature vector])
    :param model: a StanceLSTM object to be tested
    :return: accuracy for each class, overall accuracy, F1 micro and macro averaged and a confusion matrix over
    classification
    """
    model.eval()
    predicted_labels = []
    actual_labels = []

    # Define no_grad to ensure no training is performed during testing
    with torch.no_grad():
        for comment_id, label, feature_vector in data:
            # Build tensor for submission source, and condition model on this
            source_tensor = torch.tensor(feature_vector[1])
            model(source_tensor)

            # Build quote tensor and extract class probability distribution
            inputs = torch.tensor(feature_vector[0])
            label_scores = model(inputs)

            # Extract most likely class, and save predicted and actual label
            predicted = torch.argmax(label_scores.data, dim=1)
            predicted_labels.extend(predicted.numpy())
            actual_labels.append(label)

    # Generate confusion matrix
    c_matrix = sk.confusion_matrix(actual_labels, predicted_labels, labels=[0, 1, 2, 3])
    cm = c_matrix.astype('float') / c_matrix.sum(axis=1)[:, np.newaxis]
    class_acc = cm.diagonal()
    acc = sk.accuracy_score(actual_labels, predicted_labels)
    f1_macro = sk.f1_score(actual_labels, predicted_labels, average='macro')
    f1_micro = sk.f1_score(actual_labels, predicted_labels, average='micro')
    model.train()
    return class_acc, acc, f1_macro, f1_micro, c_matrix


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


def run_specific_benchmark(lstm_layers, lstm_dim, hidden_layers, hidden_dim, max_epochs, bi_directional, out_file, data,
                           save_model=False, full_run=False):
    """
    Runs a benchmark using user-input hyperparameters, and saves the results to a file. For descriptions of lstm_layers,
    lstm_dim, hidden_layers, hidden_dim and bi_directional, see the StanceLSTM class.

    :param max_epochs: the maximum number of epochs for which the benchmark will run
    :param out_file: the file to which the benchmark results should be written
    :param data: array of data, at each index containing a tuple; (ID, actual label, [feature vector])
    :param save_model: whether the trained and tested StanceLSTM model should be saved to a joblib file
    :param full_run: whether the current benchmarking run is part of a full hyperparameter space search
    :return: the StanceLSTM model with best performance
    """
    # Split training and test data
    test_data, train_data = split_test_train(data, 0.2)

    # Initiate the LSTM model using user-defined parameters, and initialize loss function and optimizer
    model = StanceLSTM(lstm_layers, lstm_dim, hidden_layers, hidden_dim, 300, bi_directional)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters())
    best_model = model
    best_f1 = 0
    if not full_run:
        out_file.write("epochs,LSTMLayers,LSTMDims,ReLULayers,ReLUDims,totalAcc,f1_macro,f1_micro,S,D;Q;C\n")
    # Run train and test for model, and print out benchmark at each epoch count in the epoch hyperparameter space

    for i in range(max_epochs):
        train(train_data, model, loss_function, optimizer, 1)
        class_acc, acc, f1_macro, f1_micro, c_matrix = test(test_data, model)

        if save_model:
            if f1_macro > best_f1:
                best_f1 = f1_macro
                best_model = model

        out_file.write(
            "%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n" %
            (i+1, lstm_layers, lstm_dim, hidden_layers, hidden_dim, acc, f1_macro, f1_micro, class_acc[0],
             class_acc[1], class_acc[2], class_acc[3]))
        out_file.flush()

        if (i+1) % 5 == 0:
            print("Confusion matrix:")
            print(c_matrix)
            print("Class acc:", class_acc)
            print("Accuracy: %.5f" % acc)
            print("F1-macro:", f1_macro)
            print("F1-micro:", f1_micro)

    if save_model:
        return best_model


def run_full_benchmark(max_epochs):
    """
    Performs benchmarking for all of the hyperparameter combinations present in the arrays lstm_layers_var,
    lstm_dims_var, relu_layers_var and relu_dims_var.

    :param max_epochs: the maximum number of epochs for which a benchmark will run
    """
    # All parameter combinations in the arrays below will be benchmarked
    lstm_layers_var = [1, 2, 3]
    lstm_dims_var = [50, 100, 200]
    relu_layers_var = [1]
    relu_dims_var = [50, 100, 200]

    data = data_loader.load_dast_lstm(full_dast_path)
    with open(os.path.join(os.path.abspath(__file__), '../../benchmarking/NoSourceBackpropLSTM.csv', 'w')) as out_f:
        out_f.write("epochs,LSTMLayers,LSTMDims,ReLULayers,ReLUDims,totalAcc,f1_macro,f1_micro,For,Against,Neutral\n")
        for lstm_layers in lstm_layers_var:
            for lstm_dim in lstm_dims_var:
                for relu_layers in relu_layers_var:
                    for relu_dim in relu_dims_var:
                        run_specific_benchmark(lstm_layers, lstm_dim, relu_layers, relu_dim, max_epochs,
                                               out_file=out_f, bi_directional=False, data=data, full_run=True)


def main(argv):
    """
    Client for initializing, training and testing an LSTM model for stance detection, performing benchmarking, and
    saving this model to a joblib file. Default values are supplied for all arguments.

    See project README for more in-depth description of command-line interfaces.

    :param argv: user-specified arguments parsed from command line.
    """

    parser = argparse.ArgumentParser(description='Training and testing LSTM model for stance detection, all variables'
                                                 'will be set to defaults, if none is entered')
    parser.add_argument('-ll', '--lstm_layers', default=3, help='Number of LSTM layers in model')
    parser.add_argument('-ld', '--lstm_dimensions', default=200, help='Number of LSTM dimensions in each layer')
    parser.add_argument('-rl', '--relu_layers', default=1, help='Number of linear layers with ReLU activation function in model')
    parser.add_argument('-rd', '--relu_dimensions', default=50, help='Number of ReLU dimensions in each layer')
    parser.add_argument('-me', '--max_epochs', default=200, help='Maximal number of epocsh in training')
    parser.add_argument('-bd', '--bi_directional', default=False, help='Make model bi-directional')
    parser.add_argument('-sm', '--save_model', default=True, help='Whether the model is to be saved to a joblib file')
    parser.add_argument('-mn', '--model_name', help='Name for model joblib file, will be generated if not given')
    parser.add_argument('-bn', '--benchmark_name', help='Name for model benchmark file, will be generated if not given')
    parser.add_argument('-dp', '--data_path', default=full_dast_path,
                        help='Path to data file, DAST dataset is used as default')
    # TODO: Re-implement arg below, when load_preprocessed_data function is used; see TODO below
    # parser.add_argument('dt', 'data_type', default='dast', help='Type of data used, currently supporting either \'dast\' or \'twitter\'. Must reflect the given data path')

    args = parser.parse_args(argv)
    if not args.benchmark_name:
        benchmark_name = 'lstm_{}_{}_{}_{}.csv'.format(args.lstm_layers, args.lstm_dimensions, args.relu_layers,
                                                       args.relu_dimensions)
    else:
        benchmark_name = args.benchmark_name

    out_path = os.path.join(benchmark_path, benchmark_name)

    with open(out_path, 'w') as out_file:
        if not args.model_name:
            model_name = 'lstm_{}_{}_{}_{}.joblib'.format(args.lstm_layers, args.lstm_dimensions, args.relu_layers,
                                                          args.relu_dimensions)
        else:
            model_name = args.model_name

        # TODO: Re-write to make use of load_preprocessed_data in data_loader, and supply data_type
        data = data_loader.load_dast_lstm(args.data_path)

        print(
            'Preparing to run benchmarking for LSTM model with following specs on a dataset of size {}:'
            '\nLSTM layers: {},\tLSTM dimensions: {},\tReLU layers: {},\tReLU dimensions: {}'
                .format(
                len(data),
                args.lstm_layers,
                args.lstm_dimensions,
                args.relu_layers,
                args.relu_dimensions))
        model = run_specific_benchmark(args.lstm_layers, args.lstm_dimensions, args.relu_layers, args.relu_dimensions,
                                       args.max_epochs, args.bi_directional, out_file, data, save_model=True)
        if args.save_model:
            joblib.dump(model, os.path.join(current_path, Path('../../../pretrained_models/{}'.format(model_name))))


if __name__ == "__main__":
    main(sys.argv[1:])
