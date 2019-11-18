import csv
import json
import os


"""
Script containing methods used for loading data for use in both stance detection and veracity determination tasks.
"""


def explore(tweet_id, current_branch, raw_data, tree):
    """
    Method used for extracting the tree-like structure present in the twitter dataset. Generates a sub-branch for each
    child node a given tweet node has, and explores each child node. When no child nodes are present, the branch is
    added to the conversational tree.

    :param tweet_id: ID of the current tweet node in the conversation tree
    :param current_branch:
    :param raw_data: a dictionary connecting tweet IDs to their json objects
    :param tree: an array containing all branches in the conversational tree
    """
    current_tweet = raw_data[tweet_id]
    if not current_tweet['children']:
        current_branch.append(current_tweet)
        tree.append(current_branch)
    else:
        for child_id in current_tweet['children']:
            new_branch = current_branch[:]
            new_branch.append(current_tweet)
            explore(child_id, new_branch, raw_data, tree)


def generate_tweet_tree(source_tweet, raw_data):
    tree = [source_tweet]
    for child in source_tweet['children']:
        branch = []
        explore(child, branch, raw_data, tree)
    return tree


def load_raw_twitter(path):
    """
    Loads raw data in a tree-like structure into an array storing a single branch of the tree at each index

    :param path: full path to the raw twitter data
    :return: array of all branches in the dataset at the given data path
    """
    data = []
    print('Loading raw Twitter data')
    with open(path) as file:
        for line in file:
            root_id = line.split('\t')[0]
            raw_data = json.loads(line.split('\t')[1])
            source_tweet = raw_data[root_id]
            tree = generate_tweet_tree(source_tweet, raw_data)
            data.append(tree)
    return data


def load_raw_dast(path):
    """
    Loads raw data from the DAST dataset into an array storing a single branch of the tree at each index

    :param path: full path to the raw DAST data
    :return: array of all branches in the dataset at the given data path
    """

    print('Loading raw dast data')
    data = []
    for rumour_folder in os.listdir(path):
        rumour_folder_path = os.path.join(path, rumour_folder)
        if not os.path.isdir(rumour_folder_path):
            continue
        print("Loading event: ", rumour_folder)
        for submission_json in os.listdir(rumour_folder_path):
            submission_json_path = os.path.join(rumour_folder_path, submission_json)
            with open(submission_json_path, "r", encoding='utf-8') as file:
                print("Loading submission: ", submission_json)
                source = []
                json_obj = json.load(file)
                sub = json_obj['redditSubmission']
                source.append(sub)
                for branch in json_obj['branches']:
                    source.append(branch)
            data.append(source)
    print('Done\n')
    return data


def load_raw_data(path, database):
    """
    Switch function which calls another data load method depending on which database type is passed as method parameter.
    See the readme in the project root for information on data structures of raw data.

    :param path: full path to the raw data
    :param database: database type, currently supporting 'dast' and 'twitter'
    :return: array of all branches in the dataset at the given data path
    """
    switch = {
        'dast': lambda: load_raw_dast(path),
        'twitter': lambda: load_raw_twitter(path)
    }
    return switch.get(database)()


def load_dast_lstm(path):
    """
    Loads preprocessed data from the DAST dataset specifically for use in the StanceLSTM class. Expected data format is
    a CSV file with a header, containing a data point at each row. Column 1 should contain comment ID, column 2 the SDQC
    value of the comment, column 3 an array of word embedding arrays for the comment and column 4 an array of word
    embedding arrays for the root comment of the conversation tree to which the comment belongs

    :param path: full path to the data
    :return: an array of tuples containing (comment ID, comment SDQC value, [word embeddings]), where [word embeddings]
    contains at [0] the word embedding array of the source comment and at index [1] the embedding array of the comment
    """
    print('Loading data from {}, for stance_LSTM'.format(path))
    data = []
    csv.field_size_limit(10000000)
    with open(path, encoding='utf-8', newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t')
        next(csv_reader)  # Read in header, and ignore it, as it is not to be used as a feature
        for row in csv_reader:
            comment_id = row[0]
            sdqc = int(row[1])  # Submission SDQC
            feature_matrix = []
            for feature in row[2:]:
                emb_matrix = feature.split('], [')
                feature = []
                for emb in emb_matrix:
                    emb = emb.replace('[', '').replace(']', '').replace('\'', '').replace('\n', '').split(', ')
                    emb = [float(i) for i in emb if i is not '']
                    feature.append(emb)
                feature_matrix.append(feature)
            data.append((comment_id, sdqc, feature_matrix))
        print('Finished loading data')
    return data


def load_veracity(path, unverified_cast):
    """
    Loads preprocessed veracity data from a given data path, and allows casting unverified rumour as either "true" or
    "false". Data at "path" is expected to be a two-column CSV file. At the first column should be the veracity status
    of the rumour, and at column 2 should be the feature vectors for the rumour, each feature contained in an array.

    :param path: full data path to the preprocessed data
    :param unverified_cast: how unverified rumours should be handled; is 'none' if they should not been cast as
        another class, or alternatively 'true' or 'false'
    :return: returns a matrix with the dimensions [number of datapoints][1 or 2]. In the first matrix dimension, each
        datapoint will be stored. The second dimension will be of size 1 or 2, depending on whether only SDQC labels
        are used for the prediction, or timestamps are also included as features.
    """
    print('Loading data from {} for veracity prediction'.format(path))
    data = []
    with open(path) as file:
        file.readline()  # Skip header
        for line in file:
            veracity, feature_vector = line.replace('\n', '').split('\t')
            feature_vector = feature_vector.replace('[[', '').replace(']]', '').split('], [')
            if veracity == '2':
                if unverified_cast == 'true':
                    veracity = 1
                elif unverified_cast == 'false':
                    veracity = 0
            data.append((int(veracity), [[float(y) for y in x.split(', ')] for x in feature_vector]))
    print('Completed data load')
    return data
