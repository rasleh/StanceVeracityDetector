import csv
import json
import os


def explore(tweet_id, current_branch, raw_data, tree):
    current_tweet = raw_data[tweet_id]
    if not current_tweet['children']:
        current_branch.append(current_tweet)
        tree.append(current_branch)
    else:
        for child_id in current_tweet['children']:
            new_branch = current_branch[:]
            new_branch.append(current_tweet)
            explore(child_id, new_branch, raw_data, tree)


def load_raw_twitter(path):
    data = []
    print('Loading raw Twitter data')
    with open(path) as file:
        for line in file:
            root_id = line.split('\t')[0]
            raw_data = json.loads(line.split('\t')[1])
            root_tweet = raw_data[root_id]
            tree = [root_tweet]
            for child in root_tweet['children']:
                branch = []
                explore(child, branch, raw_data, tree)
            data.append(tree)
    return data


def load_raw_dast(path):
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
    switch = {
        'dast': lambda: load_raw_dast(path),
        'twitter': lambda: load_raw_twitter(path)
    }
    return switch.get(database)()


# Loads data from the DAST dataset, for use in Stance_LSTM
def load_dast_lstm(path):
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
    print('Loading data from {} for veracity prediction'.format(path))
    data = []
    with open(path) as file:
        file.readline()  # Skip header
        for line in file:
            veracity, feature_vector = line.replace('\n', '').split('\t')
            feature_vector = feature_vector.replace('[[', '').replace(']]', '').split('], [')
            if veracity is '2':
                if unverified_cast is 'true':
                    veracity = 1
                elif unverified_cast is 'false':
                    veracity = 0
            data.append((int(veracity), [[float(y) for y in x.split(', ')] for x in feature_vector]))
    print('Completed data load')
    return data
