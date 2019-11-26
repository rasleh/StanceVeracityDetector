import argparse
import csv
import datetime
import sys
from pathlib import Path

from src import data_loader
import os

current_path = os.path.abspath(__file__)

"""
A script containing methods for preprocessing data for use in veracity determination. Currently the script is set up to
handle the DAST dataset, and data generated by the twitter-conversation-scraper project. Information regarding data
structures can be found in the README at the project root.
"""


def get_database_variables(database, raw_data_path):
    """
    Switch function which generates variables based on which database type is entered as argument, currently supporting
    'dast' and 'twitter'. Defines raw data path, out path and path to overview of rumour veracities.

    :param raw_data_path: full path to the raw data
    :param database: database type, currently supporting 'dast', 'twitter' and 'pheme'
    :return: three database-specific variables; raw data path, out path and path to overview of rumour veracities
    """
    if not raw_data_path:
        raw_path_switch = {
            'dast': '../../data/datasets/dast/raw/dataset/',
            'twitter': '../../data/datasets/twitter/raw/loekke.txt',
            'pheme': '../../data/datasets/pheme/raw/'
        }
        raw_data_path = os.path.join(current_path, Path(raw_path_switch.get(database)))

    out_path_switch = {
        'dast': '../../data/datasets/dast/preprocessed/veracity/',
        'twitter': '../../data/datasets/twitter/preprocessed/veracity/',
        'pheme': '../../data/datasets/pheme/preprocessed/veracity/'
    }
    out_path = os.path.join(current_path, Path(out_path_switch.get(database)))

    veracity_path_switch = {
        'dast': '../../data/datasets/dast/raw/annotations/rumour_overview.txt',
        'twitter': '../../data/datasets/twitter/raw/rumour_overview.txt',
        'pheme': '../../data/datasets/pheme/raw/rumour_overview.txt'

    }
    veracity_path = os.path.join(current_path, Path(veracity_path_switch.get(database)))

    return raw_data_path, out_path, veracity_path


def write_preprocessed(data, out_path):
    """
    Writes data which has been preprocessed by the preprocess() method to a file at a given out path.

    :param data: an array of preprocessed data
    :param out_path: a full data path at which data is to be written
    """
    if not data:
        print('No preprocessed data detected')
        return
    print('Writing veracity data to', out_path)
    with open(out_path, "w+", newline='') as out_file:
        csv_writer = csv.writer(out_file, delimiter='\t')
        header = ['TruthStatus', 'SDQC_Labels']
        csv_writer.writerow(header)
        for branch in data:
            csv_writer.writerow(branch)
    print('Done')


def preprocess_dast_branch(include_timestamp, branch, sdqc_dict):
    if include_timestamp:
        if len(branch) is 1:
            branch_features = [[sdqc_dict[branch[0]['comment']['SDQC_Submission']], 0]]

        else:
            branch_features = []
            latest = datetime.datetime.min
            earliest = datetime.datetime.max
            for comment in branch:
                created = datetime.datetime.strptime(comment['comment']['created'], '%Y-%m-%dT%H:%M:%S')
                if created < earliest:
                    earliest = created
                if created > latest:
                    latest = created

            for comment in branch:
                created = datetime.datetime.strptime(comment['comment']['created'], '%Y-%m-%dT%H:%M:%S')
                branch_features.append([sdqc_dict[comment['comment']['SDQC_Submission']],
                                        (created - earliest) / (latest - earliest)])
    else:
        branch_features = [[sdqc_dict[x['comment']['SDQC_Submission']]] for x in branch]

    return branch_features


def preprocess_branch(include_timestamp, branch, sdqc_dict):
    if include_timestamp:
        if len(branch) is 1:
            # Ignore unlabeled tuples
            if branch[0]['SDQC_Submission'] == 'Underspecified':
                return
            branch_features = [[sdqc_dict[branch[0]['SDQC_Submission']], 0]]
        else:
            branch_features = []
            latest = datetime.datetime.min
            earliest = datetime.datetime.max
            for comment in branch:
                created = datetime.datetime.strptime(comment['created_at'], '%a %b %d %H:%M:%S %z %Y').replace(tzinfo=None)
                if created < earliest:
                    earliest = created
                if created > latest:
                    latest = created

            for comment in branch:
                created = datetime.datetime.strptime(comment['created_at'], '%a %b %d %H:%M:%S %z %Y').replace(tzinfo=None)
                if comment['SDQC_Submission'] == 'Underspecified':
                    continue
                branch_features.append([sdqc_dict[comment['SDQC_Submission']],
                                        (created - earliest) / (latest - earliest)])
    else:
        branch_features = [[sdqc_dict[x['SDQC_Submission']]] for x in branch if x['SDQC_Submission'] != 'Underspecified']

    return branch_features


def preprocess(database, data_path=False, write_out=False, include_timestamp=False, out_file_name='preprocessed.csv'):
    """
    Loads raw data at a given data path, extracts features to be used for veracity prediction, formats the data, and
    returns the processed data. If so specified, saves the preprocessed data to a data file.

    :param database: a database type, supporting either 'dast' or 'twitter'
    :param data_path: the path to the raw data which is to be preprocessed
    :param write_out: whether the preprocessed data should be saved to file
    :param include_timestamp: whether timestamps should be included as features or not
    :param out_file_name: the name of the generated file containing the preprocessed data
    :return: an array of tuples, each tuple containing (veracity, [branch features]). [branch features] contains an
    array for each data point; at index [0] of which the SDQC value of a data point is found, and if timestamps are
    included as features, they are normalized and found at index [1]
    """
    veracity_dict = {'False': 0, 'True': 1, 'Unverified': 2}
    sdqc_dict = {'Supporting': 0, 'Denying': 1, 'Querying': 2, 'Commenting': 3}

    data_path, out_path, veracity_path = get_database_variables(database, data_path)
    raw_data = data_loader.load_raw_data(data_path, database)

    data = []

    for tree in raw_data:
        veracity = veracity_dict[tree[0]['TruthStatus']]
        for branch in tree[1:][1:]:  # Skips first message of each branch, as this is the source
            if database == 'dast':
                branch_features = preprocess_dast_branch(include_timestamp, branch, sdqc_dict)
            else:
                branch_features = preprocess_branch(include_timestamp, branch, sdqc_dict)
            if branch_features and branch_features is not None:
                data.append((veracity, branch_features))

    if write_out:
        out_path = os.path.join(out_path, out_file_name)
        write_preprocessed(data, out_path)

    return data


if __name__ == '__main__':
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Preprocessing data for use in veracity prediction, defaults provided.')
    parser.add_argument('-db', '--database', default='pheme', help='Database type, either \'twitter\', \'dast\' or \'pheme\'.')
    parser.add_argument('-ts', '--timestamps', default=True, help='Include timestamps as features')
    parser.add_argument('-dp', '--data_path', default=False, help='Path to raw data')
    parser.add_argument('-wo', '--write_out', default=True, help='Write preprocessed data to file?')
    parser.add_argument('-on', '--out_file_name', default='preprocessed.csv', help='Name of out file')

    args = parser.parse_args(argv)
    preprocess(database=args.database, include_timestamp=args.timestamps, data_path=args.data_path,
               write_out=args.write_out, out_file_name=args.out_file_name)
