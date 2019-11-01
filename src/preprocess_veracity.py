import argparse
import csv
import datetime
import sys
from pathlib import Path

from src import data_loader
import os

current_path = os.path.abspath(__file__)


def get_database_variables(database, raw_data_path):
    if not raw_data_path:
        raw_path_switch = {
            'dast': '../../data/datasets/dast/raw/dataset/',
            'twitter': '../../data/datasets/twitter/raw/loekke.txt'
        }
        raw_data_path = os.path.join(current_path, Path(raw_path_switch.get(database)))

    out_path_switch = {
        'dast': '../../data/datasets/dast/preprocessed/veracity/',
        'twitter': '../../data/datasets/twitter/preprocessed/veracity/'
    }
    out_path = os.path.join(current_path, Path(out_path_switch.get(database)))

    veracity_path_switch = {
        'dast': '../../data/datasets/dast/raw/annotations/rumour_overview.txt',
        'twitter': '../../data/datasets/twitter/raw/rumour_overview.txt'
    }
    veracity_path = os.path.join(current_path, Path(veracity_path_switch.get(database)))

    return raw_data_path, out_path, veracity_path


def write_preprocessed(data, out_path):
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


def preprocess(database, data_path=False, write_out=False, include_timestamp=False, out_file_name='preprocessed.csv'):
    veracity_dict = {'False': 0, 'True': 1, 'Unverified': 2}
    sdqc_dict = {'Supporting': 0, 'Denying': 1, 'Querying': 2, 'Commenting': 3}

    data_path, out_path, veracity_path = get_database_variables(database, data_path)
    raw_data = data_loader.load_raw_data(data_path, database)

    data = []

    for tree in raw_data:
        veracity = veracity_dict[tree[0]['TruthStatus']]
        for branch in tree[1:][1:]:  # Skips first message of each branch, as this is the source
            if include_timestamp:
                if len(branch) is 1:
                    branch_features = [[sdqc_dict[branch[0]['comment']['SDQC_Submission']], 0]]
                else:
                    branch_features = []
                    latest = datetime.datetime.min
                    earliest = datetime.datetime.max
                    for comment in branch:
                        if database == 'dast':
                            created = datetime.datetime.strptime(comment['comment']['created'], '%Y-%m-%dT%H:%M:%S')
                        else:
                            created = datetime.datetime.strptime(comment['created_at'], '%a %b %d %H:%M:%S %z %Y')
                        if created < earliest:
                            earliest = created
                        if created > latest:
                            latest = created

                    for comment in branch:
                        if database == 'dast':
                            created = datetime.datetime.strptime(comment['comment']['created'], '%Y-%m-%dT%H:%M:%S')
                        else:
                            created = datetime.datetime.strptime(comment['created_at'], '%a %b %d %H:%M:%S %z %Y')

                        branch_features.append([sdqc_dict[comment['comment']['SDQC_Submission']],
                                                              (created-earliest)/(latest-earliest)])
            else:
                branch_features = [[sdqc_dict[x['comment']['SDQC_Submission']]] for x in branch]
            data.append((veracity, branch_features))

    if write_out:
        out_path = os.path.join(out_path, out_file_name)
        write_preprocessed(data, out_path)

    return data


if __name__ == '__main__':
    argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description='Preprocessing data for use in veracity prediction, defaults provided.')
    parser.add_argument('-db', '--database', default='dast', help='Database type, either \'twitter\' or \'dast\'. ')
    parser.add_argument('-ts', '--timestamps', default=True, help='Include timestamps as features')
    parser.add_argument('-dp', '--data_path', default=False, help='Path to raw data')
    parser.add_argument('-wo', '--write_out', default=True, help='Write preprocessed data to file?')
    parser.add_argument('-on', '--out_file_name', default='preprocessed.csv', help='Name of out file')

    args = parser.parse_args(argv)

    preprocess(database=args.database, include_timestamp=args.timestamps, data_path=args.data_path,
               write_out=args.write_out, out_file_name=args.out_file_name)
