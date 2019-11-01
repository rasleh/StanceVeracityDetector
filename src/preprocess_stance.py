import argparse
import csv
import os
import re
import sys
from pathlib import Path

from src import data_loader
from src.dataset_classes.DAST_datasets import DastDataset
from src.dataset_classes.datasets import DataSet
from src.feature_extraction.feature_extractor import FeatureExtractor

punctuation = re.compile('[^a-zA-ZæøåÆØÅ0-9]')
current_path = os.path.abspath(__file__)


def get_database_variables(database, data_path):
    if not data_path:
        path_switch = {
            'dast': '../../data/datasets/dast/raw/dataset/',
            'twitter': '../../data/datasets/twitter/raw/loekke.txt'
        }
        data_path = os.path.join(current_path, Path(path_switch.get(database)))

    dataset_switch = {
        'dast': DastDataset(),
        'twitter': DataSet()
    }
    dataset = dataset_switch.get(database)

    out_path_switch = {
        'dast': '../../data/datasets/dast/preprocessed/stance/',
        'twitter': '../../data/datasets/twitter/preprocessed/stance/'
    }
    out_path = os.path.join(current_path, Path(out_path_switch.get(database)))

    return data_path, dataset, out_path


def write_preprocessed(header_features, feature_vectors, out_path):
    if not feature_vectors:
        print('No preprocessed data detected')
        return
    print('Writing feature vectors to', out_path)
    with open(out_path, "w+", newline='') as out_file:
        csv_writer = csv.writer(out_file, delimiter='\t')
        header = ['id', 'sdqc_submission'] + header_features
        csv_writer.writerow(header)
        written_vectors = set()
        for branch in feature_vectors:
            for (comment_id, sdqc_submission, feature_vec) in branch:
                if comment_id not in written_vectors:
                    csv_writer.writerow([comment_id, sdqc_submission, *feature_vec])
                    written_vectors.add(comment_id)
    print('Done')


def get_branch_level_features(dataset, sdqc_parent, text, lexicon, sentiment, pos, wembs, lstm_wembs):
    feature_extractor = FeatureExtractor(dataset)
    feature_vectors = []

    for source_tweet in dataset.submissions:
        for branch in source_tweet.branches:
            branch_features = []
            for annotation in branch:
                features = feature_extractor.create_feature_vector(annotation, dataset, sdqc_parent, text, lexicon,
                                                                   sentiment, pos, wembs, lstm_wembs)
                if features:
                    branch_features.append(features)

            feature_vectors.append(branch_features)

    return feature_vectors


def preprocess(database, data_path=False, sub=False, sdqc_parent=False, text=False, lexicon=False, sentiment=False,
               pos=False, wembs=False, lstm_wembs=False, write_out=False, out_file_name='preprocessed.csv'):
    feature_inputs = [sdqc_parent, text, lexicon, sentiment, pos, wembs, lstm_wembs]
    feature_names = ['sdqc_parent', 'text', 'lexicon', 'sentiment', 'pos', 'word2vec', 'comment_wembs']
    features_header = [feature_names[i] for i in range(len(feature_inputs)) if feature_inputs[i] is True]
    if lstm_wembs:
        features_header.append('source_wembs')

    data_path, dataset, out_path = get_database_variables(database, data_path)

    raw_data = data_loader.load_raw_data(data_path, database)

    for tree in raw_data:
        dataset.add_submission(tree[0])
        for branch in tree[1:]:
            dataset.add_branch(branch, sub_sample=sub)

    feature_vectors = get_branch_level_features(dataset, sdqc_parent, text, lexicon, sentiment, pos, wembs, lstm_wembs)

    if write_out:
        out_path = os.path.join(out_path, out_file_name)
        write_preprocessed(features_header, feature_vectors, out_path)

    return dataset, feature_vectors


if __name__ == "__main__":
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser(description='Preprocessing data for use in stance detection, defaults provided. '
                                                 'LSTM stance model is currently only compatible with lstm_wembs.')
    parser.add_argument('-db', '--database', default='dast', help='Database type, either \'twitter\' or \'dast\'')
    parser.add_argument('-dp', '--data_path', default=False, help='Path to raw data')
    parser.add_argument('-ss', '--sub_sample', default=False,
                        help='Implement sub-sampling by removing conversation branches of only "commenting" labels')
    parser.add_argument('-sp', '--sdqc_parent', default=False, help='Include sdqc_parent as feature?')
    parser.add_argument('-tf', '--text_features', default=False, help='Include textual features?')
    parser.add_argument('-sm', '--sentiment', default=False, help='Include comment sentiment as feature?')
    parser.add_argument('-lx', '--lexicon', default=False, help='Include lexicon-based features, e.g. swear word count?')
    parser.add_argument('-pos', '--pos', default=False, help='Include POS tags as feature?')
    parser.add_argument('-we', '--word_embs', default=False, help='Include embedding-based features, e.g. cosine '
                                                                  'similarity across branches?')
    parser.add_argument('-le', '--lstm_wembs', default=True, help='Include LSTM-formatted word embedding features?')
    parser.add_argument('-wo', '--write_out', default=True, help='Write preprocessed data to file?')
    parser.add_argument('-on', '--out_file_name', default='preprocessed.csv', help='Name of out file')

    args = parser.parse_args(argv)

    preprocess(database=args.database, data_path=args.data_path, sub=args.sub_sample, sdqc_parent=args.sdqc_parent,
               text=args.text_features, sentiment=args.sentiment, lexicon=args.lexicon, pos=args.pos,
               wembs=args.word_embs, lstm_wembs=args.lstm_wembs, write_out=args.write_out,
               out_file_name=args.out_file_name)
