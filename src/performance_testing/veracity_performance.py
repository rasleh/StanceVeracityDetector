import os
import sys

from src.data_loader import load_veracity
from src.models import hmm_veracity, veracity_majority_baseline
import random
import copy

current_path = os.path.abspath(__file__)
pheme_data_path = os.path.join(current_path, '../../../data/datasets/pheme/preprocessed/veracity/')
dast_data_path = os.path.join(current_path, '../../../data/datasets/dast/preprocessed/veracity/')


class HiddenPrints:
    """Class for suppressing the abundant print statements from hmm_veracity.test, courtesy of
    https://stackoverflow.com/questions/8391411/suppress-calls-to-print-python"""
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def format_data(pheme_ts, pheme_nts, dast_ts, dast_nts, testdata_type):
    dast_ts = cross_validation_splits(dast_ts, 5)
    dast_nts = cross_validation_splits(dast_nts, 5)

    if testdata_type == 'dastpheme':
        pheme_ts = cross_validation_splits(pheme_ts, 5)
        pheme_nts = cross_validation_splits(pheme_nts, 5)

    return pheme_ts, pheme_nts, dast_ts, dast_nts


def load_datasets(unverified_cast, testdata_type, remove_commenting):
    """Loads the relevant datasets used for testing of veracity performance"""
    if testdata_type not in ['dast', 'dastpheme']:
        err_msg = "Unrecognized test_dataset type {}, please use 'dast' or 'dastpheme'"
        raise RuntimeError(
            err_msg.format(testdata_type))

    pheme_ts = load_veracity(os.path.join(pheme_data_path, 'timestamps.csv'), unverified_cast, remove_commenting)
    pheme_nts = load_veracity(os.path.join(pheme_data_path, 'no_timestamps.csv'), unverified_cast, remove_commenting)

    dast_ts = load_veracity(os.path.join(dast_data_path, 'timestamps.csv'), unverified_cast, remove_commenting)
    dast_nts = load_veracity(os.path.join(dast_data_path, 'no_timestamps.csv'), unverified_cast, remove_commenting)
    return format_data(pheme_ts, pheme_nts, dast_ts, dast_nts, testdata_type)


def cross_validation_splits(dataset, no_splits):
    """Generate an array of dataset splits, each index containing a tuple with train data at 0 and test data at 1. Takes
    preprocessed veracity data, as generated by data_loader.load_veracity() as input.

    :param dataset: an array which, at each index, contains a tuple, with the veracity of a claim at 0 and an array
    containing the features at index 1
    :param no_splits: The number of partitions the dataset should be split into
    :return: an array of dimensions (no_splits x 2 (train data at 0 and test data at 1) x len(train) or len(test))
    """
    splits = []
    random.shuffle(dataset)

    # A few data-points might be lost due to rounding
    splitsize = int(len(dataset)/no_splits)
    last_loaded_index = 0
    for i in range(no_splits):
        test_data = dataset[last_loaded_index:last_loaded_index+splitsize]
        train_data = dataset[:last_loaded_index]
        train_data.extend(dataset[last_loaded_index+splitsize:])
        splits.append((train_data, test_data))
        last_loaded_index += splitsize
    return splits


def update_metrics(performance, model, acc, f1, branch_length):
    if branch_length:
        performance[model][branch_length]['f1_macro'] += f1
        performance[model][branch_length]['accuracy'] += acc
    else:
        performance[model]['f1_macro'] += f1
        performance[model]['accuracy'] += acc


def test_setup(pheme_data, testdata_type, empty_performance, model_type):
    performance = {x: copy.deepcopy(empty_performance) for x in ['pheme', 'dastpheme', 'dast', 'dast_majority',
                                                                 'pheme_majority', 'dastpheme_majority']}
    models = {}
    if testdata_type == 'dast':
        models['pheme'] = hmm_veracity.HMM(2, model_type).fit(pheme_data)
        models['pheme_majority'] = veracity_majority_baseline.VeracityMajorityBaseline().fit(pheme_data)
    return performance, models


def split_setup(split_index, testdata_type, dast_data, pheme_data, models, model_type):
    if testdata_type == 'dast':
        test_data = dast_data[split_index][1]
        models['dastpheme'] = hmm_veracity.HMM(2, model_type).fit(pheme_data + dast_data[split_index][0])
        models['dast'] = hmm_veracity.HMM(2, model_type).fit(dast_data[split_index][0])
        models['dast_majority'] = veracity_majority_baseline.VeracityMajorityBaseline().fit(dast_data[split_index][0])
        models['dastpheme_majority'] = veracity_majority_baseline.VeracityMajorityBaseline().fit(pheme_data + dast_data[split_index][0])
    else:
        test_data = dast_data[split_index][1] + pheme_data[split_index][1]
        models['pheme'] = hmm_veracity.HMM(2, model_type).fit(pheme_data[split_index][0])
        models['dastpheme'] = hmm_veracity.HMM(2, model_type).fit(pheme_data[split_index][0] + dast_data[split_index][0])
        models['dast'] = hmm_veracity.HMM(2, model_type).fit(dast_data[split_index][0])
        models['pheme_majority'] = veracity_majority_baseline.VeracityMajorityBaseline().fit(pheme_data[split_index][0])
        models['dast_majority'] = veracity_majority_baseline.VeracityMajorityBaseline().fit(dast_data[split_index][0])
        models['dastpheme_majority'] = veracity_majority_baseline.VeracityMajorityBaseline().fit(
            pheme_data[split_index][0] + dast_data[split_index][0])
    return models, test_data


def evaluate_for_splits_dataset(pheme_data, dast_data, unverified_cast, testdata_type, model_type):
    empty_performance = {'f1_macro': 0.0, 'accuracy': 0.0}
    performance, models = test_setup(pheme_data, testdata_type, empty_performance, model_type)

    for i in range(len(dast_data)):
        models, test_data = split_setup(i, testdata_type, dast_data, pheme_data, models, model_type)

        for model_name, model in models.items():
            _, acc, f1_macro, _ = model.test(test_data, unverified_cast)
            update_metrics(performance, model_name, acc, f1_macro, branch_length=False)

    for dataset, results in performance.items():
        for metric, value in results.items():
            performance[dataset][metric] = value / len(dast_data)

    return performance


def evaluate_for_splits_length(pheme_data, dast_data, unverified_cast, testdata_type, model_type):
    empty_performance = {x: {'f1_macro': 0.0, 'accuracy': 0.0} for x in [1, 2, 3, 4, 6, 8, 10]}
    performance, models = test_setup(pheme_data, testdata_type, empty_performance, model_type)

    for i in range(len(dast_data)):
        length_separated_data = {1: [], 2: [], 3: [], 4: [], 6: [], 8: [], 10: []}

        models, test_data = split_setup(i, testdata_type, dast_data, pheme_data, models, model_type)

        for branch in test_data:
            if len(branch[1]) in length_separated_data:
                length_separated_data[len(branch[1])].append(branch)
            elif len(branch[1]) == 5:
                length_separated_data[4].append(branch)
            elif len(branch[1]) == 7:
                length_separated_data[6].append(branch)
            elif len(branch[1]) == 9:
                length_separated_data[8].append(branch)
            elif len(branch[1]) >= 10:
                length_separated_data[10].append(branch)

        for length, datapoints in length_separated_data.items():
            if datapoints:
                for model_name, model in models.items():
                    _, acc, f1_macro, _ = model.test(datapoints, unverified_cast)
                    update_metrics(performance, model_name, acc, f1_macro, branch_length=length)

    for dataset, results in performance.items():
        for length, metrics in results.items():
            for metric, value in metrics.items():
                performance[dataset][length][metric] = value / len(dast_data)

    return performance


def write_out(include_branch_length, performance, out_path='veracity.csv'):
    with open(out_path, mode='w', encoding='utf-8') as out_file:
        if include_branch_length:
            out_file.write('model;length;f1_macro;accuracy\n')
            for dataset, results in performance[0].items():
                for length, metrics in results.items():
                    out_file.write('{};{};{:.2f};{:.2f}\n'.format(dataset + '_ts', length, results[length]['f1_macro'],
                                                                  results[length]['accuracy']))
            if len(performance) > 1:
                for dataset, results in performance[1].items():
                    for length, metrics in results.items():
                        if 'majority' in dataset:
                            continue
                        out_file.write('{};{};{:.2f};{:.2f}\n'.format(dataset, length, results[length]['f1_macro'],
                                                                  results[length]['accuracy']))

        else:
            out_file.write('model;f1_macro;accuracy\n')
            for dataset, results in performance[0].items():
                out_file.write('{};{:.2f};{:.2f}\n'.format(dataset + '_ts', results['f1_macro'], results['accuracy']))
            if len(performance) > 1:
                for dataset, results in performance[1].items():
                    if 'majority' in dataset:
                        continue
                    out_file.write('{};{:.2f};{:.2f}\n'.format(dataset, results['f1_macro'], results['accuracy']))


def evaluate_performance(unverified_cast, remove_commenting, include_branch_length=False, testdata_type='dast', model_type='gaussian'):
    pheme_ts, pheme_nts, dast_ts, dast_nts = load_datasets(unverified_cast, testdata_type, remove_commenting)
    performance = []
    if include_branch_length:
        nts_performance = evaluate_for_splits_length(pheme_nts, dast_nts, unverified_cast, testdata_type, model_type)
        performance.append(nts_performance)
        if model_type in ['gaussian']:
            ts_performance = evaluate_for_splits_length(pheme_ts, dast_ts, unverified_cast, testdata_type, model_type)
            performance.append(ts_performance)
    else:
        nts_performance = evaluate_for_splits_dataset(pheme_nts, dast_nts, unverified_cast, testdata_type, model_type)
        performance.append(nts_performance)
        if model_type in ['gaussian']:
            ts_performance = evaluate_for_splits_dataset(pheme_ts, dast_ts, unverified_cast, testdata_type, model_type)
            performance.append(ts_performance)
    write_out(include_branch_length, performance)


evaluate_performance(unverified_cast='true', remove_commenting=True, include_branch_length=False,
                     testdata_type='dastpheme', model_type='multinomial')
