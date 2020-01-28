import json
import os
from pathlib import Path

current_path = os.path.abspath(__file__)
default_raw_path = os.path.join(current_path, '../../data/datasets/twitter/raw/')
unlabeled_data_path = Path(os.path.join(os.path.abspath(__file__), '../../../data/datasets/twitter/raw/unlabeled'))


def generate_label_data(file_name: str, stance_out_name: str, claim_out_name: str):
    file_path = Path(unlabeled_data_path, file_name)
    stance_out_path = Path(unlabeled_data_path, stance_out_name)
    claim_out_path = Path(unlabeled_data_path, claim_out_name)
    with file_path.open() as data, stance_out_path.open(mode='w') as stance_out, claim_out_path.open(mode='w') as claim_out:
        for line in data:
            tweet_dict = json.loads(line.split('\t')[1])
            source_tweet = tweet_dict[line.split('\t')[0]]
            source_tweet['text'] = source_tweet['full_text']
            source_tweet['labels'] = []
            json.dump(source_tweet, claim_out)
            claim_out.write('\n')
            for tweet_id, tweet in tweet_dict.items():
                if source_tweet == tweet:
                    continue
                tweet['text'] = 'Source: {}\n\nReply: {}'.format(source_tweet['full_text'], tweet['full_text'])
                tweet['labels'] = []
                json.dump(tweet, stance_out)
                stance_out.write('\n')


def anno_agreement_check(anno_data_file: str, agree_file: str, disagree_file: str):
    anno_data_path = Path(os.path.join(default_raw_path, anno_data_file))
    agree_path = Path(os.path.join(default_raw_path, agree_file))
    disagree_path = Path(os.path.join(default_raw_path, disagree_file))

    with anno_data_path.open(encoding='utf-8') as anno_data, agree_path.open(mode='w', encoding='utf-8') as agree_data, disagree_path.open(
            mode='w', encoding='utf-8') as disagree_data:
        for line in anno_data:
            disagreement = False
            annotations = json.loads(line)['annotations']
            if len(annotations) == 1:
                line = json.loads(line)
                line['annotations'] = [annotations[0]['label']]
                json.dump(line, agree_data)
                agree_data.write('\n')
            else:
                user_labels = {}
                for annotation in annotations:
                    user_labels.setdefault(annotation['user'], set()).add(annotation['label'])
                for user_id_a, labels_a in user_labels.items():
                    for user_id_b, labels_b in user_labels.items():
                        if labels_a != labels_b:
                            disagree_data.write(line)
                            disagreement = True
                            break
                    if disagreement:
                        break
                if not disagreement:
                    line = json.loads(line)
                    if user_labels:
                        line['annotations'] = list(user_labels[1])
            if not disagreement:
                print(line)
                json.dump(line, agree_data)
                agree_data.write('\n')


def integrate_claim_label(annotation, tweet):
    veracity_map = {5: 'True', 6: 'Unverified', 7: 'False'}
    if 1 or 2 not in annotation['annotations']:
        err_msg = "Error in claim labels, must contain either '1' or '2', denominating 'claim'" \
                  " and 'non-claim' respectively. Given labels: {}"
        raise RuntimeError(
            err_msg.format(annotation['annotations']))
    if 2 in annotation['annotations']:
        tweet['Claim'] = False
    else:
        tweet['Claim'] = True
        if 3 or 4 not in annotation['annotations']:
            err_msg = "Error in claim labels, must contain either '3' or '4', denominating " \
                      "'verifiable' and 'subjective' respectively. Given labels: {}"
            raise RuntimeError(
                err_msg.format(annotation['annotations']))
        if 4 in annotation['annotations']:
            tweet['Verifiability'] = 'Subjective'
        else:
            tweet['Verifiability'] = 'Verifiable'
            if 5 or 6 or 7 not in annotation['annotations']:
                err_msg = "Error in claim labels, must contain either '5', '6' or '7', " \
                          "denominating 'True', 'Unverified' and 'False' respectively. Given " \
                          "labels: {}"
                raise RuntimeError(
                    err_msg.format(annotation['annotations']))
            for x in [5, 6, 7]:
                if x in annotation['annotations']:
                    tweet['TruthStatus'] = veracity_map[x]


def integrate_sdqc_label(annotation, tweet):
    sdqc_map = {1: 'Supporting', 2: 'Denying', 3: 'Querying', 4: 'Commenting'}
    if len(annotation['annotations']) > 1:
        err_msg = "{} SDQC labels found, only one allowed"
        raise RuntimeError(
            err_msg.format(len(annotation['annotations'])))
    tweet['SDQC_Submission'] = sdqc_map[annotation['annotations'][0]]


def integrate_label_data(anno_data_path: Path, database_path: Path, label_scheme: str):
    if label_scheme not in ['claim', 'sdqc']:
        err_msg = "Unrecognized label scheme: {}, please use 'sdqc' or 'claim'"
        raise RuntimeError(
            err_msg.format(label_scheme))
    with anno_data_path.open(encoding='utf-8') as labeled_data, database_path.open(encoding='utf-8') as database:
        data = []
        for line in database:
            not_altered = True
            tweet_dict = json.loads(line.split('\t')[1])
            for annotation in labeled_data:
                annotation = json.loads(annotation)
                # Data-point not yet annotated
                if not annotation['annotations']:
                    continue
                for tweet_id, tweet in tweet_dict.items():
                    if tweet['full_text'] == annotation['text']:
                        if label_scheme == 'claim':
                            integrate_claim_label(annotation, tweet)
                        if label_scheme == 'sdqc':
                            integrate_sdqc_label(annotation, tweet)
                        not_altered = False
                        break
            if not_altered:
                data.append(line)
            else:
                data.append(line.split('\t')[0] + '\t' + json.dumps(tweet_dict))

    with database_path.open(mode='w', encoding='utf-8') as database:
        for line in data:
            database.write(line)


#anno_agreement_check(Path('test.json'), Path('agree.json'), Path('disagree.json'))
#generate_label_data(test_data, 'stance.jsonl', 'claim.jsonl')
