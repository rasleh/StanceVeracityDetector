import json
import os
from pathlib import Path

test_data = Path(os.path.join(os.path.abspath(__file__), '../../../data/datasets/twitter/raw/2019-12-05.txt'))


def generate_label_data(data_path: Path, stance_out_path: Path, claim_out_path: Path):
    with data_path.open() as data, stance_out_path.open(mode='w') as stance_out, claim_out_path.open(mode='w') as claim_out:
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


def anno_agreement_check(anno_data_path: Path, agree_path: Path, disagree_path: Path):
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


def integrate_label_data(anno_data_path: Path, database_path: Path, label_scheme: str):
    sdqc_mapping = {1: ''}
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
                for tweet_id, tweet in tweet_dict.items():
                    if tweet['full_text'] == annotation['text']:
                        if label_scheme == 'claim':

                        if label_scheme == 'sdqc':
                            if len(annotation['annotations']) != 1:
                                err_msg = "{} SDQC labels found, only one allowed"
                                raise RuntimeError(
                                    err_msg.format(len(annotation['annotations'])))

                        not_altered = False
            if not_altered:
                data.append(line)
            else:
                data.append(line.split('\t')[0] + '\t' + json.dumps(tweet_dict))

    with database_path.open(mode='w', encoding='utf-8') as database:
        for line in data:
            database.write(line)

anno_agreement_check(Path('test.json'), Path('agree.json'), Path('disagree.json'))

#generate_label_data(test_data, Path('stance.jsonl'), Path('claim.jsonl'))
