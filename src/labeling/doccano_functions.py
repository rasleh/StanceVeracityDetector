import json
import os
from pathlib import Path

test_data = Path(os.path.join(os.path.abspath(__file__), '../../../data/datasets/twitter/raw/2019-12-05.txt'))


def generate_label_data(in_path: Path, stance_out_path: Path, claim_out_path: Path):
    with in_path.open() as in_file:
        with stance_out_path.open(mode='w') as stance_out_file:
            with claim_out_path.open(mode='w') as claim_out_file:
                for line in in_file:
                    tweet_dict = json.loads(line.split('\t')[1])
                    source_tweet = tweet_dict[line.split('\t')[0]]
                    source_tweet['text'] = source_tweet['full_text']
                    source_tweet['labels'] = []
                    json.dump(source_tweet, claim_out_file)
                    claim_out_file.write('\n')
                    for tweet_id, tweet in tweet_dict.items():
                        if source_tweet == tweet:
                            continue
                        tweet['text'] = 'Source: {}\n\nReply: {}'.format(source_tweet['full_text'], tweet['full_text'])
                        tweet['labels'] = []
                        json.dump(tweet, stance_out_file)
                        stance_out_file.write('\n')


def anno_agreement_check(anno_data_path: Path, agree_path: Path, disagree_path: Path):
    with anno_data_path.open(encoding='utf-8') as anno_data, agree_path.open(mode='w', encoding='utf-8') as agree_data, disagree_path.open(
            mode='w', encoding='utf-8') as disagree_data:
        for line in anno_data:
            print(line)
            disagreement = False
            annotations = json.loads(line)['annotations']
            if len(annotations) == 1:
                agree_data.write(line)
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
                agree_data.write(line)


def integrate_label_data(labeled_data_path: Path, database_path: Path):
    with labeled_data_path.open() as labeled_data:
        with database_path.open() as database:
            for line in database:
                tweet_dict = json.loads(line.split('\t')[1])


anno_agreement_check(Path('test.json'), Path('agree.json'), Path('disagree.json'))

#generate_label_data(test_data, Path('stance.jsonl'), Path('claim.jsonl'))
