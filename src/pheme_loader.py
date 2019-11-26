# The following import should be removed once PEP 563 becomes the default
# https://www.python.org/dev/peps/pep-0563/
import json
import os
from pathlib import Path

default_pheme_path = Path(os.path.join(os.path.abspath(__file__), '../../data/datasets/pheme/raw/'))


def create_annotation_dict(annotation_file):
    sdqc_dict = {"agreed": "Supporting",
                 "disagreed": "Denying",
                 "appeal-for-more-information": "Querying",
                 "comment": "Commenting"}

    annotations = {}
    reply_tweets = False
    for line in annotation_file:
        line = line.strip()
        if reply_tweets:
            if line != "# Deep Replies":
                annotation = json.loads(line)
                annotations[annotation['tweetid']] = sdqc_dict[annotation['responsetype-vs-source']]
        elif line == "# Direct Replies":
            reply_tweets = True
    return annotations


def load_annotations(pheme_path, language):
    annotation_dir = pheme_path / "annotations"
    for reaction_file in annotation_dir.glob("*.json"):
        if str(reaction_file).split('\\')[-1].split('-')[0] == language:
            annotation_file = reaction_file.open(mode='r', encoding='utf-8')
            return create_annotation_dict(annotation_file)

    err_msg = "No annotation file found in {} for the given language {}"
    raise RuntimeError(
        err_msg.format(pheme_path, language))


def read_tweet(tweet_file: Path):
    with tweet_file.open(mode='r', encoding='utf-8') as f:
        tweet = json.load(f)
        return tweet


def read_reactions(reaction_directory):
    reactions = {}
    for reaction_file in reaction_directory.glob("*.json"):
        reaction = read_tweet(reaction_file)
        reactions[reaction['id_str']] = reaction
    return reactions


def read_source(source_directory):
    if 0 == len(os.listdir(source_directory)) > 1:
        err_msg = "Source tweet folder at {} contains {} source tweet(s). Should contain 1."
        raise RuntimeError(
            err_msg.format(source_directory, len(os.listdir(source_directory))))

    for source_file in source_directory.glob("*.json"):
        return read_tweet(source_file)


def read_conversation(conversation_folder):
    source_dir = conversation_folder / 'source-tweets'
    reaction_dir = conversation_folder / 'reactions'
    source = read_source(source_dir)
    reactions = read_reactions(reaction_dir)
    return source, reactions


def append_annotations(reactions, annotations):
    for id_str, reaction in reactions.items():
        if id_str not in annotations:
            reaction['SDQC_Submission'] = 'Underspecified'
        else:
            reaction['SDQC_Submission'] = annotations[id_str]
    return reactions


def explore(tweet_id, current_branch, reactions, tree, branch_structure):
    current_tweet = reactions[tweet_id]
    if not branch_structure[str(tweet_id)]:
        current_branch.append(current_tweet)
        tree.append(current_branch)
    else:
        for reply in branch_structure[tweet_id]:
            new_branch = current_branch[:]
            new_branch.append(current_tweet)
            explore(reply, new_branch, reactions, tree, branch_structure[tweet_id])


def apply_tree_structure(source, reactions, conversation_folder):
    structure_file = conversation_folder / 'structure.json'
    branch_structure = json.load(structure_file.open(mode='r', encoding='utf-8'))
    tree = [source]
    for reply in branch_structure[source['id_str']]:
        branch = []
        explore(reply, branch, reactions, tree, branch_structure[source['id_str']])
    return tree


def read_all_tweets(base_directory: Path):
    veracity_translator = {'0': 'False', '1': 'True', '2': 'Unverified'}
    languages = ['en', 'de']
    annotations = {}
    data = []
    for language in languages:
        annotations.update(load_annotations(base_directory, language))
    for language in languages:
        thread_path = Path(base_directory) / "threads" / language
        with open(Path(base_directory / 'rumour_overview.txt'), mode='r', encoding='utf-8') as veracity_overview:
            veracity_dict = {}
            for line in veracity_overview:
                veracity_dict[line.split('\t')[0]] = line.replace('\n', '').split('\t')[1]
            for rumour_folder in thread_path.iterdir():
                for conversation_folder in rumour_folder.iterdir():
                    folder_dir = thread_path / rumour_folder / conversation_folder
                    source, reactions = read_conversation(folder_dir)
                    reactions = append_annotations(reactions, annotations)
                    # Exclude branches with no annotated datapoints
                    source['TruthStatus'] = veracity_translator[veracity_dict[source['id_str']]]
                    data.append(apply_tree_structure(source, reactions, conversation_folder))
    return data


def read_pheme(path=default_pheme_path):
    tweets = read_all_tweets(Path(path))
    return tweets


def generate_veracity_overview(path=default_pheme_path):
    veracity_dict = {}
    for language_dir in Path(path / 'threads').iterdir():
        for rumour_folder in language_dir.iterdir():
            for conversation_folder in rumour_folder.iterdir():
                conversation_folder_path = path / 'threads' / rumour_folder / conversation_folder
                source_folder_path = conversation_folder_path / 'source-tweets'
                if 0 == len(os.listdir(source_folder_path)) > 1:
                    err_msg = 'Source tweet folder at {} contains {} source tweet(s). Should contain 1.'
                    raise RuntimeError(
                        err_msg.format(source_folder_path, len(os.listdir(source_folder_path))))

                for source_file in source_folder_path.glob("*.json"):
                    source_id = str(source_file).split('\\')[-1].split('.json')[0]

                veracity_annotation = json.load(open(Path(conversation_folder_path / 'annotation.json'), mode='r', encoding='utf-8'))
                if 'true' in veracity_annotation and veracity_annotation['true'] == '1':
                    veracity = 1
                elif veracity_annotation['misinformation'] == '1':
                    veracity = 0
                # Rumor is unverified
                else:
                    veracity = 2
                veracity_dict[source_id] = veracity

    with open(Path(path / 'rumour_overview.txt'), mode='w', encoding='utf-8') as rumour_overview:
        rumour_overview.write('IDs\tVeracity\n')
        for source_id, veracity in veracity_dict.items():
            rumour_overview.write('{}\t{}\n'.format(source_id, veracity))
