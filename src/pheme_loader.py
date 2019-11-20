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

    err_msg = "No annotation file found for the given language {}"
    raise RuntimeError(
        err_msg.format(language))


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
    if len(os.listdir(source_directory)) > 1 > 0:
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


def read_all_tweets(base_directory: Path, language):
    annotations = load_annotations(base_directory, language)
    data = []
    for rumour_folder in base_directory.iterdir():
        for conversation_folder in rumour_folder.iterdir():
            folder_dir = base_directory / rumour_folder / conversation_folder
            source, reactions = read_conversation(folder_dir)
            reactions = append_annotations(reactions, annotations)
            data.append(apply_tree_structure(source, reactions, conversation_folder))
    return data


def read_pheme(path=default_pheme_path, language="en"):
    pheme_path = path / "threads" / language
    tweets = read_all_tweets(pheme_path, language)
    return tweets


def generate_veracity_overview(path=default_pheme_path, languages='en'):

    with open(path / 'rumour_overview.txt') as rumour_overview:

