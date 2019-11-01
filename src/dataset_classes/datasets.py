import datetime
import os
import random
import re
from pathlib import Path

from afinn import Afinn
from nltk import word_tokenize
from sklearn.model_selection import train_test_split

from src.feature_extraction import word_embeddings

url_tag = 'urlurlurl '
regex_url = re.compile('(https://t\\.co/*?)')
ref_tag = 'refrefref '
regex_ref = re.compile('(@.*? )')
punctuation = re.compile('[^a-zA-ZæøåÆØÅ0-9]')
rand = random.Random(42)
afinn = Afinn(language='da', emoticons=True)


def compute_similarity(annotation, previous, source, branch_tokens, is_source=False):
    # TODO: exclude itself???
    annotation.sim_to_branch = word_embeddings.cosine_similarity(annotation.tokens, branch_tokens)
    if not is_source:
        annotation.sim_to_src = word_embeddings.cosine_similarity(annotation.tokens, source.tokens)
        annotation.sim_to_prev = word_embeddings.cosine_similarity(annotation.tokens, previous.tokens)


def read_lexicon(file_path):
    """Loads lexicon file given path. Assumes file has one word per line"""
    file_path = os.path.join(os.path.abspath(__file__), Path(file_path))
    with open(file_path, "r", encoding='utf8') as lexicon_file:
        return set([line.strip().lower() for line in lexicon_file.readlines()])


def count_lexicon_occurence(words, lexion):
    return sum([1 if word in lexion else 0 for word in words])


def tokenize(text):
    # Convert all words to lower case and tokenize
    text_tokens = word_tokenize(text.lower(), language='danish')
    tokens = []
    # Remove non-alphabetic characters, not contained in abbreviations
    for token in text_tokens:
        if not punctuation.match(token):
            tokens.append(token)
    return tokens


class Annotation:
    def __init__(self, data):
        # Tweet-specific information
        self.id = data['id_str']
        self.is_source = True if data['in_reply_to_status_id'] is None else False
        self.children = data['children']
        self.created = datetime.datetime.strptime(data['created_at'], '%a %b %d %H:%M:%S %z %Y').replace(tzinfo=None)
        self.upvotes = data['favorite_count']
        self.reply_count = len(self.children)
        if not self.is_source:
            self.parent_id = data["in_reply_to_status_id"]

        # SDQC info - is either annotated or contains placeholder values named "Underspecified"
        self.sdqc_parent = data['SDQC_Parent']
        self.sdqc_submission = data['SDQC_Submission']
        self.sdqc_source = data['SourceSDQC']

        # user info
        self.user_id = data["user"]["id"]
        self.user_name = data["user"]["screen_name"]
        self.user_created = data["user"]["created_at"]

        # Extract text, filter out URLs and references and extract tokens from text
        self.text = data['full_text'].replace('\n', '')
        self.text = self.filter_text_urls(self.text)
        self.text = self.filter_text_ref(self.text)
        self.tokens = word_tokenize(self.text.lower())

        # Placeholder values for cosine similarity calculation
        self.sim_to_src = 0
        self.sim_to_prev = 0
        self.sim_to_branch = 0

    def filter_text_ref(self, text):
        """filters text of all annotations to replace twitter references with the tag 'refrefref'"""
        return regex_ref.sub(ref_tag, text)

    def filter_text_urls(self, text):
        """filters text of all annotations to replace urls with the tag'URLURLURL'"""
        return regex_url.sub(url_tag, text)


class SourceSubmission:
    def __init__(self, source):
        self.source = source
        self.branches = []

    def add_annotation_branch(self, annotation_branch):
        """Add a branch as a list of annotations to this submission"""
        self.branches.append(annotation_branch)


class DataSet:
    def __init__(self):
        self.annotations = {}
        self.anno_to_branch_tokens = {}
        self.anno_to_prev = {}
        self.anno_to_source = {}
        self.submissions = []
        self.last_submission = lambda: len(self.submissions) - 1
        # mapping from property to tuple: (min, max)
        self.min_max = {
            'txt_len': [0, 0],
            'tokens_len': [0, 0],
            'avg_word_len': [0, 0],
            'upvotes': [0, 0],
            'reply_count': [0, 0],
            'afinn_score': [0, 0],
            'url_count': [0, 0],
            'quote_count': [0, 0],
            'cap_sequence_max_len': [0, 0],
            'tripDotCount': [0, 0],
            'q_mark_count': [0, 0],
            'e_mark_count': [0, 0],
            'cap_count': [0, 0],
            'swear_count': [0, 0],
            'negation_count': [0, 0],
            'positive_smiley_count': [0, 0],
            'negative_smiley_count': [0, 0]
        }
        self.min_i = 0
        self.max_i = 1
        self.karma_max = 0
        self.karma_min = 0
        # dictionary at idx #num is used for label #num, example: support at 0
        self.freq_histogram = [dict(), dict(), dict(), dict()]
        self.unique_freq_histogram = {}
        self.bow = set()
        self.freq_tri_gram = [dict(), dict(), dict(), dict()]
        self.sdqc_to_int = {
            "Supporting": 0,
            "Denying": 1,
            "Querying": 2,
            "Commenting": 3
        }

        self.positive_smileys = read_lexicon('../../../data/featurization/lexicon/positive_smileys.txt')
        self.negative_smileys = read_lexicon('../../../data/featurization/lexicon/negative_smileys.txt')
        self.swear_words = read_lexicon('../../../data/featurization/lexicon/swear_words.txt')
        swear_words_en = read_lexicon('../../../data/featurization/lexicon/swear_words_en.txt')
        for word in swear_words_en:
            self.swear_words.add(word)
        self.negation_words = read_lexicon('../../../data/featurization/lexicon/negation_words.txt')

    def add_annotation(self, annotation):
        """Add to self.annotations. Should only be uses for testing purposes"""
        annotation = self.analyse_annotation(Annotation(annotation))
        if annotation.id not in self.annotations:
            self.annotations[annotation.id] = annotation

    def add_submission(self, source):
        self.submissions.append(SourceSubmission(Annotation(source)))

    def create_annotation(self, annotation):
        return Annotation(annotation)

    def add_branch(self, branch, sub_sample=False):
        annotation_branch = []
        branch_tokens = []
        class_comments = 0
        # First, convert to Python objects
        for annotation in branch:
            annotation = self.create_annotation(annotation)
            if not annotation.sdqc_submission == 'Underspecified' and self.sdqc_to_int[annotation.sdqc_submission] == 3:
                class_comments += 1
            branch_tokens.extend(annotation.tokens)
            annotation_branch.append(annotation)

        # Filter out branches with pure commenting class labels
        if sub_sample and class_comments == len(branch):
            return

        # Compute cosine similarity
        source = self.submissions[self.last_submission()].source
        prev = source
        for annotation in annotation_branch:
            if annotation.id not in self.annotations:  # Skip repeated annotations
                compute_similarity(annotation, prev, source, branch_tokens)
                self.analyse_annotation(annotation)  # Analyse relevant annotations
                self.annotations[annotation.id] = annotation
                self.anno_to_branch_tokens[annotation.id] = branch_tokens
                self.anno_to_prev[annotation.id] = prev
                self.anno_to_source[annotation.id] = source
            prev = annotation
        self.submissions[self.last_submission()].add_annotation_branch(annotation_branch)

    def analyse_annotation(self, annotation):
        if not annotation:
            return
        self.handle(self.min_max['txt_len'], len(annotation.text))
        self.handle(self.min_max['afinn_score'], afinn.score(annotation.text))
        self.handle(self.min_max['url_count'], annotation.tokens.count('urlurlurl'))
        self.handle(self.min_max['quote_count'], annotation.tokens.count('refrefref'))
        self.handle(self.min_max['cap_sequence_max_len'],
                    len(max(re.findall(r"[A-ZÆØÅ]+", annotation.text), key=len, default='')))
        self.handle(self.min_max['tripDotCount'], annotation.text.count('...'))
        self.handle(self.min_max['q_mark_count'], annotation.text.count('?'))
        self.handle(self.min_max['e_mark_count'], annotation.text.count('!'))
        self.handle(self.min_max['cap_count'], sum(1 for c in annotation.text if c.isupper()))
        self.handle(self.min_max['swear_count'], count_lexicon_occurence(annotation.tokens, self.swear_words))
        self.handle(self.min_max['negation_count'], count_lexicon_occurence(annotation.tokens, self.negation_words))
        self.handle(self.min_max['positive_smiley_count'], count_lexicon_occurence(annotation.text.split(),
                                                                                   self.positive_smileys))
        self.handle(self.min_max['negative_smiley_count'], count_lexicon_occurence(annotation.text.split(),
                                                                                   self.negative_smileys))

        word_len = len(annotation.tokens)
        if not word_len == 0:
            self.handle(self.min_max['tokens_len'], word_len)
            self.handle(self.min_max['avg_word_len'],
                        sum([len(word) for word in annotation.tokens]) / word_len)
        self.handle(self.min_max['upvotes'], annotation.upvotes)
        self.handle(self.min_max['reply_count'], annotation.reply_count)
        return annotation

    def handle(self, entries, prop):
        if prop > entries[self.max_i]:
            entries[self.max_i] = prop
        if prop < entries[self.min_i] or entries[self.min_i] == 0:
            entries[self.min_i] = prop

    def get_min(self, key):
        return self.min_max[key][self.min_i]

    def get_max(self, key):
        return self.min_max[key][self.max_i]

    def iterate_annotations(self):
        for anno_id, annotation in self.annotations.items():
            yield annotation

    def iterate_branches(self, with_source=True):
        for source_tweet in self.submissions:
            for branch in source_tweet.branches:
                if with_source:
                    yield source_tweet.source, branch
                else:
                    yield branch

    def iterate_submissions(self):
        for source_tweet in self.submissions:
            yield source_tweet

    def size(self):
        return len(self.annotations)

    def train_test_split(self, test_size=0.25, rand_state=42, shuffle=True, stratify=True):
        x = []
        y = []
        for annotation in self.iterate_annotations():
            x.append(annotation)
            y.append(self.sdqc_to_int[annotation.sdqc_submission])
        print('Splitting with test size', test_size)
        x_train, x_test, _, _ = train_test_split(
            x, y, test_size=test_size, random_state=rand_state, shuffle=shuffle, stratify=(y if stratify else None)
        )
        print('Train stats:')
        self.print_status_report(x_train)
        print('Test stats:')
        self.print_status_report(x_test)
        return x_train, x_test

    def print_status_report(self, annotations=None):
        histogram = {
            0: 0,
            1: 0,
            2: 0,
            3: 0
        }
        ix_to_sdqc = {0: 'S', 1: 'D', 2: 'Q', 3: 'C'}
        n = 0
        for annotation in (self.iterate_annotations() if not annotations else annotations):
            histogram[self.sdqc_to_int[annotation.sdqc_submission]] += 1
            n += 1
        print('Number of data points:', n)
        print('SDQC distribution:')
        for label, count in histogram.items():
            print('{}: {:4d} ({:.3f})'.format(ix_to_sdqc[label], count, float(count) / float(n)))
