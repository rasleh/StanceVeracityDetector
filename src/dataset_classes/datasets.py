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
    """
    Computes the cosine similarity between the word embeddings of a given text and:
    1) The parent text in the conversational tree
    2) The root node text in the conversational tree
    3) The averaged word embeddings across a full conversational branch
    These values are saved in the Annotation object

    :param annotation: an object of the Annotation class, for which knowledge regarding cosine similarity is desired
    :param previous: the parent text in the conversational tree
    :param source: the root node text in the conversational tree
    :param branch_tokens: nltk-generated word tokens for each text in the conversational branch
    :param is_source: whether or not the given Annotation is the root node (source) itself
    """
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
    """Counts the number of words in 'words' which occur in 'lexicon' and returns the result"""
    return sum([1 if word in lexion else 0 for word in words])


def tokenize(text):
    """Converts a given text 'text' into NLTK word tokens and returns these as an array"""
    # Convert all words to lower case and tokenize
    text_tokens = word_tokenize(text.lower(), language='danish')
    tokens = []
    # Remove non-alphabetic characters, not contained in abbreviations
    for token in text_tokens:
        if not punctuation.match(token):
            tokens.append(token)
    return tokens


class Annotation:
    """
        Class for representing texts, to be stored in an object of the DataSet class or a child class inheriting from
        DataSet. Meant to be extended for future inclusions of new data-sources into the StanceVeracityDetector project.

        Attributes
        text : str
            textual content of a comment
        is_source : boolean
            whether the comment is at the root of the conversation tree
        id : str
            id of the comment
        children : array
            array containing IDs of all child nodes in a conversational structure
        sdqc_source = str
            the label of the root of the conversation tree as either supporting, denying, querying or commenting
        sdqc_parent : str
            the label of the parent comment in  the conversation tree as either supporting, denying, querying or commenting
        sdqc_submission : str
            the label of the comment as either supporting, denying, querying or commenting
        tokens : array
            the comment text tokenized using the nltk package
        parent_id : str
            the id of the parent comment in the conversation tree
        sim_to_src : float
            cosine similarity between the average word embeddings of the comment and the source of the conversation tree
        sim_to_prev : float
            cosine similarity between the average word embeddings of the comment and parent comment in the conversation tree
        sim_to_branch : float
            cosine similarity between the average word embeddings of the comment and the full conversation branch
        created : datetime
            date and time of comment submission

        Methods
        filter_text_ref(text)
            replaces references with a reference tag in comment text and returns text
        filter_text_url(text)
            replaces URL with a URL tag in comment text and returns text
        """

    def __init__(self, data):
        # Tweet-specific information
        self.id = data['id_str']
        self.is_source = True if data['in_reply_to_status_id'] is None else False
        self.children = data['children']
        self.created = datetime.datetime.strptime(data['created_at'], '%a %b %d %H:%M:%S %z %Y').replace(tzinfo=None)
        if not self.is_source:
            self.parent_id = data["in_reply_to_status_id"]

        # SDQC info - is either annotated or contains placeholder values named "Underspecified"
        self.sdqc_parent = data['SDQC_Parent']
        self.sdqc_submission = data['SDQC_Submission']
        self.sdqc_source = data['SourceSDQC']

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
        """filters text of all annotations to replace references with the tag 'refrefref'"""
        return regex_ref.sub(ref_tag, text)

    def filter_text_urls(self, text):
        """filters text of all annotations to replace urls with the tag'URLURLURL'"""
        return regex_url.sub(url_tag, text)


class SourceSubmission:
    """
    Class for representing texts at the root of a given conversational structure, and the branches connected to the
    source.

    Attributes
    source : Annotation
        the Annotation object at the source of the conversation tree
    branches : array
        an array of arrays of Annotation objects, each array containing a single branch stemming from the source

    Methods
    add_annotation_branch(annotation_branch)
        adds a branch, in the form of an array, to the list of branches currently connected to the SourceSubmission
    """

    def __init__(self, source: Annotation):
        self.source = source
        self.branches = []

    def add_annotation_branch(self, annotation_branch):
        """Add a branch as a list of annotations to this submission"""
        self.branches.append(annotation_branch)


class DataSet:
    """
        Class for representing a dataset of texts, as generated by the Annotation class.

        Attributes
        annotations : dict
            dictionary connecting text IDs to their corresponding Annotation objects
        anno_to_branch_tokens : dict
            dictionary connecting text IDs to NLTK tokens in the conversation branch, for easier calculation of cosine
            similarity between text and branch
        anno_to_prev : dict
            dictionary connecting text IDs to NLTK tokens in the previous comment, for easier calculation of cosine
            similarity between text and previous comment
        anno_to_source : dict
            dictionary connecting text IDs to NLTK tokens in the source comment of the conversation tree, for
            easier calculation of cosine similarity between text and source comment
        submissions : array
            array of SourceSubmission objects representing the conversation trees in the dataset
        last_submission : pointer
            pointer targeted at the last added submission to the dataset
        min_max : dict
            dictionary of minimum and maximum values for a number of text features used for normalization
        min_i, max_i : int, int
            used as pointers for referring to minimum and maximum value locations in min_max
        sdqc_to_int : dict
            dictionary for translating SDQC labels to ints
        positive_smileys, negative_smileys, swear_words, negation_words : set, set, set, set
            sets used for creating features of the presence of positive_smileys, negative_smileys, swear_words and
            negation_words, created using the read_lexicon function

        Methods
        add_annotation(annotation)
            creates an Annotation object and adds it to the DataSet's annotations dict
        add_submission(source)
            creates an Annotation object, uses this to create a SourceSubmission object and adds this to the DataSet's
            submissions array
        create_annotation(annotation)
            creates an Annotation object
        add_branch(branch, sub_sample=False)
            converts an array of data into Annotation objects, adds objects to the relevant dicts in the DataSet, and
            adds the branch of Annotation objects to the latest submission
        analyze_annotation(annotation)
            updates minimum and maximum values in min_max based on the given Annotation object
        handle(entries, prop)
            checks the minimum and maximum values in min_max for a given entry against a property in a given annotation
        get_min(key)
            returns the minimum value for a given property
        get_max(key)
            returns the maximum value for a given property
        iterate_annotations()
            iterates over the Annotation objects in the dataset
        iterate_branches(with_source=True)
            iterates over the branches in the dataset. Also returns the source tweet, if so specified
        iterate_submissions()
            iterates over the submissions in the dataset
        size()
            returns the number of annotations added to the dataset
        train_test_split(test_size=0.25, rand_state=42, shuffle=True, stratify=True)
            creates a split of the dataset, using sklearn's test_train_split method
        print_status_report(annotations=None)
            prints a histogram of label distributions
    """

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
        # dictionary at idx #num is used for label #num, example: support at 0
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
        """Creates an Annotation object, uses this to create a SourceSubmission object and adds this to the DataSet's
        submissions array"""
        self.submissions.append(SourceSubmission(Annotation(source)))

    def create_annotation(self, annotation):
        """Creates an Annotation object"""
        return Annotation(annotation)

    def add_branch(self, branch, sub_sample=False):
        """Converts an array of data into Annotation objects, adds objects to the relevant dicts in the DataSet, and
        adds the branch of Annotation objects to the latest submission

        :param branch: array containing comment data
        :param sub_sample: if True, will not add branch to dataset if it only containg comments with a 'commenting'
        SDQC label
        """
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
        """
        updates minimum and maximum values in min_max based on the given Annotation object

        :param annotation: an object of the Annotation class
        """

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
        return annotation

    def handle(self, entries, prop):
        """
        Checks the minimum and maximum values in min_max for a given entry against a property in a given annotation

        :param entries: the minimum and maximum value in min_max for a given property
        :param prop: a property value in a given annotation
        """
        if prop > entries[self.max_i]:
            entries[self.max_i] = prop
        if prop < entries[self.min_i] or entries[self.min_i] == 0:
            entries[self.min_i] = prop

    def get_min(self, key):
        """Get the minimum value for a given property"""
        return self.min_max[key][self.min_i]

    def get_max(self, key):
        """Get the maximum value for a given property"""
        return self.min_max[key][self.max_i]

    def iterate_annotations(self):
        """Iterate over the annotations in the dataset"""
        for anno_id, annotation in self.annotations.items():
            yield annotation

    def iterate_branches(self, with_source=True):
        """Iterates over the branches in the dataset. Also returns the source tweet, if so specified in 'with_source'"""
        for source_tweet in self.submissions:
            for branch in source_tweet.branches:
                if with_source:
                    yield source_tweet.source, branch
                else:
                    yield branch

    def iterate_submissions(self):
        """Iterates over the submissions in the dataset"""
        for source_tweet in self.submissions:
            yield source_tweet

    def size(self):
        """Returns the number of annotations in the dataset"""
        return len(self.annotations)

    def train_test_split(self, test_size=0.25, rand_state=42, shuffle=True, stratify=True):
        """
        Creates a split of the dataset, using sklearn's test_train_split method, and returns this split.

        :param test_size: the part of the dataset used for testing, the rest is used for training
        :param rand_state: the random state used for shuffling
        :param shuffle: whether the data should be shuffled before splitting
        :param stratify: whether stratification of the data should be applied
        """
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
        """
        Prints a histogram of the dispersion of annotations between the SDQC labels

        :param annotations: allows running the method with with iterate_annotations (if annotations=None), else uses the
        annotations given
        :return:
        """
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
