import datetime
import re

from src.dataset_classes.datasets import Annotation, DataSet, SourceSubmission, tokenize

# Tags and regex used for replacing urls and quotes in comment text
ref_tag = 'refrefref'
url_tag = 'urlurlurl'
regex_ref = re.compile(r">(.+?)\n")
regex_url = re.compile(
    r"([(\[]?(https?://)|(https?://www.)|(www.))(?:[a-zæøåA-ZÆØÅ]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)


class DastAnnotation(Annotation):
    """
    Class for representing Reddit comments in the DAST dataset. DastAnnotation objects are meant to be stored in the
     DastDataset class also present within this file. Inherits from the Annotation class.

    Attributes
    text : str
        textual content of a comment
    is_source : boolean
        whether the comment is at the root of the conversation tree
    id : str
        id of the comment
    title : str
        only used if the comment is a source. Sometimes sources do not include text but only a title, in which case the
        title is used as text
    is_rumour : boolean
        whether the given comment is used as a rumour in the dataset
    truth_status : str
        the veracity of the comment, if it is used as a rumour
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

    def __init__(self, data, is_source=False):
        """
        :param data: json object representing comment from the DAST dataset
        :param is_source: Whether the given comment is the source of a conversation tree
        """

        self.is_source = is_source

        comment_json = data if is_source else data["comment"]
        self.text = comment_json["text"].replace('\n', '')
        self.text = self.filter_text_ref(self.text)
        self.text = self.filter_text_urls(self.text)

        if is_source:
            self.id = comment_json["submission_id"]
            self.title = data["title"]
            if not self.text:
                self.text = self.title
            self.is_rumour = data["IsRumour"]
            self.truth_status = data["TruthStatus"]
            sdqc_source = data["SourceSDQC"]
            self.sdqc_submission = "Commenting" if sdqc_source == "Underspecified" else sdqc_source
            self.sdqc_parent = None
            self.tokens = tokenize(self.title)
        else:
            # comment specific info
            self.id = comment_json["comment_id"]
            self.parent_id = comment_json["parent_id"]
            self.tokens = tokenize(self.text)

            # annotation info
            self.sdqc_parent = comment_json["SDQC_Parent"]
            self.sdqc_submission = comment_json["SDQC_Submission"]

        # Placeholder values for cosine similarity calculaton
        self.sim_to_src = 0
        self.sim_to_prev = 0
        self.sim_to_branch = 0

        self.created = datetime.datetime.strptime(comment_json["created"], '%Y-%m-%dT%H:%M:%S')

    def filter_text_ref(self, text):
        """filters text of all annotations to replace references with the tag 'refrefref'"""
        return regex_ref.sub(ref_tag, text)

    def filter_text_urls(self, text):
        """filters text of all annotations to replace urls with the tag'URLURLURL'"""
        return regex_url.sub(url_tag, text)


class DastDataset(DataSet):
    """
        Class for representing a dataset of Reddit comments from the DAST dataset, as generated by the DastAnnotation
        class. Inherits from the DataSet class to overwrite methods which require initialization of objects of the
        DastAnnotation class, where the DataSet class would initialize objects from the Annotation class instead.
    """
    def add_submission(self, source):
        """
        Creates a DastAnnotation object, and uses this to create a SourceSubmission object, adding this to dataset's
        array of source submissions named 'submissions'.

        :param source: a json object representing a source comment of a conversation tree from the DAST dataset
        """
        self.submissions.append(SourceSubmission(DastAnnotation(source, is_source=True)))

    def create_annotation(self, annotation):
        """
        Creates an object of class DastAnnotation, overwriting create_annotation() of the DataSet class which creates an
        object of type Annotation.

        :param annotation: a json object representing a comment from the DAST dataset
        :return: a newly generated DastAnnotation object, based on the json object passed as a parameter
        """
        return DastAnnotation(annotation)
