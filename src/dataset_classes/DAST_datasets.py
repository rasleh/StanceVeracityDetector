import datetime

from src.dataset_classes.datasets import *

url_tag = 'urlurlurl'
reddit_regex_url = re.compile(
    r"([(\[]?(https?://)|(https?://www.)|(www.))(?:[a-zæøåA-ZÆØÅ]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)
twitter_regex_url = re.compile('(https://t\\.co/*?)')

punctuation = re.compile('[^a-zA-ZæøåÆØÅ0-9]')
quote_tag = 'refrefref'
reddit_regex_ref = re.compile(r">(.+?)\n")
twitter_regex_ref = re.compile('(@.*? )')
timestamp = '2'

rand = random.Random(42)

afinn = Afinn(language='da', emoticons=True)


class DastAnnotation(Annotation):
    # initialises comment annotation class given json
    def __init__(self, data, is_source=False):
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
            self.num_comments = data["num_comments"]
            self.url = data["url"]
            self.text_url = data["text_url"]
            self.is_video = data["is_video"]
            self.reply_count = comment_json["num_comments"]
            self.is_submitter = True
            self.is_rumour = data["IsRumour"]
            self.truth_status = data["TruthStatus"]
            self.rumour = data["RumourDescription"]
            sdqc_source = data["SourceSDQC"]
            sdqc = "Commenting" if sdqc_source == "Underspecified" else sdqc_source
            self.sdqc_parent = sdqc
            self.sdqc_submission = sdqc
            self.tokens = tokenize(self.title)
        else:
            # comment specific info
            self.id = comment_json["comment_id"]
            self.parent_id = comment_json["parent_id"]
            self.comment_url = comment_json["comment_url"]
            self.is_submitter = comment_json["is_submitter"]
            self.is_deleted = comment_json["is_deleted"]
            self.reply_count = comment_json["replies"]
            self.tokens = tokenize(self.text)

            # annotation info
            # self.annotator = json["annotator"]
            self.sdqc_parent = comment_json["SDQC_Parent"]
            self.sdqc_submission = comment_json["SDQC_Submission"]
            self.certainty = comment_json["Certainty"]
            self.evidentiality = comment_json["Evidentiality"]
            self.annotated_at = comment_json["AnnotatedAt"]

        # Placeholder values for cosine similarity calculaton
        self.sim_to_src = 0
        self.sim_to_prev = 0
        self.sim_to_branch = 0

        # general info
        self.submission_id = comment_json["submission_id"]
        self.created = datetime.datetime.strptime(comment_json["created"], '%Y-%m-%dT%H:%M:%S')
        self.upvotes = comment_json["upvotes"]

    def filter_text_ref(self, text):
        """filters text of all annotations to replace reddit quotes with 'refrefref'"""
        return reddit_regex_ref.sub(quote_tag, text)

    def filter_text_urls(self, text):
        """filters text of all annotations to replace 'URLURLURL'"""
        return reddit_regex_url.sub(url_tag, text)


class DastDataset(DataSet):
    def add_submission(self, source):
        self.submissions.append(SourceSubmission(DastAnnotation(source, is_source=True)))

    def create_annotation(self, annotation):
        return DastAnnotation(annotation)
