import re

from afinn import Afinn

from src.feature_extraction import word_embeddings
from src.feature_extraction.polyglot_pos import pos_tags_occurrence

# Module for extracting features from comment annotations

afinn = Afinn(language='da', emoticons=True)


class FeatureExtractor:
    def __init__(self, dataset):
        word_embeddings.load_saved_word_embeddings()
        # using passed annotations if not testing
        self.dataset = dataset
        self.bow_words = set()
        self.sdqc_to_int = {
            "Supporting": 0,
            "Denying": 1,
            "Querying": 2,
            "Commenting": 3,
            "Underspecified": None
        }

    def create_feature_vectors(self, data, dataset, sdqc_parent=False, text=False, lexicon=False, sentiment=False, pos=False,
                               wembs=False, lstm_wembs=False):
        feature_vectors = []

        for annotation in data:
            instance = self.create_feature_vector(annotation, dataset, sdqc_parent, text, lexicon, sentiment, pos, wembs, lstm_wembs)
            print(instance)
            # Skipping instances with missing data
            if instance is None:
                continue
            feature_vectors.append(instance)
        return feature_vectors

    # Extracts features from comment annotation and extends the different kind of features to eachother.
    def create_feature_vector(self, comment, dataset, sdqc_parent=False, text=False, lexicon=False, sentiment=False,
                              pos=False, wembs=False, lstm_wembs=False):
        feature_vec = list()
        if sdqc_parent:
            feature_vec.append(self.sdqc_to_int[comment.sdqc_parent])
        if text:
            feature_vec.append(self.text_features(comment.text, comment.tokens))
        if lexicon:
            feature_vec.append(self.special_words_in_text(comment.tokens, comment.text))
        if sentiment:
            feature_vec.append(self.normalize(afinn.score(comment.text), 'afinn_score'))
        if pos:
            feature_vec.append(pos_tags_occurrence(comment.text))
        if wembs:
            word_embs = [comment.sim_to_src, comment.sim_to_prev, comment.sim_to_branch]
            avg_wembs = word_embeddings.avg_word_emb(comment.tokens)
            word_embs.append(avg_wembs)
            feature_vec.append(word_embs)
        if lstm_wembs:
            comment_wembs = word_embeddings.full_comment_emb(comment.tokens)
            source_wembs = word_embeddings.full_comment_emb(dataset.anno_to_source[comment.id].tokens)
            # Exclude comments with no word embedding for source or comment text
            if comment_wembs in [None, []] or source_wembs in [None, []]:
                return None
            feature_vec.append(comment_wembs)
            feature_vec.append(source_wembs)

        return comment.id, self.sdqc_to_int[comment.sdqc_submission], feature_vec

    def text_features(self, text, tokens):
        # **Binary occurrence features**
        period = int('.' in text)
        e_mark = int('!' in text)
        q_mark = int('?' in text or any(word.startswith('hv') for word in text.split()))
        hasTripDot = int('...' in text)

        # **(Normalized) count features**
        txt_len = self.normalize(len(text), 'txt_len') if len(text) > 0 else 0
        url_count = self.normalize(tokens.count('urlurlurl'), 'url_count')
        # longest sequence of capital letters, default empty for 0 length
        cap_sequence_max_len = len(max(re.findall(r"[A-ZÆØÅ]+", text), key=len, default=''))
        cap_sequence_max_len = self.normalize(cap_sequence_max_len, 'cap_sequence_max_len')
        tripDotCount = self.normalize(text.count('...'), 'tripDotCount')
        q_mark_count = self.normalize(text.count('?'), 'q_mark_count')
        e_mark_count = self.normalize(text.count('!'), 'e_mark_count')
        # Ratio of capital letters
        cap_count = self.normalize(sum(1 for c in text if c.isupper()), 'cap_count')
        cap_ratio = float(cap_count) / float(len(text)) if len(text) > 0 else 0.0
        # number of words
        tokens_len = 0
        avg_word_len = 0
        if len(tokens) > 0:
            tokens_len = self.normalize(len(tokens), 'tokens_len')
            avg_word_len_true = sum([len(word) for word in tokens]) / len(tokens)
            avg_word_len = self.normalize(avg_word_len_true, 'avg_word_len')
        return [period, e_mark, q_mark, hasTripDot, url_count, tripDotCount, q_mark_count,
                e_mark_count, cap_ratio, txt_len, tokens_len, avg_word_len, cap_sequence_max_len]

    def special_words_in_text(self, tokens, text):
        swear_count = self.count_lexicon_occurence(tokens, self.dataset.swear_words)
        negation_count = self.count_lexicon_occurence(tokens, self.dataset.negation_words)
        positive_smiley_count = self.count_lexicon_occurence(text.split(), self.dataset.positive_smileys)
        negative_smiley_count = self.count_lexicon_occurence(text.split(), self.dataset.negative_smileys)

        return [
            self.normalize(swear_count, 'swear_count'),
            self.normalize(negation_count, 'negation_count'),
            self.normalize(positive_smiley_count, 'positive_smiley_count'),
            self.normalize(negative_smiley_count, 'negative_smiley_count')]

    ### HELPER METHODS ###
    # Counts the amount of words which appear in the lexicon
    def count_lexicon_occurence(self, words, lexion):
        return sum([1 if word in lexion else 0 for word in words])

    def normalize(self, x_i, prop):
        if x_i == 0:
            return 0
        min_x = self.dataset.get_min(prop)
        max_x = self.dataset.get_max(prop)
        if max_x - min_x != 0:
            return (x_i - min_x) / (max_x - min_x)

        return x_i

    ### END OF HELPER METHODS ###
