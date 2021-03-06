# Polyglot requires numpy and libicu-dev, where the latter is only available on  ubuntu/debian linux distributions
# To install on Windows, follow these steps:
# 1. download PyICU.whl from https://www.lfd.uci.edu/~gohlke/pythonlibs/
# 1.1 $ pip install <path to package>
# 2. download PyCLD2.whl from https://www.lfd.uci.edu/~gohlke/pythonlibs/
# 2.1 $ pip install <path to package>
# 3. $ git clone https://github.com/aboSamoor/polyglot
# 4. $ cd polyglot
# 5. $ python setup.py install
# Then, to use it for the danish language download the necessary models as such:
# $ polyglot download embeddings2.da pos2.da
# Docs: https://polyglot.readthedocs.io/en/latest/POS.html
from polyglot.text import Text

tag_set = {
    "ADJ": "adjective",
    "ADP": "adposition",
    "ADV": "adverb",
    "AUX": "auxiliary verb",
    "CONJ": "coordinating conjunction",
    "DET": "determiner",
    "INTJ": "interjection",
    "NOUN": "noun",
    "NUM": "numeral",
    "PART": "particle",
    "PRON": "pronoun",
    "PROPN": "proper noun",
    "PUNCT": "punctuation",
    "SCONJ": "subordinating conjunction",
    "SYM": "symbol",
    "VERB": "verb",
    "X": "other"
}
N = 17  # length of tag set
"""
Script containing functions for extracting POS tags for a given text or corpus using the Polyglot POS tagger 
"""

def format_tag_corpus(corpus_file, output_file):
    """
    Converts a text file at a given input, converts each line into a Polyglot Text object, and extracts POS tags for
    that line, writing it to a given output file.

    :param corpus_file: a corpus file to be converted into POS tags
    :param output_file: the out file which will contain the generated POS tags
    """
    with open(corpus_file, 'r', encoding='utf-8') as f, open(output_file, 'w+', encoding='utf-8') as out:
        for line in f.readlines():
            text = Text(line, hint_language_code='da')
            out.write(line + ' ')
            for _, tag in text.pos_tags:
                out.write(tag + ' ')
            out.write('\n')


def pos_tags(text):
    """
    Extracts POS tags from a string and returns these as an array

    :param text: str object to be ocnverted to POS tags
    :return: an array of POS tags extracted from the text
    """
    text = Text(text, hint_language_code='da')
    pos_tags = []
    for _, tag in text.pos_tags:
        pos_tags.append(tag)
    return pos_tags


def pos_tags_occurrence(text):
    """
    Returns the occurrence of each POS tag in a str text

    :param text: a str object to be converted to POS tags
    :return: an array sized based on the number of used POS tags in tag_set, containing at each index the count of
    occurrences of the POS tag corresponding to that index in the tag_set
    """
    tags = pos_tags(text)
    res = [0] * N
    for i, tag in enumerate(tag_set.keys()):
        if tag in tags:
            res[i] = 1
    return res

