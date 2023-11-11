import string
from typing import List

from gensim.models.phrases import Phrases
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize


def normalize_text(raw_text: str) -> List[str]:
    """
    Normalize a text.

    :param raw_text: text to normalize
    :return: list of normalized words
    """
    stop_words = set(stopwords.words("english"))
    punctuation_table = str.maketrans({key: None for key in string.punctuation})
    stemmer = SnowballStemmer("english")

    try:
        words = word_tokenize(raw_text)
        sentence = []
        for word in words:
            try:
                word = str(word)
                lower_case_word = str.lower(word)
                stemmed_word = stemmer.stem(lower_case_word)
                no_punctuation = stemmed_word.translate(punctuation_table)

                if len(no_punctuation) > 1 and no_punctuation not in stop_words:
                    sentence.append(no_punctuation)
            except:
                continue

        return sentence
    except Exception as e:
        return ""


def normalize_sentences(sentences_tokenized: List[str]) -> List[str]:
    """
    Normalize a list of sentences.

    :param sentences_tokenized: list of sentences to normalize
    :return: list of normalized sentences
    """
    normalized_sentences = []
    for s in sentences_tokenized:
        normalized_text = normalize_text(s)
        normalized_sentences.append(normalized_text)
    return normalized_sentences


def extract_phrases(
    sentences_normalized: List[str], save_path: str | None = None
) -> List[str]:
    """
    Extract phrases from a list of sentences.

    :param sentences_normalized: list of normalized sentences
    :param save: whether to save the model
    :return: list of phrases
    """
    bigram_model = Phrases(sentences_normalized, min_count=100)
    bigrams = [bigram_model[sent] for sent in sentences_normalized]
    # NOTE: is it really tri-gram every time?
    trigram_model = Phrases(bigrams, min_count=50)
    phrased_sentences = [trigram_model[sent] for sent in bigrams]

    if save_path is not None:
        with open(save_path, "w") as f:
            trigram_model.save(f)
    return phrased_sentences
