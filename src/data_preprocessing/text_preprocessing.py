import string
from typing import List

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

from src.consts.wine import CORE_DESCRIPTORS


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
        trigram_model.save(save_path)
    return phrased_sentences


def return_mapped_descriptor(
    word: str, mapping: pd.DataFrame, taste: bool = False
) -> str:
    if word in list(mapping.index):
        if taste:
            normalized_word = mapping["combined"][word]
        else:
            normalized_word = mapping.at[word, "level_3"]
        return normalized_word
    else:
        return word


def normalize_aromas(
    sentences: List[str], descriptor_mapping: pd.DataFrame
) -> List[str]:
    normalized_sentences = []
    for sent in sentences:
        normalized_sentence = []
        for word in sent:
            normalized_word = return_mapped_descriptor(word, descriptor_mapping)
            normalized_sentence.append(str(normalized_word))
        normalized_sentences.append(normalized_sentence)
    return normalized_sentences


def wine_food_word2vec(
    wine_sentences: List[str],
    food_sentences: List[str],
    descriptor_mapping: pd.DataFrame,
    save_path: str,
) -> None:
    normalized_wine_sentences = normalize_aromas(wine_sentences, descriptor_mapping)
    aroma_descriptor_mapping = descriptor_mapping.loc[
        descriptor_mapping["type"] == "aroma"
    ]
    normalized_food_sentences = normalize_aromas(
        food_sentences, aroma_descriptor_mapping
    )
    normalized_sentences = normalized_wine_sentences + normalized_food_sentences
    wine_word2vec_model = Word2Vec(
        sentences=normalized_sentences, vector_size=300, min_count=8
    )
    wine_word2vec_model.save(save_path)


def normalize_nonaromas(
    wine_reviews: List[str],
    descriptor_mapping: pd.DataFrame,
    wine_trigram_model: Phraser,
) -> List:
    descriptor_mappings = dict()
    for c in CORE_DESCRIPTORS:
        col = ("primary taste", "type")[c == "aroma"]
        descriptor_mappings[c] = descriptor_mapping.loc[descriptor_mapping[col] == c]

    review_descriptors = []
    for review in wine_reviews:
        taste_descriptors = []
        normalized_review = normalize_text(review)
        phrased_review = wine_trigram_model[normalized_review]

        for c in CORE_DESCRIPTORS:
            descriptors_only = [
                return_mapped_descriptor(word, descriptor_mappings[c], taste=True)
                for word in phrased_review
            ]
            no_nones = [str(d).strip() for d in descriptors_only if d is not None]
            descriptorized_review = " ".join(no_nones)
            taste_descriptors.append(descriptorized_review)
        review_descriptors.append(taste_descriptors)
    return review_descriptors


def calculate_tfidf_embeddings(
    df: pd.DataFrame, review_descriptors: List, wine_word2vec_model: Word2Vec
) -> pd.DataFrame:
    taste_descriptors = []
    taste_vectors = []

    for n in len(CORE_DESCRIPTORS):
        taste_words = [r[n] for r in review_descriptors]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit(taste_words)
        dict_of_tfidf_weightings = dict(zip(X.get_feature_names(), X.idf_))
        wine_review_descriptors = []
        wine_review_vectors = []

        for d in taste_words:
            weighted_review_terms = []
            terms = d.split(" ")
            for term in terms:
                if term in dict_of_tfidf_weightings.keys():
                    tfidf_weighting = dict_of_tfidf_weightings[term]
                    try:
                        word_vector = wine_word2vec_model.wv.get_vector(term).reshape(
                            1, 300
                        )
                        weighted_word_vector = tfidf_weighting * word_vector
                        weighted_review_terms.append(weighted_word_vector)
                    except:
                        continue
            try:
                review_vector = sum(weighted_review_terms) / len(weighted_review_terms) # np.mean
                review_vector = review_vector[0]
            except:
                review_vector = np.nan
            wine_review_vectors.append(review_vector)
            wine_review_descriptors.append(terms)

        taste_vectors.append(wine_review_vectors)
        taste_descriptors.append(wine_review_descriptors)

    taste_vectors_t = list(map(list, zip(*taste_vectors)))
    taste_descriptors_t = list(map(list, zip(*taste_descriptors)))
    review_vecs_df = pd.DataFrame(taste_vectors_t, columns=CORE_DESCRIPTORS)

    columns_taste_descriptors = [a + "_descriptors" for a in CORE_DESCRIPTORS]
    review_descriptors_df = pd.DataFrame(
        taste_descriptors_t, columns=columns_taste_descriptors
    )
    wine_df_vecs = pd.concat([df, review_descriptors_df, review_vecs_df], axis=1)
    return wine_df_vecs
