import csv
import string
from typing import List

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser, Phrases
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from scipy.spatial import distance
from sklearn.feature_extraction.text import TfidfVectorizer

from src.consts.food import CORE_TASTES
from src.consts.wine import CORE_DESCRIPTORS


def read_csv_to_list_of_lists(file_path) -> List[List[str]]:
    """
    Read a csv file and return a list of lists.

    :param file_path: path to csv file
    """
    data = []
    with open(file_path, "r", newline="") as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if any(row):
                data.append(row)
    return data


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
    trigram_model = Phrases(bigrams, min_count=50)
    phrased_sentences = [trigram_model[sent] for sent in bigrams]

    if save_path is not None:
        trigram_model.save(save_path)
    return phrased_sentences


def return_mapped_descriptor(
    word: str, mapping: pd.DataFrame, taste: bool = False
) -> str:
    """
    Return the mapped descriptor for a given word.

    :param word: word to map
    :param mapping: mapping dataframe
    :param taste: whether to map taste descriptors
    :return: mapped descriptor
    """
    if word in list(mapping.index):
        if taste:
            normalized_word = mapping["combined"][word]
        else:
            normalized_word = mapping.at[word, "level_3"]
        return normalized_word
    elif taste:
        return None
    else:
        return word


def normalize_aromas(
    sentences: List[str], descriptor_mapping: pd.DataFrame
) -> List[str]:
    """
    Normalize aroma descriptors in a list of sentences.

    :param sentences: list of sentences to normalize
    :param descriptor_mapping: mapping dataframe
    :return: list of normalized aroma descriptors
    """
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
    """
    Train a word2vec model on wine and food sentences.

    :param wine_sentences: list of wine sentences
    :param food_sentences: list of food sentences
    :param descriptor_mapping: mapping dataframe
    :param save_path: path to save the model
    :return: None
    """
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
) -> List[List[str]]:
    """
    Normalize non-aroma descriptors in a list of wine reviews.

    :param wine_reviews: list of wine reviews
    :param descriptor_mapping: mapping dataframe
    :param wine_trigram_model: trigram model of wine reviews
    :return: list of normalized non-aroma descriptors for each sentence of each wine review
    """
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

    for n in range(len(CORE_DESCRIPTORS)):
        taste_words = [r[n] for r in review_descriptors]
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit(taste_words)
        dict_of_tfidf_weightings = dict(zip(X.get_feature_names_out(), X.idf_))
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
                review_vector = sum(weighted_review_terms) / len(
                    weighted_review_terms
                )  # np.mean
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


def preprocess_food_list(food_list: List[str], trigram_model: Phraser) -> List[str]:
    """
    Preprocesses a list of foods by normalizing the text and extracting trigrams.

    :param food_list: list of foods
    :param trigram_model: trigram model
    :return: list of preprocessed foods
    """
    food_list = [normalize_text(f) for f in food_list]
    food_list_preprocessed = [trigram_model[f][0] for f in food_list]
    return list(set(food_list_preprocessed))


def make_food_embedding_dict(
    food_list: List[str], wine_word2vec_model: Word2Vec
) -> dict:
    """
    Creates a dictionary of food embeddings.

    :param food_list: list of foods
    :param wine_word2vec_model: word2vec model
    :return: dictionary of food embeddings
    """
    word_vectors = wine_word2vec_model.wv
    food_vecs_dict = {}
    for food in food_list:
        try:
            food_vec = word_vectors[food]
            food_vecs_dict[food] = food_vec
        except:
            continue
    return food_vecs_dict


def compute_core_tastes_embeddings(wine_word2vec_model: Word2Vec):
    """
    Compute the average vector of the core tastes.

    :param wine_word2vec_model: word2vec model
    :return: dictionary of core tastes and their average vector
    """
    word_vectors = wine_word2vec_model.wv
    average_taste_vecs = {}
    for taste, keywords in CORE_TASTES.items():
        all_keyword_vecs = []
        for keyword in keywords:
            keyword_vec = word_vectors[keyword]
            all_keyword_vecs.append(keyword_vec)

        avg_taste_vec = np.average(all_keyword_vecs, axis=0)
        average_taste_vecs[taste] = avg_taste_vec
    return average_taste_vecs


def compute_core_tastes_distances(
    food_embedding_dict: dict, core_tastes_embeddings: dict[str, np.ndarray]
) -> dict[dict[str, float]]:
    """
    Compute the distances between the core tastes and the food embeddings.

    :param food_embedding_dict: dictionary of food embeddings
    :param core_tastes_embeddings: dictionary of core tastes and their average vector
    :return: dictionary of core tastes and their distances to the food embeddings
    """
    core_tastes_distances = {}

    for taste in CORE_TASTES.keys():
        taste_distances = {}
        for food, food_vector in food_embedding_dict.items():
            similarity = 1 - distance.cosine(
                core_tastes_embeddings[taste], food_vector
            )
            taste_distances[food] = similarity

        core_tastes_distances[taste] = taste_distances
    return core_tastes_distances


def get_food_nonaroma_info(
    core_tastes_distances: dict[dict[str, float]],
    average_taste_vecs: dict[str, np.ndarray],
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Find the closest and farthest food items for each core taste and the average vector of the core tastes.

    :param core_tastes_distances: dictionary of core tastes and their distances to the food embeddings
    :param average_taste_vecs: dictionary of core tastes and their average vector
    :param verbose: whether to print the results
    :return: dictionary of core tastes and their closest and farthest food items and average vector
    """
    food_nonaroma_info = {}

    for taste in CORE_TASTES.keys():
        farthest, farthest_distance = min(
            core_tastes_distances[taste].items(), key=lambda x: x[1]
        )
        closest, closest_distance = max(
            core_tastes_distances[taste].items(), key=lambda x: x[1]
        )

        if verbose:
            print(f"Core taste: {taste}")
            print(f"Closest item: {closest} - {closest_distance}")
            print(f"Farthest item: {farthest} - {farthest_distance}")

        food_nonaroma_info[taste] = {
            "farthest": farthest_distance,
            "closest": closest_distance,
            "average_vec": average_taste_vecs[taste],
        }

    food_aroma_info_df = pd.DataFrame.from_dict(food_nonaroma_info).T
    return food_aroma_info_df
