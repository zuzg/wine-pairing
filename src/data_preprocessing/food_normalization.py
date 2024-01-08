import ast
import re

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from scipy import spatial


def minmax_scaler(val: float, minval: float, maxval: float) -> float:
    """
    Scale a numerical value using min-max scaling within a specified range.

    :param val: The numerical value to be scaled.
    :param minval: The minimum value of the range for scaling.
    :param maxval: The maximum value of the range for scaling.
    :return: The normalized value within the specified range.
    """
    val = max(min(val, maxval), minval)
    normalized_val = (val - minval) / (maxval - minval)
    return normalized_val


def calculate_avg_food_vec(
    sample_foods: list[str], word_vectors: KeyedVectors
) -> np.ndarray:
    """
    Calculate the average vector representation for a list of sample foods using word vectors.

    :param sample_foods: A list of food items for which vector representations will be averaged.
    :param word_vectors: Pre-trained word vectors, such as those obtained from Word2Vec or FastText.
    :return: The average vector representation for the list of sample foods.
    """
    sample_food_vecs = []
    for s in sample_foods:
        sample_food_vec = word_vectors[s]
        sample_food_vecs.append(sample_food_vec)
    sample_food_vecs_avg = np.average(sample_food_vecs, axis=0)
    return sample_food_vecs_avg


def nonaroma_values(
    nonaroma: str, average_food_embedding: np.ndarray, food_nonaroma_infos: pd.DataFrame
) -> float:
    """
    Calculate the scaled similarity between a non-aromatic attribute and the average food embedding.

    :param nonaroma: The non-aromatic attribute for which similarity is calculated.
    :param average_food_embedding: The average vector representation of a set of sample foods.
    :param food_nonaroma_infos: DataFrame containing information about non-aromatic attributes.
    :return: Scaled similarity between the non-aromatic attribute and the average food embedding.
    """
    average_taste_vec = food_nonaroma_infos.at[nonaroma, "average_vec"]
    average_taste_vec = re.sub("\s+", ",", average_taste_vec)
    average_taste_vec = average_taste_vec.replace("[,", "[")
    average_taste_vec = np.array(ast.literal_eval(average_taste_vec))

    similarity = 1 - spatial.distance.cosine(average_taste_vec, average_food_embedding)
    scaled_similarity = minmax_scaler(
        similarity,
        food_nonaroma_infos.at[nonaroma, "farthest"],
        food_nonaroma_infos.at[nonaroma, "closest"],
    )
    return scaled_similarity


def get_food_descriptors(
    sample_foods: list[str],
    word_vectors: KeyedVectors,
    food_nonaroma_infos: pd.DataFrame,
) -> tuple[dict[str, float], np.ndarray]:
    """
    Retrieve non-aromatic descriptors and the average vector representation for a list of sample foods.

    :param sample_foods: A list of food items for which descriptors and an average vector will be obtained.
    :param word_vectors: Pre-trained word vectors, such as those obtained from Word2Vec or FastText.
    :param food_nonaroma_infos: DataFrame containing information about non-aromatic attributes.
    :return: A tuple containing:
      - A dictionary of non-aromatic descriptors and their scaled similarity scores.
      - The average vector representation for the list of sample foods.
    """
    food_nonaromas = dict()
    average_food_embedding = calculate_avg_food_vec(sample_foods, word_vectors)
    for nonaroma in ["weight", "sweet", "acid", "salt", "piquant", "fat", "bitter"]:
        food_nonaromas[nonaroma] = nonaroma_values(
            nonaroma, average_food_embedding, food_nonaroma_infos
        )
    return food_nonaromas, average_food_embedding
