import ast
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from scipy import spatial


def minmax_scaler(val: float, minval: float, maxval: float) -> float:
    val = max(min(val, maxval), minval)
    normalized_val = (val - minval) / (maxval - minval)
    return normalized_val


def calculate_avg_food_vec(
    sample_foods: List[str], word_vectors: KeyedVectors
) -> np.ndarray:
    sample_food_vecs = []
    for s in sample_foods:
        sample_food_vec = word_vectors[s]
        sample_food_vecs.append(sample_food_vec)
    sample_food_vecs_avg = np.average(sample_food_vecs, axis=0)
    return sample_food_vecs_avg


def nonaroma_values(
    nonaroma: str, average_food_embedding: np.ndarray, food_nonaroma_infos: pd.DataFrame
) -> float:
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
    sample_foods: List[str],
    word_vectors: KeyedVectors,
    food_nonaroma_infos: pd.DataFrame,
) -> Tuple[Dict[str, float], np.ndarray]:
    food_nonaromas = dict()
    average_food_embedding = calculate_avg_food_vec(sample_foods, word_vectors)
    for nonaroma in ["weight", "sweet", "acid", "salt", "piquant", "fat", "bitter"]:
        food_nonaromas[nonaroma] = nonaroma_values(nonaroma, average_food_embedding, food_nonaroma_infos)
    return food_nonaromas, average_food_embedding
