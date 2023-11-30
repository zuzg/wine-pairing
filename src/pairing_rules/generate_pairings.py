from typing import List, Optional

import pandas as pd
from gensim.models import KeyedVectors

from src.pairing_rules.congruent_contrasting import congruent_or_contrasting
from src.pairing_rules.elimination import eliminate_not_well_together
from src.pairing_rules.similarity import sort_by_aroma_similarity
from src.data_preprocessing.food_normalization import get_food_descriptors


def generate_single_pairing(
    food: str,
    wine_vectors: pd.DataFrame,
    word_vectors: KeyedVectors,
    food_nonaroma_infos: pd.DataFrame,
    K: int,
) -> Optional[List[str]]:
    """
    Generate wine-pairing for single food item

    :param food: food item
    :param wine_vectors: wine vectors
    :param word_vectors: word_vectors
    :param food_nonaroma infos: df with nonaromas
    :param K: top K pairings to return
    """
    try:
        food_nonaromas, aroma_embedding = get_food_descriptors(
            [food.lower()], word_vectors, food_nonaroma_infos
        )
        wine_recommendations = wine_vectors.copy()
        wine_recommendations = eliminate_not_well_together(
            wine_recommendations, food_nonaromas
        )
        wine_recommendations = congruent_or_contrasting(
            wine_recommendations, food_nonaromas
        )
        wine_recommendations = sort_by_aroma_similarity(
            wine_recommendations, aroma_embedding
        )
        # TODO forgot to remove that before, leaving to have consistent comparison
        # to remove for next iteration of experiments!
        wine_recommendations = wine_recommendations.sort_values(
            by="pairing_type", axis=0, ascending=False
        )
        #
        if len(wine_recommendations) > 0:
            paired = wine_recommendations[
                wine_recommendations["pairing_type"].isin(["congruent", "contrasting"])
            ]
            top = wine_recommendations.head(3)
            top = list(top.index)

            if len(top) < K:
                similar = wine_recommendations[
                    wine_recommendations["pairing_type"] == ""
                ].head(K)
                similar_top = list(similar.index)
                top = top + similar_top[: K - len(top)]
            return top
    except KeyError:
        print(f"{food} not found")


def generate_pairs(
    food_list: List[str],
    wine_vectors: pd.DataFrame,
    word_vectors: KeyedVectors,
    food_nonaroma_infos: pd.DataFrame,
    K: int,
) -> None:
    """
    Generate wine-pairing for list of foods

    :param food_list: list of foods
    :param wine_vectors: wine vectors
    :param word_vectors: word_vectors
    :param food_nonaroma infos: df with nonaromas
    :param K: top K pairings to return
    """
    top_wine_dict = dict()

    for food in food_list:
        top = generate_single_pairing(food, wine_vectors, word_vectors, food_nonaroma_infos, K)
        top_wine_dict[food] = dict()
        for i, top_k in enumerate(top):
            top_wine_dict[food][f"top{i+1}"] = top_k

    df = pd.DataFrame.from_dict(top_wine_dict, orient="index")
    df.to_csv("../data/pairing_top3_new.csv")
