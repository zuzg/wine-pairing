import ast
import re

import numpy as np
import pandas as pd
from scipy import spatial


def array_str_to_list(array_string: str) -> np.ndarray:
    """
    Transform string to numpy array

    :param array_string: string to be transformed
    :return: numpy array containing array string
    """
    average_taste_vec = re.sub("\s+", ",", array_string)
    average_taste_vec = average_taste_vec.replace("[,", "[")
    average_taste_vec = np.array(ast.literal_eval(average_taste_vec))
    return average_taste_vec


def sort_by_aroma_similarity(df: pd.DataFrame, food_aroma: pd.DataFrame) -> pd.DataFrame:
    """
    Sort dataframe by aroma similarity

    :param df: dataframe to be sorted
    :param food_aroma: dataframe with food aromas
    :return: sorted dataframe
    """
    df["aroma"] = df["aroma"].apply(array_str_to_list)
    df["aroma_distance"] = df["aroma"].apply(
        lambda x: spatial.distance.cosine(x, food_aroma)
    )
    df.sort_values(by=["aroma_distance"], ascending=True, inplace=True)
    return df
