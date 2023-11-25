from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.decomposition import PCA

from src.consts.wine import CORE_DESCRIPTORS


def normalize_geo(wine_df_vecs: pd.DataFrame) -> List[Tuple[str]]:
    normalized_geos = []
    l = list(zip(wine_df_vecs["Variety"], wine_df_vecs["Location"]))
    for tup in l:
        if str(tup[0]) != "nan" and tup not in normalized_geos:
            normalized_geos.append(tup)
    return normalized_geos


def get_average_vectors(wine_df_vecs: pd.DataFrame) -> Dict[str, np.ndarray]:
    avg_taste_vecs = dict()
    for t in CORE_DESCRIPTORS:
        review_arrays = wine_df_vecs[t].dropna()
        average_taste_vec = np.average(review_arrays)
        avg_taste_vecs[t] = average_taste_vec
    return avg_taste_vecs


def subset_wine_vectors(
    wine_df_vecs: pd.DataFrame,
    avg_taste_vecs: Dict[str, np.ndarray],
    list_of_varieties: List[str],
    wine_attribute: str,
) -> List[List[Any]]:
    wine_variety_vectors = []
    for v in list_of_varieties:
        one_var_only = wine_df_vecs.loc[
            (wine_df_vecs["Variety"] == v[0]) & (wine_df_vecs["Location"] == v[1])
        ]
        if len(one_var_only) >= 1:
            taste_vecs = list(one_var_only[wine_attribute])
            taste_vecs = [
                avg_taste_vecs[wine_attribute] if "numpy" not in str(type(x)) else x
                for x in taste_vecs
            ]
            average_variety_vec = np.average(taste_vecs, axis=0)
            all_descriptors = list(one_var_only[wine_attribute + "_descriptors"].str[0])
            all_descriptors = [x for x in all_descriptors if str(x) != "nan"]
            word_freqs = Counter(all_descriptors)
            most_common_words = word_freqs.most_common(50)
            top_n_words = [
                (i[0], "{:.2f}".format(i[1] / len(taste_vecs)))
                for i in most_common_words
            ]
            top_n_words = [i for i in top_n_words if len(i[0]) > 2]
            wine_variety_vector = [v, average_variety_vec, top_n_words]
            wine_variety_vectors.append(wine_variety_vector)

    return wine_variety_vectors


def pca_wine_variety(
    wine_df_vecs: pd.DataFrame,
    avg_taste_vecs: Dict[str, np.ndarray],
    list_of_varieties: List[str],
    wine_attribute: str,
    pca: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    wine_var_vectors = subset_wine_vectors(
        wine_df_vecs, avg_taste_vecs, list_of_varieties, wine_attribute
    )
    wine_varieties = [
        str(w[0]).replace("(", "").replace(")", "").replace("'", "").replace('"', "")
        for w in wine_var_vectors
    ]
    wine_var_vec = [w[1] for w in wine_var_vectors]
    if pca:
        pca = PCA(1)
        wine_var_vec = pca.fit_transform(wine_var_vec)
        wine_var_vec = pd.DataFrame(wine_var_vec, index=wine_varieties)
    else:
        wine_var_vec = pd.Series(wine_var_vec, index=wine_varieties)
    wine_var_vec.sort_index(inplace=True)

    wine_descriptors = pd.DataFrame(
        [w[2] for w in wine_var_vectors], index=wine_varieties
    )
    wine_descriptors = pd.melt(wine_descriptors.reset_index(), id_vars="index")
    wine_descriptors.sort_index(inplace=True)

    return wine_var_vec, wine_descriptors


def normalize_aromas_nonaromas(
    wine_df_vecs: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nonaromas = [
        "weight",
        "sweet",
        "acid",
        "piquant",
        "fat",
        "bitter",
    ]
    taste_dataframes = []
    varieties_list = normalize_geo(wine_df_vecs)
    avg_taste_vecs = get_average_vectors(wine_df_vecs)
    aroma_vec, aroma_descriptors = pca_wine_variety(
        wine_df_vecs, avg_taste_vecs, varieties_list, "aroma", pca=False
    )
    taste_dataframes.append(aroma_vec)

    for descriptor in nonaromas:
        pca_w_dataframe, nonaroma_descriptors = pca_wine_variety(
            wine_df_vecs, avg_taste_vecs, varieties_list, descriptor, pca=True
        )
        taste_dataframes.append(pca_w_dataframe)

    all_nonaromas = pd.concat(taste_dataframes, axis=1)
    all_nonaromas.columns = ["aroma"] + nonaromas
    return aroma_descriptors, all_nonaromas


def save_top_descriptors(aroma_descriptors: pd.DataFrame) -> None:
    aroma_descriptors_copy = aroma_descriptors.copy()
    aroma_descriptors_copy.set_index("index", inplace=True)
    aroma_descriptors_copy.dropna(inplace=True)

    aroma_descriptors_copy = pd.DataFrame(
        aroma_descriptors_copy["value"].tolist(), index=aroma_descriptors_copy.index
    )
    aroma_descriptors_copy.columns = ["descriptors", "relative_frequency"]
    aroma_descriptors_copy.to_csv("wine_variety_descriptors.csv")


def normalize_nonaroma_scalers(
    df: pd.DataFrame, cols_to_normalize: List[str]
) -> pd.DataFrame:
    for feature_name in cols_to_normalize:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        df[feature_name] = df[feature_name].apply(
            lambda x: (x - min_value) / (max_value - min_value)
        )
    return df
