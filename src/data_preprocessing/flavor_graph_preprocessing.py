import pickle
from copy import deepcopy

import numpy as np
import pandas as pd


def create_wine_nodes(wine_items: list[str], max_id: int) -> pd.DataFrame:
    """
    Create wine nodes from wine_items list.

    :param wine_items: list of wine items
    :param max_id: max node_id in nodes_df
    :return: wine_nodes_df
    """
    wine_nodes = []
    for i, wine in enumerate(wine_items):
        wine_nodes.append(
            {
                "node_id": max_id + i + 1,
                "name": wine,
                "node_type": "wine",
                "is_hub": "wine",
            }
        )

    wine_nodes_df = pd.DataFrame(wine_nodes)
    return wine_nodes_df


def create_food_wine_edges(
    food_wine_pairing_df: pd.DataFrame, nodes_with_wine_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create food-wine edges from food_wine_pairing_df.

    :param food_wine_pairing_df: food-wine similarity dataframe
    :param nodes_with_wine_df: nodes dataframe with both food and wine nodes
    :return: food wine edges dataframe
    """
    edges = []
    for index, row in food_wine_pairing_df.iterrows():
        food_name = row["food_name"]

        wine_columns = sorted(
            [col for col in food_wine_pairing_df.columns if col.startswith("top")]
        )
        for i, wine_column in enumerate(wine_columns, start=1):
            wine_name = row[wine_column]
            if wine_name and str(wine_name) != "nan":
                food_node_id = nodes_with_wine_df[
                    nodes_with_wine_df["name"] == food_name
                ]["node_id"].values[0]
                wine_node_id = nodes_with_wine_df[
                    nodes_with_wine_df["name"] == wine_name
                ]["node_id"].values[0]
                new_edge = {
                    "id_1": food_node_id,
                    "id_2": wine_node_id,
                    "score": 1 / i,  # TODO: add score as a similarity?
                    "edge_type": "ingr-wine",
                }
                edges.append(new_edge)

    edges_df = pd.DataFrame(edges)
    return edges_df


def load_embedding(file_name: str) -> dict:
    """
    Load embedding from pickle file.

    :param file_name: path to pickle file
    :return: dictionary of embeddings
    """
    with open(file_name, "rb") as handle:
        embed_dict = pickle.load(handle)
    return embed_dict


def split_unanonimize_nodes(
    nodes_df: pd.DataFrame, embed_dict: dict
) -> dict[dict[np.array]]:
    """
    Split nodes into ingredient, compound and wine nodes.

    :param nodes_df: nodes dataframe
    :param embed_dict: dictionary of embeddings
    :return: dictionary of embeddings split by node type
    """
    embed_dict_names = {}
    for id, embed in embed_dict.items():
        row = nodes_df[nodes_df["node_id"] == int(id)]
        name = row["name"].values[0]
        node_type = row["node_type"].values[0]

        if node_type not in embed_dict_names:
            embed_dict_names[node_type] = {}

        embed_dict_names[node_type][name] = embed
    return embed_dict_names


def pair_item_with_category(
    items: str,
    fg_embed_dict: dict[dict[np.ndarray]],
    pairing_category: str = "wine",
    top_n: int = 1,
):
    """
    Pair item with category.

    :param item: item to pair
    :param fg_embed_dict: dictionary of embeddings
    :param pairing_category: category to pair with
    :param top_n: number of items to return
    :return: list of items
    """
    item_list = items.split("+")
    item_embeddings = {}
    fg_embed_dict = deepcopy(fg_embed_dict)

    # TODO: change wine_ids (remove unnecessary spaces, ',' -> '_' to unify items)
    for item in item_list:
        ingredient_category = None
        for key in fg_embed_dict.keys():
            if item in fg_embed_dict[key].keys():
                ingredient_category = key
                item_embeddings[item] = fg_embed_dict[ingredient_category][item]
                if ingredient_category == pairing_category:
                    del fg_embed_dict[ingredient_category][item]
                break
            elif item.lower().strip().replace(" ", "_") in fg_embed_dict[key].keys():
                item_lower = item.lower().strip().replace(" ", "_")
                ingredient_category = key
                item_embeddings[item] = fg_embed_dict[ingredient_category][item_lower]
                if ingredient_category == pairing_category:
                    del fg_embed_dict[ingredient_category][item_lower]
                break
        if ingredient_category is None:
            raise ValueError(
                f"Item {item} not found in any category: {fg_embed_dict.keys()}"
            )
        if len(item_list) > 1 and ingredient_category != "ingredient":
            raise ValueError(
                f"Multiple items not supported for non-ingredient items: {item_list}"
            )

    sum_item_embedding = np.sum(list(item_embeddings.values()), axis=0)

    pairing_category_embeddings = fg_embed_dict[pairing_category].values()
    pairing_category_names = fg_embed_dict[pairing_category].keys()
    pairing_category_embeddings = np.array(list(pairing_category_embeddings))
    pairing_category_names = np.array(list(pairing_category_names))
    distances = np.linalg.norm(sum_item_embedding - pairing_category_embeddings, axis=1)

    min_idxs = np.argsort(distances)[:top_n]
    return pairing_category_names[min_idxs].tolist()
