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
    food_wine_similarity_df: pd.DataFrame, nodes_with_wine_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Create food-wine edges from food_wine_similarity_df.

    :param food_wine_similarity_df: food-wine similarity dataframe
    :param nodes_with_wine_df: nodes dataframe with both food and wine nodes
    :return: food wine edges dataframe
    """
    edges = []
    for index, row in food_wine_similarity_df.iterrows():
        food_name = row["food_name"]
        wine_name = row["wine_item"]
        similarity = row["similarity"]

        food_node_id = nodes_with_wine_df[nodes_with_wine_df["name"] == food_name][
            "node_id"
        ].values[0]
        wine_node_id = nodes_with_wine_df[nodes_with_wine_df["name"] == wine_name][
            "node_id"
        ].values[0]
        new_edge = {
            "id_1": food_node_id,
            "id_2": wine_node_id,
            "score": similarity,
            "edge_type": "ingr-wine",
        }
        edges.append(new_edge)

    edges_df = pd.DataFrame(edges)
    return edges_df
