import pandas as pd


def weight_rule(df: pd.DataFrame, food_nonaromas: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the weight rule to filter wines based on body weight compatibility with food.

    :param df: DataFrame containing wine attributes.
    :param food_nonaromas: DataFrame containing non-aromatic attributes of the food.
    :return: Filtered DataFrame of wines based on the weight rule.
    """
    df = df.loc[
        (df["weight"] >= food_nonaromas["weight"] - 1)
        & (df["weight"] <= food_nonaromas["weight"])
    ]
    return df


def acidity_rule(df: pd.DataFrame, food_nonaromas: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the acidity rule to filter wines based on acidity compatibility with food.

    :param df: DataFrame containing wine attributes.
    :param food_nonaromas: DataFrame containing non-aromatic attributes of the food.
    :return: Filtered DataFrame of wines based on the acidity rule.
    """
    df = df.loc[df["acid"] >= food_nonaromas["acid"]]
    return df


def sweetness_rule(df: pd.DataFrame, food_nonaromas: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the sweetness rule to filter wines based on sweetness compatibility with food.

    :param df: DataFrame containing wine attributes.
    :param food_nonaromas: DataFrame containing non-aromatic attributes of the food.
    :return: Filtered DataFrame of wines based on the sweetness rule.
    """
    df = df.loc[df["sweet"] >= food_nonaromas["sweet"]]
    return df


def bitterness_rule(df: pd.DataFrame, food_nonaromas: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the bitterness rule to filter wines based on bitterness compatibility with food.

    :param df: DataFrame containing wine attributes.
    :param food_nonaromas: DataFrame containing non-aromatic attributes of the food.
    :return: Filtered DataFrame of wines based on the bitterness rule.
    """
    if food_nonaromas["bitter"] >= 0.75:
        df = df.loc[df["bitter"] < 0.5]
    return df


def bitter_salt_rule(df: pd.DataFrame, food_nonaromas: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the bitter and salt rule to filter wines based on bitterness and salt compatibility with food.

    :param df: DataFrame containing wine attributes.
    :param food_nonaromas: DataFrame containing non-aromatic attributes of the food.
    :return: Filtered DataFrame of wines based on the bitter and salt rule.
    """
    if food_nonaromas["bitter"] >= 0.75:
        df = df.loc[(df["salt"] < 0.5)]
    if food_nonaromas["salt"] >= 0.75:
        df = df.loc[(df["bitter"] < 0.5)]
    return df


def acid_bitter_rule(df: pd.DataFrame, food_nonaromas: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the acid and bitter rule to filter wines based on acidness and bitterness compatibility with food.

    :param df: DataFrame containing wine attributes.
    :param food_nonaromas: DataFrame containing non-aromatic attributes of the food.
    :return: Filtered DataFrame of wines based on the acid and bitter rule.
    """
    if food_nonaromas["acid"] >= 0.75:
        df = df.loc[(df["bitter"] < 0.5)]
    if food_nonaromas["bitter"] >= 0.75:
        df = df.loc[(df["acid"] < 0.5)]
    return df


def acid_piquant_rule(df: pd.DataFrame, food_nonaromas: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the acid and piquant rule to filter wines based on acid and piquant compatibility with food.

    :param df: DataFrame containing wine attributes.
    :param food_nonaromas: DataFrame containing non-aromatic attributes of the food.
    :return: Filtered DataFrame of wines based on the acid and piquant rule.
    """
    if food_nonaromas["acid"] >= 0.75:
        df = df.loc[(df["piquant"] < 0.5)]
    if food_nonaromas["piquant"] >= 0.75:
        df = df.loc[(df["acid"] < 0.5)]
    return df


def eliminate_not_well_together(
    wine_df: pd.DataFrame, food_nonaromas: pd.DataFrame
) -> pd.DataFrame:
    """
    Filter wines based on multiple compatibility rules with non-aromatic attributes of food.

    :param wine_df: DataFrame containing wine attributes.
    :param food_nonaromas: DataFrame containing non-aromatic attributes of the food.
    :return: Filtered DataFrame of wines based on compatibility rules.
    """
    df = weight_rule(wine_df, food_nonaromas)
    list_of_tests = [
        acidity_rule,
        sweetness_rule,
        bitterness_rule,
        bitter_salt_rule,
        acid_bitter_rule,
        acid_piquant_rule,
    ]
    for t in list_of_tests:
        df_test = t(df, food_nonaromas)
        if df_test.shape[0] > 5:
            df = t(df, food_nonaromas)
    return df
