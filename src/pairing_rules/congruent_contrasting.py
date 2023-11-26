import numpy as np
import pandas as pd


def sweet_pairing(df: pd.DataFrame, food_nonaromas: pd.DataFrame)-> pd.DataFrame:
    # Rule 1: sweet food goes well with highly bitter, fat, piquant, salt or acid wine
    if food_nonaromas["sweet"] >= 0.75:
        df["pairing_type"] = np.where(
            (
                (df.bitter >= 0.75)
                | (df.fat >= 0.75)
                | (df.piquant >= 0.75)
                | (df.acid >= 0.75)
            ),
            "contrasting",
            df.pairing_type,
        )
    return df


def acid_pairing(df: pd.DataFrame, food_nonaromas: pd.DataFrame)-> pd.DataFrame:
    # Rule 2: acidic food goes well with highly sweet, fat, or salt wine
    if food_nonaromas["acid"] >= 0.75:
        df["pairing_type"] = np.where(
            ((df.sweet >= 0.75) | (df.fat >= 0.75) | (df.salt >= 0.75)),
            "contrasting",
            df.pairing_type,
        )
    return df


def salt_pairing(df: pd.DataFrame, food_nonaromas: pd.DataFrame)-> pd.DataFrame:
    # Rule 3: sweet food goes well with highly bitter, fat, piquant, salt or acid wine
    if food_nonaromas["salt"] >= 0.75:
        df["pairing_type"] = np.where(
            (
                (df.bitter >= 0.75)
                | (df.sweet >= 0.75)
                | (df.piquant >= 0.75)
                | (df.fat >= 0.75)
                | (df.acid >= 0.75)
            ),
            "contrasting",
            df.pairing_type,
        )
    return df


def piquant_pairing(df: pd.DataFrame, food_nonaromas: pd.DataFrame)-> pd.DataFrame:
    # Rule 4: piquant food goes well with highly sweet, fat, or salt wine
    if food_nonaromas["piquant"] >= 0.75:
        df["pairing_type"] = np.where(
            ((df.sweet >= 0.75) | (df.fat >= 0.75)),
            "contrasting",
            df.pairing_type,
        )
    return df


def fat_pairing(df: pd.DataFrame, food_nonaromas: pd.DataFrame)-> pd.DataFrame:
    # Rule 5: fatty food goes well with highly bitter, fat, piquant, salt or acid wine
    if food_nonaromas["fat"] >= 0.75:
        df["pairing_type"] = np.where(
            (
                (df.bitter >= 0.75)
                | (df.sweet >= 0.75)
                | (df.piquant >= 0.75)
                | (df.acid >= 0.75)
            ),
            "contrasting",
            df.pairing_type,
        )
    return df


def bitter_pairing(df: pd.DataFrame, food_nonaromas: pd.DataFrame)-> pd.DataFrame:
    # Rule 6: bitter food goes well with highly sweet, fat, or salt wine
    if food_nonaromas["bitter"] >= 0.75:
        df["pairing_type"] = np.where(
            ((df.sweet >= 0.75) | (df.fat >= 0.75)),
            "contrasting",
            df.pairing_type,
        )
    return df


def congruent_pairing(pairing_type: str, max_food_nonaroma_val: float, wine_nonaroma_val: float) -> str:
    if pairing_type == "congruent":
        return "congruent"
    elif wine_nonaroma_val >= max_food_nonaroma_val:
        return "congruent"
    else:
        return ""


def congruent_or_contrasting(
    wine_df: pd.DataFrame, food_nonaromas: pd.DataFrame
) -> pd.DataFrame:
    max_nonaroma_val = max(food_nonaromas.values())
    most_defining_tastes = [
        key for key, val in food_nonaromas.items() if val == max_nonaroma_val
    ]
    wine_df["pairing_type"] = ""
    for m in most_defining_tastes:
        wine_df["pairing_type"] = wine_df.apply(
            lambda x: congruent_pairing(x["pairing_type"], food_nonaromas[m], x[m]),
            axis=1,
        )
    list_of_tests = [
        sweet_pairing,
        acid_pairing,
        salt_pairing,
        piquant_pairing,
        fat_pairing,
        bitter_pairing,
    ]
    for t in list_of_tests:
        wine_df = t(wine_df, food_nonaromas)
    return wine_df
