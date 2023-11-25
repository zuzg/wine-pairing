def weight_rule(df, food_nonaromas):
    # Rule 1: the wine should have at least the same body as the food
    df = df.loc[
        (df["weight"] >= food_nonaromas["weight"] - 1)
        & (df["weight"] <= food_nonaromas["weight"])
    ]
    return df


def acidity_rule(df, food_nonaromas):
    # Rule 2: the wine should be at least as acidic as the food
    df = df.loc[df["acid"] >= food_nonaromas["acid"]]
    return df


def sweetness_rule(df, food_nonaromas):
    # Rule 3: the wine should be at least as sweet as the food
    df = df.loc[df["sweet"] >= food_nonaromas["sweet"]]
    return df


def bitterness_rule(df, food_nonaromas):
    # Rule 4: bitter wines do not pair well with bitter foods
    if food_nonaromas["bitter"] >= 0.75:
        df = df.loc[df["bitter"] < 0.5]
    return df


def bitter_salt_rule(df, food_nonaromas):
    # Rule 5: bitter and salt do not go well together
    if food_nonaromas["bitter"] >= 0.75:
        df = df.loc[(df["salt"] < 0.5)]
    if food_nonaromas["salt"] >= 0.75:
        df = df.loc[(df["bitter"] < 0.5)]
    return df


def acid_bitter_rule(df, food_nonaromas):
    # Rule 6: acid and bitterness do not go well together
    if food_nonaromas["acid"] >= 0.75:
        df = df.loc[(df["bitter"] < 0.5)]
    if food_nonaromas["bitter"] >= 0.75:
        df = df.loc[(df["acid"] < 0.5)]
    return df


def acid_piquant_rule(df, food_nonaromas):
    # Rule 7: acid and piquant do not go well together
    if food_nonaromas["acid"] >= 0.75:
        df = df.loc[(df["piquant"] < 0.5)]
    if food_nonaromas["piquant"] >= 0.75:
        df = df.loc[(df["acid"] < 0.5)]
    return df


def eliminate_not_well_together(wine_df, food_nonaromas):
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
