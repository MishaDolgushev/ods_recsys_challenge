import numpy as np
import pandas as pd


def split_users_by_temp(train_intersactions_df, train_users_df, test_users):
    users_has_intersactions = train_intersactions_df["user_id"].unique()
    hot_users = np.intersect1d(test_users, users_has_intersactions)
    warm_users = np.setdiff1d(
        np.intersect1d(test_users, train_users_df["user_id"].unique()),
        hot_users,
    )
    cold_users = np.setdiff1d(
        np.setdiff1d(test_users, users_has_intersactions),
        train_users_df["user_id"].unique(),
    )
    return {
        "hot": hot_users,
        "warm": warm_users,
        "cold": cold_users,
    }

def flatten_columns(
    df: pd.DataFrame, features: list[str], id_name: str
) -> pd.DataFrame:
    flatten_df = pd.DataFrame()
    for feature in features:
        frame = df.reindex(columns=[id_name, feature])
        frame = frame.explode(feature)
        frame.rename(columns={feature: "value"}, inplace=True)
        frame["feature"] = feature
        flatten_df = pd.concat([flatten_df, frame])
    return flatten_df

