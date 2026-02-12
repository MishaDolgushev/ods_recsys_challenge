import pandas as pd
from preprocessors import split_users_by_temp, flatten_columns
from rectools import Columns
import yaml


interactions_df = pd.read_csv("../data/interactions_processed.csv")
users_df = pd.read_csv("../data/users_processed.csv")
items_df = pd.read_csv("../data/items_processed.csv")
submisson_df = pd.read_csv("../data/sample_submission.csv")
with open("hot_pipeline/configs/cfg.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

hot_users_ids = split_users_by_temp(
    interactions_df, users_df, submisson_df["user_id"].unique()
)["hot"]

interactions_df = interactions_df.loc[interactions_df["user_id"].isin(hot_users_ids)]
users_df = users_df.loc[users_df["user_id"].isin(hot_users_ids)]

users_df_flat = flatten_columns(
    users_df,
    features=cfg["user_features"],
    id_name=Columns.User,
).reset_index(drop=True)
users_df_flat["value"] = users_df_flat["value"].astype(str).astype("category")
items_df_flat = flatten_columns(
    items_df,
    features=cfg["item_features"],
    id_name=Columns.Item,
).reset_index(drop=True)
items_df_flat["value"] = items_df_flat["value"].astype(str).astype("category")
interactions_df.rename(
    columns={"watched_pct": Columns.Weight, "last_watch_dt": Columns.Datetime},
    inplace=True,
)
interactions_df[Columns.Weight] = interactions_df[Columns.Weight] / 100

users_df_flat.to_csv("../data/hot_preprocessed/users.csv", index=False)
items_df_flat.to_csv("../data/hot_preprocessed/items.csv", index=False)
interactions_df.to_csv("../data/hot_preprocessed/interactions.csv", index=False)
