from utils.rectools_new_torch_setup import rectools_setup

rectools_setup(r"C:\rectools_tmp")
import warnings

warnings.simplefilter("ignore")
import os
import pandas as pd
from rectools.dataset import Dataset
import hydra
from hydra.utils import get_original_cwd, instantiate

from rectools import Columns
from rectools.models import load_model

from rectools.metrics import (
    calc_metrics,
)
import torch
from lightning_fabric import seed_everything
import numpy as np
from clearml import Task, OutputModel
from omegaconf import OmegaConf

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


RANDOM_STATE = 60
torch.use_deterministic_algorithms(True)
seed_everything(RANDOM_STATE, workers=True)


def fit_cv_model(dataset, model, splitter, top_n, metrics):
    metrics_df = pd.DataFrame()
    for fold, (train_ids, test_ids, _) in enumerate(
        splitter.split(dataset.interactions)
    ):
        fold_ds = dataset.filter_interactions(
            row_indexes_to_keep=train_ids,
            keep_external_ids=True,
            keep_features_for_removed_entities=True,
        )
        interactions_df_test = dataset.interactions.df.loc[test_ids]
        interactions_df_test[Columns.User] = dataset.user_id_map.convert_to_external(
            interactions_df_test[Columns.User]
        )
        interactions_df_test[Columns.Item] = dataset.item_id_map.convert_to_external(
            interactions_df_test[Columns.Item]
        )

        test_users = interactions_df_test[Columns.User].unique()
        model.fit(fold_ds)
        recs = model.recommend(
            users=test_users,
            dataset=fold_ds,
            k=top_n,
            filter_viewed=True,
            on_unsupported_targets="warn",
        )
        metr = calc_metrics(
            metrics=metrics,
            reco=recs,
            interactions=interactions_df_test.loc[
                interactions_df_test["user_id"].isin(test_users)
            ],
        )
        cur_metrics = pd.DataFrame([metr])
        cur_metrics["fold"] = fold
        metrics_df = pd.concat([metrics_df, cur_metrics])
        print(f"eval fold: {fold}")
    return metrics_df, model, recs


@hydra.main(config_path="./configs", config_name="cfg", version_base=None)
def sas_rec_pipeline(cfg):
    os.chdir(get_original_cwd())

    task = Task.init(
        cfg.logging.project_name,
        cfg.logging.task_name,
        tags=cfg.logging.tags,
        output_uri="C:/clearml_outputs",
    )
    logger = task.get_logger()
    task.upload_artifact(
        name="cfg", artifact_object=OmegaConf.to_yaml(cfg, resolve=True)
    )

    interactions_df = pd.read_csv("../data/hot_preprocessed/interactions.csv")
    users_df = pd.read_csv("../data/hot_preprocessed/users.csv")
    items_df = pd.read_csv("../data/hot_preprocessed/items.csv")

    ds = Dataset.construct(
        interactions_df=interactions_df,
        user_features_df=users_df,
        item_features_df=items_df,
        cat_user_features=cfg.user_features,
        cat_item_features=cfg.item_features,
    )
    model = instantiate(cfg.model)
    splitter = instantiate(cfg.splitter)
    metrics = {
        f"{name}@{cfg.metrics.TOP_K}": instantiate(metric, k=cfg.metrics.TOP_K)
        for name, metric in cfg.metrics.metrics.items()
    }
    metrics_df, result_model, recs = fit_cv_model(
        ds, model, splitter, cfg.TOP_N, metrics=metrics
    )
    logger.report_table(
        title="metrics", series="pandas DataFrame", table_plot=metrics_df
    )
    result_model.save("output/models/sasrec_hot.pkl")

    output_model = OutputModel(task=task, name="model_model")
    output_model.update_weights("output/models/sasrec_hot.pkl")
    task.close()


if __name__ == "__main__":
    sas_rec_pipeline()
