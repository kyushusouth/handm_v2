from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from candidates_generator import CandidatesGenerator
from cat_ranker import CatRanker
from dataset import Dataset
from embedding_generator import EmbeddingGenerator
from metrics_calculator import MetricsCalculator
from mmr_reranker import MMRReranker
from schema.config import Config
from two_tower_model import (
    TwoTowerModel,
    define_cat_dim,
    train,
)
from utils import set_seed

sns.set_style("whitegrid")


def main():
    cfg = Config.load(Path(__file__).parent.parent.joinpath("conf", "config.yaml"))
    current_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_dir = Path(__file__).parent.parent.joinpath("result", current_datetime)
    result_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)

    dataset = Dataset(cfg)

    dataset.article_df[["article_id"]].merge(
        dataset.past_trans_df.groupby("article_id")
        .size()
        .to_frame(name="purchase_cnt")
        .reset_index(),
        on="article_id",
        how="left",
    )

    purchase_cnt_df = (
        dataset.past_trans_df.copy()
        .merge(
            dataset.article_df[["article_id", "product_type_name"]],
            on="article_id",
            how="left",
        )
        .groupby(["customer_id", "product_type_name"])
        .agg(purchase_cnt=("article_id", "size"))
    )

    breakpoint()


if __name__ == "__main__":
    main()
