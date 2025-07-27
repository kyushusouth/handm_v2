from pathlib import Path

import numpy as np
import pandas as pd

from src.logger import get_logger
from src.schema.config import Config

logger = get_logger(__file__)


class Dataset:
    cfg: Config
    customer_df: pd.DataFrame
    article_df: pd.DataFrame
    past_trans_df: pd.DataFrame
    train_trans_df: pd.DataFrame
    val_trans_df: pd.DataFrame
    test_trans_df: pd.DataFrame
    all_customers: np.ndarray
    all_articles: np.ndarray

    def __init__(self, cfg: Config) -> None:
        logger.info("load data")

        self.cfg = cfg

        self.past_start_date = pd.to_datetime(self.cfg.data.past_start_date)
        self.past_end_date = pd.to_datetime(self.cfg.data.past_end_date)
        self.train_start_date = pd.to_datetime(self.cfg.data.train_start_date)
        self.train_end_date = pd.to_datetime(self.cfg.data.train_end_date)
        self.val_start_date = pd.to_datetime(self.cfg.data.val_start_date)
        self.val_end_date = pd.to_datetime(self.cfg.data.val_end_date)
        self.test_start_date = pd.to_datetime(self.cfg.data.test_start_date)
        self.test_end_date = pd.to_datetime(self.cfg.data.test_end_date)

        filtered_chunks = []
        transactions_path = Path(__file__).parent.parent.joinpath(
            self.cfg.data.transactions_path
        )
        for chunk in pd.read_csv(transactions_path, chunksize=self.cfg.data.chunksize):
            chunk["t_dat"] = pd.to_datetime(chunk["t_dat"])
            filtered_chunk = chunk.loc[
                (self.past_start_date <= chunk["t_dat"])
                & (chunk["t_dat"] <= self.test_end_date)
            ]
            if not filtered_chunk.empty:
                filtered_chunks.append(filtered_chunk)

        trans_df = pd.concat(filtered_chunks)
        used_customers = np.random.choice(
            trans_df["customer_id"].unique(),
            size=self.cfg.data.num_customer,
            replace=False,
        )
        trans_df = trans_df.loc[trans_df["customer_id"].isin(used_customers)]

        self.all_customers = trans_df["customer_id"].unique()
        self.all_articles = trans_df["article_id"].unique()

        filtered_chunks = []
        customers_path = Path(__file__).parent.parent.joinpath(
            self.cfg.data.customers_path
        )
        for chunk in pd.read_csv(
            customers_path,
            chunksize=self.cfg.data.chunksize,
            usecols=cfg.data.customer.usecols,
        ):
            filtered_chunk = chunk.loc[chunk["customer_id"].isin(self.all_customers)]
            # filtered_chunk = chunk
            if not filtered_chunk.empty:
                filtered_chunks.append(filtered_chunk)
        self.customer_df = pd.concat(filtered_chunks)

        filtered_chunks = []
        articles_path = Path(__file__).parent.parent.joinpath(
            self.cfg.data.articles_path
        )
        for chunk in pd.read_csv(
            articles_path,
            chunksize=self.cfg.data.chunksize,
            usecols=cfg.data.article.usecols,
        ):
            filtered_chunk = chunk.loc[chunk["article_id"].isin(self.all_articles)]
            # filtered_chunk = chunk
            if not filtered_chunk.empty:
                filtered_chunks.append(filtered_chunk)
        self.article_df = pd.concat(filtered_chunks)

        # self.all_customers = self.customer_df["customer_id"].unique()
        # self.all_articles = self.article_df["article_id"].unique()

        self.past_trans_df = trans_df.loc[
            (self.past_start_date <= trans_df["t_dat"])
            & (trans_df["t_dat"] <= self.past_end_date)
        ].copy()

        self.train_trans_df = (
            trans_df.loc[
                (self.train_start_date <= trans_df["t_dat"])
                & (trans_df["t_dat"] <= self.train_end_date)
            ]
            .copy()
            .merge(self.past_trans_df[["customer_id"]], on=["customer_id"], how="inner")
            .merge(self.past_trans_df[["article_id"]], on="article_id", how="inner")
        )

        self.val_trans_df = (
            trans_df.loc[
                (self.val_start_date <= trans_df["t_dat"])
                & (trans_df["t_dat"] <= self.val_end_date)
            ]
            .copy()
            .merge(self.past_trans_df[["customer_id"]], on=["customer_id"], how="inner")
            .merge(self.past_trans_df[["article_id"]], on="article_id", how="inner")
        )

        self.test_trans_df = (
            trans_df.loc[
                (self.test_start_date <= trans_df["t_dat"])
                & (trans_df["t_dat"] <= self.test_end_date)
            ]
            .copy()
            .merge(self.past_trans_df[["customer_id"]], on=["customer_id"], how="inner")
            .merge(self.past_trans_df[["article_id"]], on="article_id", how="inner")
        )
