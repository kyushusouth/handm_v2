import itertools
from collections import defaultdict
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from src.dataset import Dataset
from src.schema.config import Config
from src.two_tower_model import TTMDataset, TwoTowerModel


class CandidatesGenerator:
    cfg: Config
    candidates_df: pd.DataFrame

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def generate_popular_items(self, trans_df: pd.DataFrame) -> list[str]:
        if "popular" not in self.cfg.candidates.use:
            return []

        popular_items = (
            trans_df["article_id"]
            .value_counts()
            .nlargest(self.cfg.candidates.popular.topk)
            .index.tolist()
        )
        return popular_items

    def generate_age_popular_items(
        self, trans_df: pd.DataFrame, customer_df: pd.DataFrame
    ) -> tuple[dict[str, list[str]], dict[int, str]]:
        if "age_popular" not in self.cfg.candidates.use:
            return {}, {}

        df = trans_df.merge(customer_df[["customer_id", "age"]], on="customer_id")
        bins = [0, 19, 29, 39, 49, 59, 69, 120]
        labels = ["<20", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
        df["age_bin"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)

        age_popular_items = df.groupby("age_bin", observed=True)["article_id"].apply(
            lambda x: x.value_counts()
            .nlargest(self.cfg.candidates.age_popular.topk)
            .index.tolist()
        )

        customer_age_map = customer_df.set_index("customer_id")["age"].copy()
        customer_age_map = pd.cut(
            customer_age_map, bins=bins, labels=labels, right=True
        ).to_dict()

        return age_popular_items.to_dict(), customer_age_map

    def generate_user_past_items(self, trans_df: pd.DataFrame) -> dict[str, list[str]]:
        if "user_past" not in self.cfg.candidates.use:
            return {}

        user_past_items = (
            trans_df.sort_values("t_dat", ascending=False)
            .groupby("customer_id")["article_id"]
            .apply(list)
        )
        return user_past_items.to_dict()

    def generate_most_freq_cat_popular_items(
        self, trans_df: pd.DataFrame, article_df: pd.DataFrame
    ) -> tuple[dict[str, str], dict[str, list[str]]]:
        if "most_freq_cat_popular" not in self.cfg.candidates.use:
            return {}, {}

        df = trans_df.merge(
            article_df[["article_id", "product_type_name"]], on="article_id"
        )

        customer_cat_counts = (
            df.groupby(["customer_id", "product_type_name"])
            .size()
            .reset_index(name="counts")
        )
        customer_top_cat = (
            customer_cat_counts.loc[
                customer_cat_counts.groupby("customer_id")["counts"].idxmax()
            ]
            .set_index("customer_id")["product_type_name"]
            .to_dict()
        )

        cat_popular_items = (
            df.groupby("product_type_name")["article_id"]
            .apply(
                lambda x: x.value_counts()
                .nlargest(self.cfg.candidates.most_freq_cat_popular.topk)
                .index.tolist()
            )
            .to_dict()
        )

        return customer_top_cat, cat_popular_items

    def generate_same_product_items(
        self, article_df: pd.DataFrame
    ) -> dict[str, list[str]]:
        if "same_product" not in self.cfg.candidates.use:
            return {}

        product_to_articles = article_df.groupby("product_code")["article_id"].apply(
            list
        )

        same_product_items = {}
        for article_id, product_code in article_df[
            ["article_id", "product_code"]
        ].values:
            items = product_to_articles[product_code]
            if len(items) > 1:
                same_product_items[article_id] = [
                    item for item in items if item != article_id
                ]
            else:
                same_product_items[article_id] = []

        return same_product_items

    def generate_cooccurence_based_candidates(
        self, past_trans_df: pd.DataFrame
    ) -> dict[str, list[str]]:
        if "cooc" not in self.cfg.candidates.use:
            return {}

        cooc_map = (
            past_trans_df[["customer_id", "article_id"]]
            .merge(
                past_trans_df[["customer_id", "article_id"]],
                on="customer_id",
                how="inner",
            )
            .query("article_id_x != article_id_y")
            .reset_index(drop=True)
            .drop(columns="customer_id")
            .groupby(["article_id_x", "article_id_y"])
            .size()
            .to_frame("cnt")
            .reset_index()
            .sort_values(["article_id_x", "cnt"], ascending=[True, False])
            .groupby("article_id_x")
            .agg(cooc_article_ids=("article_id_y", list))["cooc_article_ids"]
            .to_dict()
        )

        candidates = defaultdict(list)
        for customer_id, group_df in past_trans_df.groupby("customer_id"):
            for row in group_df.drop_duplicates("article_id").itertuples():
                candidates[customer_id] += cooc_map.get(row.article_id, [])[
                    : self.cfg.candidates.cooc.topk
                ]
        return candidates

    def generate_item2vec_candidates(
        self, past_trans_df: pd.DataFrame, embeddings: dict[str, np.ndarray]
    ) -> dict[str, list[str]]:
        if "item2vec" not in self.cfg.candidates.use:
            return {}

        index_faiss = faiss.IndexFlatL2(self.cfg.model.params.item2vec.vector_size)
        index_faiss.add(np.stack([emb for emb in embeddings.values()]))
        index_to_id = {
            index: article_id for index, article_id in enumerate(embeddings.keys())
        }

        candidates = {}
        for customer_id, group_df in past_trans_df.groupby("customer_id"):
            item_embs = [
                embeddings[row.article_id]
                for row in group_df.drop_duplicates("article_id").itertuples()
                if row.article_id in embeddings
            ]
            if not item_embs:
                continue
            item_embs_arr = np.stack(item_embs)
            _, article_indices_mat = index_faiss.search(
                item_embs_arr, self.cfg.candidates.faiss.topk
            )
            candidates[customer_id] = [
                index_to_id[article_index]
                for article_indices in article_indices_mat
                for article_index in article_indices
            ]

        return candidates

    def generate_ttm_candidates(
        self, article_embs: dict[str, np.ndarray], customer_embs: dict[str, np.ndarray]
    ):
        index_faiss = faiss.IndexFlatL2(self.cfg.model.params.ttm.emb_size)
        index_faiss.add(np.stack([emb for emb in article_embs.values()]))
        index_to_id = {
            index: article_id for index, article_id in enumerate(article_embs.keys())
        }
        candidates = {}
        customer_id_batch = []
        customer_emb_batch = []
        for customer_id, customer_emb in customer_embs.items():
            customer_id_batch.append(customer_id)
            customer_emb_batch.append(customer_emb)
            if len(customer_id_batch) >= 100:
                _, article_indices_mat = index_faiss.search(
                    np.stack(customer_emb_batch), self.cfg.candidates.faiss.topk
                )
                for batch_i in range(len(customer_id_batch)):
                    candidates[customer_id_batch[batch_i]] = [
                        index_to_id[article_index]
                        for article_index in article_indices_mat[batch_i]
                    ]
        if customer_id_batch:
            _, article_indices_mat = index_faiss.search(
                np.stack(customer_emb_batch), self.cfg.candidates.faiss.topk
            )
            for batch_i in range(len(customer_id_batch)):
                candidates[customer_id_batch[batch_i]] = [
                    index_to_id[article_index]
                    for article_index in article_indices_mat[batch_i]
                ]
        return candidates

    def generate_candidates(
        self,
        dataset: Dataset,
        all_embeddings: dict[str, dict[str, np.ndarray]],
        article_ttm_embs: dict[str, np.ndarray],
        customer_ttm_embs: dict[str, np.ndarray],
    ) -> None:
        candidates_popular = self.generate_popular_items(dataset.past_trans_df)
        candidates_age_popular, customer_age_map = self.generate_age_popular_items(
            dataset.past_trans_df, dataset.customer_df
        )
        candidates_user_past = self.generate_user_past_items(dataset.past_trans_df)
        customer_top_cat, cat_popular_items = self.generate_most_freq_cat_popular_items(
            dataset.past_trans_df, dataset.article_df
        )
        candidates_same_product = self.generate_same_product_items(dataset.article_df)
        candidates_cooc = self.generate_cooccurence_based_candidates(
            dataset.past_trans_df
        )
        if "item2vec" in all_embeddings:
            candidates_item2vec = self.generate_item2vec_candidates(
                dataset.past_trans_df, all_embeddings["item2vec"]
            )
        else:
            candidates_item2vec = {}
        candidates_ttm = self.generate_ttm_candidates(
            article_ttm_embs, customer_ttm_embs
        )

        candidate_sources = []
        for cid in dataset.all_customers:
            for aid in candidates_user_past.get(cid, []):
                candidate_sources.append(
                    {"customer_id": cid, "article_id": aid, "source": "user_past"}
                )
                for same_aid in candidates_same_product.get(aid, []):
                    candidate_sources.append(
                        {
                            "customer_id": cid,
                            "article_id": same_aid,
                            "source": "same_product",
                        }
                    )
            if cid in customer_top_cat:
                top_cat = customer_top_cat[cid]
                for aid_top_cat in cat_popular_items.get(top_cat, []):
                    candidate_sources.append(
                        {
                            "customer_id": cid,
                            "article_id": aid_top_cat,
                            "source": "top_cat",
                        }
                    )
            if cid in customer_age_map:
                age_bin = customer_age_map[cid]
                for aid_age in candidates_age_popular.get(age_bin, []):
                    candidate_sources.append(
                        {"customer_id": cid, "article_id": aid_age, "source": "age"}
                    )
            for aid in candidates_popular:
                candidate_sources.append(
                    {"customer_id": cid, "article_id": aid, "source": "popular"}
                )
            for aid in candidates_cooc.get(cid, []):
                candidate_sources.append(
                    {"customer_id": cid, "article_id": aid, "source": "cooc"}
                )
            for aid in candidates_item2vec.get(cid, []):
                candidate_sources.append(
                    {"customer_id": cid, "article_id": aid, "source": "item2vec"}
                )
            for aid in candidates_ttm.get(cid, []):
                candidate_sources.append(
                    {"customer_id": cid, "article_id": aid, "source": "ttm"}
                )

        self.candidates_df = pd.DataFrame(candidate_sources).drop_duplicates()

    def evaluate_candidates(self, dataset: Dataset, result_dir: Path) -> None:
        eval_data = {
            "train": dataset.train_trans_df.groupby("customer_id")["article_id"]
            .apply(set)
            .to_dict(),
            "val": dataset.val_trans_df.groupby("customer_id")["article_id"]
            .apply(set)
            .to_dict(),
            "test": dataset.test_trans_df.groupby("customer_id")["article_id"]
            .apply(set)
            .to_dict(),
        }

        metrics = []
        for kind, ground_truth in eval_data.items():
            total_true_items = sum(len(s) for s in ground_truth.values())
            for source_name in self.candidates_df["source"].unique():
                source_candidates = self.candidates_df[
                    self.candidates_df["source"] == source_name
                ]
                source_preds = (
                    source_candidates.groupby("customer_id")["article_id"]
                    .apply(set)
                    .to_dict()
                )
                total_pred_items = sum(len(s) for s in source_preds.values())
                hits = 0
                for cid, true_items in ground_truth.items():
                    pred_items = source_preds.get(cid, set())
                    hits += len(true_items & pred_items)
                metrics.append(
                    {
                        "kind": kind,
                        "source": source_name,
                        "num_candidates": total_pred_items,
                        "recall": hits / total_true_items,
                    }
                )

            all_preds = (
                self.candidates_df.groupby("customer_id")["article_id"]
                .apply(set)
                .to_dict()
            )
            total_pred_items = sum(len(s) for s in all_preds.values())
            all_hits = 0
            for cid, true_items in ground_truth.items():
                pred_items = all_preds.get(cid, set())
                all_hits += len(true_items & pred_items)
            metrics.append(
                {
                    "kind": kind,
                    "source": "all",
                    "num_candidates": total_pred_items,
                    "recall": all_hits / total_true_items,
                }
            )

        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(result_dir.joinpath("candidates_metrics.csv"))

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
        sns.barplot(metrics_df, x="source", y="recall", hue="kind", ax=ax)
        plt.tight_layout()
        plt.savefig(result_dir.joinpath("candidates_recall.png"))
        plt.close(fig)
