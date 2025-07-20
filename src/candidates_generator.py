from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.dataset import Dataset
from src.schema.config import Config


class CandidatesGenerator:
    cfg: Config
    candidates_df: pd.DataFrame

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def generate_popular_items(self, trans_df: pd.DataFrame) -> list[str]:
        if "popular" not in self.cfg.candidates.use:
            return []

        print("Generating popular items...")
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

        print("Generating age-segmented popular items...")
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

        print("Generating user past purchase items...")
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

        print("Generating most frequent category popular items...")
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

    def generate_similar_items(self, article_df: pd.DataFrame) -> dict[str, list[str]]:
        if "similar" not in self.cfg.candidates.use:
            return {}

        print("Generating similar items...")
        feature_cols = [
            "product_type_name",
            "graphical_appearance_name",
            "colour_group_name",
            "department_name",
            "index_name",
        ]
        article_df["features"] = (
            article_df[feature_cols].astype(str).agg(" ".join, axis=1)
        )

        tfidf = TfidfVectorizer(min_df=self.cfg.candidates.similar.min_df)
        tfidf_matrix = tfidf.fit_transform(article_df["features"])

        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        cosine_sim *= -1
        cosine_sim_sorted_indices = np.argsort(cosine_sim, axis=1)

        similar_items = {}
        for idx, row in enumerate(article_df.itertuples()):
            article_id = row.article_id
            top_indices = cosine_sim_sorted_indices[idx][
                1 : self.cfg.candidates.similar.topk + 1
            ]
            similar_items[article_id] = article_df["article_id"].iloc[top_indices]

        return similar_items

    def generate_same_product_items(
        self, article_df: pd.DataFrame
    ) -> dict[str, list[str]]:
        if "same_product" not in self.cfg.candidates.use:
            return {}

        print("Generating same product code items...")
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

    def generate_transition_prob_candidates(
        self, past_trans_df: pd.DataFrame, user_past_items: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        if "transition" not in self.cfg.candidates.use:
            return {}

        print("Generating transition probability candidates...")

        all_articles = past_trans_df["article_id"].unique()
        id2idx = {aid: i for i, aid in enumerate(all_articles)}
        idx2id = {i: aid for i, aid in enumerate(all_articles)}

        purchase_histories = (
            past_trans_df.sort_values(["customer_id", "t_dat"], ascending=True)
            .groupby("customer_id")["article_id"]
            .apply(list)
        )
        n_items = len(all_articles)
        transition_matrix = np.zeros((n_items, n_items))

        for articles in purchase_histories:
            for i in range(len(articles) - 1):
                prev_item_idx = id2idx[articles[i]]
                next_item_idx = id2idx[articles[i + 1]]
                transition_matrix[prev_item_idx, next_item_idx] += 1

        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        prob_matrix = transition_matrix / row_sums

        candidates = {}
        for user_id, past_items in user_past_items.items():
            if not past_items:
                continue

            last_item = past_items[0]
            if last_item not in id2idx:
                continue

            last_item_idx = id2idx[last_item]
            next_item_probs = prob_matrix[last_item_idx, :]
            top_indices = np.argsort(next_item_probs)[
                -self.cfg.candidates.transition.topk * 2 :
            ][::-1]
            candidate_aids = [idx2id[i] for i in top_indices if next_item_probs[i] > 0]

            purchased_set = set(past_items)
            candidate_aids = [aid for aid in candidate_aids if aid not in purchased_set]
            candidates[user_id] = candidate_aids[: self.cfg.candidates.transition.topk]

        return candidates

    def generate_embedding_based_candidates(
        self, past_trans_df: pd.DataFrame, embeddings: dict[str, np.ndarray]
    ) -> dict[str, list[str]]:
        if "item2vec" not in self.cfg.candidates.use:
            return {}

        print("Generating embedding candidates...")
        emb_df = pd.DataFrame(embeddings).T.reset_index()
        emb_df.columns = ["article_id"] + [
            f"emb_{i}" for i in range(emb_df.shape[1] - 1)
        ]

        user_history_df = past_trans_df[["customer_id", "article_id"]].merge(
            emb_df, on="article_id", how="inner"
        )
        user_profile_df = (
            user_history_df.drop(columns="article_id").groupby("customer_id").mean()
        )

        item_vectors = emb_df.drop(columns="article_id").values
        item_ids = emb_df["article_id"].values

        user_profiles_norm = user_profile_df.values / np.linalg.norm(
            user_profile_df.values, axis=1, keepdims=True
        )
        item_vectors_norm = item_vectors / np.linalg.norm(
            item_vectors, axis=1, keepdims=True
        )

        similarity_matrix = user_profiles_norm @ item_vectors_norm.T

        candidates = {}
        user_past_items = past_trans_df.groupby("customer_id")["article_id"].apply(set)

        for i, user_id in enumerate(user_profile_df.index):
            top_indices = np.argsort(similarity_matrix[i, :])[
                -self.cfg.candidates.item2vec.topk * 2 :
            ][::-1]
            candidate_aids = item_ids[top_indices]
            if user_id in user_past_items:
                purchased_set = user_past_items[user_id]
                candidate_aids = [
                    aid for aid in candidate_aids if aid not in purchased_set
                ]
            candidates[user_id] = candidate_aids[: self.cfg.candidates.item2vec.topk]

        return candidates

    def generate_candidates(
        self, dataset: Dataset, all_embeddings: dict[str, dict[str, np.ndarray]]
    ) -> None:
        candidates_popular = self.generate_popular_items(dataset.past_trans_df)
        candidates_age_popular, customer_age_map = self.generate_age_popular_items(
            dataset.past_trans_df, dataset.customer_df
        )
        candidates_user_past = self.generate_user_past_items(dataset.past_trans_df)
        customer_top_cat, cat_popular_items = self.generate_most_freq_cat_popular_items(
            dataset.past_trans_df, dataset.article_df
        )
        candidates_similar = self.generate_similar_items(dataset.article_df.copy())
        candidates_same_product = self.generate_same_product_items(dataset.article_df)
        candidates_transition = self.generate_transition_prob_candidates(
            dataset.past_trans_df, candidates_user_past
        )
        if "item2vec" in all_embeddings:
            candidates_item2vec = self.generate_embedding_based_candidates(
                dataset.past_trans_df, all_embeddings["item2vec"]
            )
        else:
            candidates_item2vec = {}

        candidate_sources = []
        for cid in dataset.all_customers:
            for aid in candidates_user_past.get(cid, []):
                candidate_sources.append(
                    {"customer_id": cid, "article_id": aid, "source": "user_past"}
                )
                for sim_aid in candidates_similar.get(aid, []):
                    candidate_sources.append(
                        {"customer_id": cid, "article_id": sim_aid, "source": "similar"}
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
            for aid in candidates_transition:
                candidate_sources.append(
                    {"customer_id": cid, "article_id": aid, "source": "transition"}
                )
            for aid in candidates_item2vec:
                candidate_sources.append(
                    {"customer_id": cid, "article_id": aid, "source": "item2vec"}
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
                        "precision": hits / total_pred_items,
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
                    "precision": all_hits / total_pred_items,
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
