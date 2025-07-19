import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from schema.config import Config


class CandidatesGenerator:
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
        self,
        past_trans_df: pd.DataFrame,
        item_embeddings: dict[str, np.ndarray],
        topk: int,
    ) -> dict[str, list[str]]:
        print("Generating embedding candidates...")
        emb_df = pd.DataFrame(item_embeddings).T.reset_index()
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
            top_indices = np.argsort(similarity_matrix[i, :])[-topk * 2 :][::-1]
            candidate_aids = item_ids[top_indices]
            if user_id in user_past_items:
                purchased_set = user_past_items[user_id]
                candidate_aids = [
                    aid for aid in candidate_aids if aid not in purchased_set
                ]
            candidates[user_id] = candidate_aids[:topk]

        return candidates
