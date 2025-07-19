from datetime import date

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from schema.config import Config


class Ranker:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def create_ranking_features(
        self,
        df: pd.DataFrame,
        trans_df: pd.DataFrame,
        customer_df: pd.DataFrame,
        article_df: pd.DataFrame,
        ref_date: date,
        all_embeddings: dict[str, dict[str, np.ndarray]],
        meta_features: pd.DataFrame,
        col_le: dict[str, LabelEncoder] | dict,
    ) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
        trans_df["days_since_purchase"] = (ref_date - trans_df["t_dat"]).dt.days
        item_trans_features = trans_df.groupby("article_id").agg(
            item_purchase_cnt=("customer_id", "size"),
            item_purchase_nunique=("customer_id", "nunique"),
            item_days_since_last_purchase=("days_since_purchase", "min"),
            item_price_mean=("price", "mean"),
            item_price_std=("price", "std"),
        )
        user_trans_features = trans_df.groupby("customer_id").agg(
            user_purchase_cnt=("article_id", "size"),
            user_purchase_nunique=("article_id", "nunique"),
            user_days_since_last_purchase=("days_since_purchase", "min"),
            user_price_mean=("price", "mean"),
            user_price_std=("price", "std"),
            user_price_sum=("price", "sum"),
        )
        item_user_trans_features = trans_df.groupby(["customer_id", "article_id"]).agg(
            user_item_purchase_cnt=("customer_id", "size")
        )

        user_feature_cols = [
            "FN",
            "Active",
            "club_member_status",
            "fashion_news_frequency",
            "age",
        ]
        user_features = customer_df[["customer_id"] + user_feature_cols].copy()
        for col in user_feature_cols:
            if col == "age":
                continue
            if col in col_le:
                le = col_le[col]
                user_features[col] = le.transform(user_features[col].astype(str))
            else:
                le = LabelEncoder()
                user_features[col] = le.fit_transform(user_features[col].astype(str))
                col_le[col] = le

        item_feature_cols = [
            "colour_group_name",
            "department_name",
            "department_no",
            "garment_group_name",
            "graphical_appearance_name",
            "index_group_name",
            "index_name",
            "perceived_colour_master_name",
            "perceived_colour_value_name",
            "product_group_name",
            "product_type_name",
            "section_name",
        ]
        item_features = article_df[["article_id"] + item_feature_cols].copy()
        for col in item_feature_cols:
            if col in col_le:
                le = col_le[col]
                item_features[col] = le.transform(item_features[col].astype(str))
            else:
                le = LabelEncoder()
                item_features[col] = le.fit_transform(item_features[col].astype(str))
                col_le[col] = le

        df = (
            df.merge(item_trans_features, on="article_id", how="left")
            .merge(user_trans_features, on="customer_id", how="left")
            .merge(
                item_user_trans_features, on=["article_id", "customer_id"], how="left"
            )
            .merge(user_features, on="customer_id", how="left")
            .merge(item_features, on="article_id", how="left")
            .merge(meta_features, on=["customer_id", "article_id"], how="left")
        )

        for emb_name, item_embeddings in all_embeddings.items():
            if not item_embeddings:
                continue  # 埋め込みが空ならスキップ

            embedding_dim = next(iter(item_embeddings.values())).shape[0]
            emb_df = pd.DataFrame(item_embeddings).T.reset_index()
            emb_df.columns = ["article_id"] + [
                f"{emb_name}_emb_{i}" for i in range(embedding_dim)
            ]

            user_history_df = trans_df[["customer_id", "article_id"]].merge(
                emb_df, on="article_id", how="inner"
            )
            user_profile = (
                user_history_df.drop(columns="article_id").groupby("customer_id").mean()
            )
            user_profile = user_profile.add_prefix("user_profile_")

            df = df.merge(user_profile, on="customer_id", how="left")
            df = df.merge(emb_df, on="article_id", how="left")

            user_vec_cols = [
                f"user_profile_{emb_name}_emb_{i}" for i in range(embedding_dim)
            ]
            item_vec_cols = [f"{emb_name}_emb_{i}" for i in range(embedding_dim)]

            valid_rows = df[user_vec_cols[0]].notna() & df[item_vec_cols[0]].notna()
            user_vectors = df.loc[valid_rows, user_vec_cols].values
            item_vectors = df.loc[valid_rows, item_vec_cols].values

            dot_products = (user_vectors * item_vectors).sum(axis=1)

            df[f"{emb_name}_affinity_score"] = 0.0
            df.loc[valid_rows, f"{emb_name}_affinity_score"] = dot_products

            df = df.drop(columns=user_vec_cols + item_vec_cols)

        return df, col_le
