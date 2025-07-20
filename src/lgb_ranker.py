import random
from datetime import date
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.preprocessing import LabelEncoder

from src.dataset import Dataset
from src.metrics_calculator import MetricsCalculator
from src.schema.config import Config


class LGBRanker:
    def __init__(self, cfg: Config, metrics_calculator: MetricsCalculator) -> None:
        self.cfg = cfg
        self.metrics_calculator = metrics_calculator

    def create_ranking_features(
        self,
        df: pd.DataFrame,
        trans_df: pd.DataFrame,
        customer_df: pd.DataFrame,
        article_df: pd.DataFrame,
        ref_date: date,
        all_embeddings: dict[str, dict[str, np.ndarray]],
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
        )

        for emb_name, item_embeddings in all_embeddings.items():
            if not item_embeddings:
                continue

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

    def preprocess(self, dataset: Dataset, candidates_df: pd.DataFrame):
        train_trans_df = (
            dataset.train_trans_df.drop(columns=["t_dat", "price", "sales_channel_id"])
            .drop_duplicates(["customer_id", "article_id"])
            .copy()
        )
        val_trans_df = (
            dataset.val_trans_df.drop(columns=["t_dat", "price", "sales_channel_id"])
            .drop_duplicates(["customer_id", "article_id"])
            .copy()
        )
        test_trans_df = (
            dataset.test_trans_df.drop(columns=["t_dat", "price", "sales_channel_id"])
            .drop_duplicates(["customer_id", "article_id"])
            .copy()
        )

        train_trans_df.loc[:, "purchased"] = 1
        val_trans_df.loc[:, "purchased"] = 1
        test_trans_df.loc[:, "purchased"] = 1

        train_trans_df = candidates_df.merge(
            train_trans_df, on=["customer_id", "article_id"], how="left"
        ).fillna(0)
        val_trans_df = candidates_df.merge(
            val_trans_df, on=["customer_id", "article_id"], how="left"
        ).fillna(0)
        test_trans_df = candidates_df.merge(
            test_trans_df, on=["customer_id", "article_id"], how="left"
        ).fillna(0)

        train_trans_df = train_trans_df.merge(
            train_trans_df.groupby("customer_id")
            .agg(has_purchased_item=("purchased", "max"))
            .query("has_purchased_item == 1")
            .reset_index(drop=False)[["customer_id"]],
            on=["customer_id"],
            how="inner",
        )
        val_trans_df = val_trans_df.merge(
            val_trans_df.groupby("customer_id")
            .agg(has_purchased_item=("purchased", "max"))
            .query("has_purchased_item == 1")
            .reset_index(drop=False)[["customer_id"]],
            on=["customer_id"],
            how="inner",
        )
        test_trans_df = test_trans_df.merge(
            test_trans_df.groupby("customer_id")
            .agg(has_purchased_item=("purchased", "max"))
            .query("has_purchased_item == 1")
            .reset_index(drop=False)[["customer_id"]],
            on=["customer_id"],
            how="inner",
        )

        col_le = {}
        train_trans_df, col_le = self.create_ranking_features(
            train_trans_df,
            dataset.past_trans_df,
            dataset.customer_df,
            dataset.article_df,
            dataset.train_start_date,
            {},
            col_le,
        )
        val_trans_df, col_le = self.create_ranking_features(
            val_trans_df,
            dataset.past_trans_df,
            dataset.customer_df,
            dataset.article_df,
            dataset.val_start_date,
            {},
            col_le,
        )
        test_trans_df, col_le = self.create_ranking_features(
            test_trans_df,
            dataset.past_trans_df,
            dataset.customer_df,
            dataset.article_df,
            dataset.test_start_date,
            {},
            col_le,
        )

        print(
            "\nFeature Engineering Complete.",
            f"\nShape of the training data: {train_trans_df.shape}",
            f"\nShape of the validation data: {val_trans_df.shape}",
            f"\nShape of the test data: {test_trans_df.shape}",
        )

        self.train_trans_df = train_trans_df.sort_values("customer_id")
        self.val_trans_df = val_trans_df.sort_values("customer_id")
        self.test_trans_df = test_trans_df.sort_values("customer_id")

    def train(self, result_dir: Path) -> "LGBRanker":
        self.ranker = lgb.LGBMRanker(
            objective=self.cfg.model.params.lgb.objective,
            metric=self.cfg.model.params.lgb.eval_metric,
            max_depth=self.cfg.model.params.lgb.max_depth,
            learning_rate=self.cfg.model.params.lgb.learning_rate,
            n_estimators=self.cfg.model.params.lgb.n_estimators,
            importance_type=self.cfg.model.params.lgb.importance_type,
            random_state=self.cfg.seed,
        )
        self.ranker.fit(
            self.train_trans_df[
                self.cfg.model.features.cat + self.cfg.model.features.num
            ],
            self.train_trans_df["purchased"],
            group=self.train_trans_df.groupby("customer_id").size().values,
            eval_set=[
                (
                    self.val_trans_df[
                        self.cfg.model.features.cat + self.cfg.model.features.num
                    ],
                    self.val_trans_df["purchased"],
                )
            ],
            eval_group=[
                self.val_trans_df.groupby("customer_id").size().values,
            ],
            eval_metric=self.cfg.model.params.lgb.eval_metric,
            eval_at=self.cfg.model.params.lgb.eval_at,
            categorical_feature=self.cfg.model.features.cat,
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=self.cfg.model.params.lgb.early_stopping_round
                )
            ],
        )

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
        metric_colormaps = {"ndcg": "Blues", "map": "Oranges"}
        eval_at_list = self.cfg.model.params.lgb.eval_at
        num_k_values = len(eval_at_list)
        for metric_name, base_cmap_name in metric_colormaps.items():
            cmap = plt.get_cmap(base_cmap_name)
            colors = [cmap(i) for i in np.linspace(0.3, 0.9, num_k_values)]
            for i, k in enumerate(eval_at_list):
                label = f"{metric_name}@{k}"
                metric_values = self.ranker.evals_result_["valid_0"][label]
                ax.plot(metric_values, color=colors[i], label=label)
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Score")
        ax.legend(title="Metrics", bbox_to_anchor=(1, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(result_dir.joinpath("ranker_learning_curve.png"))
        plt.close(fig)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
        lgb.plot_importance(
            self.ranker, ax=ax, max_num_features=10, importance_type="gain"
        )
        plt.tight_layout()
        plt.savefig(result_dir.joinpath("ranker_feature_importance.png"))
        plt.close(fig)

        explainer = shap.TreeExplainer(self.ranker)
        shap_samples = self.val_trans_df.sample(
            self.cfg.exp.summary_plot_num_sample, random_state=self.cfg.seed
        )[self.cfg.model.features.cat + self.cfg.model.features.num]
        shap_values = explainer.shap_values(shap_samples)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
        shap.summary_plot(shap_values, shap_samples, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(result_dir.joinpath("shap_summary_plot.png"))
        plt.close(fig)

        return self

    def evaluate(self, result_dir: Path) -> None:
        metrics_df_rows = []
        for _, group_df in self.test_trans_df.groupby("customer_id"):
            y_pred = self.ranker.predict(
                group_df[self.cfg.model.features.cat + self.cfg.model.features.num]
            )
            group_df["pred"] = y_pred
            group_df = group_df.sort_values("pred", ascending=False)
            true_items = group_df.loc[
                group_df["purchased"] == 1, "article_id"
            ].values.tolist()
            pred_items = group_df.head(self.cfg.exp.num_rec)[
                "article_id"
            ].values.tolist()
            pred_rand_items = random.sample(
                group_df["article_id"].values.tolist(), self.cfg.exp.num_rec
            )
            for k in self.cfg.exp.ks:
                pred_items_k = pred_items[:k]
                pred_rand_items_k = pred_rand_items[:k]
                metrics_df_rows.append(
                    [
                        "lightgbm",
                        k,
                        self.metrics_calculator.precision_at_k(
                            true_items, pred_items_k
                        ),
                        self.metrics_calculator.recall_at_k(true_items, pred_items_k),
                        self.metrics_calculator.ap_at_k(true_items, pred_items_k),
                        self.metrics_calculator.ndcg_at_k(true_items, pred_items_k),
                    ]
                )
                metrics_df_rows.append(
                    [
                        "rand",
                        k,
                        self.metrics_calculator.precision_at_k(
                            true_items, pred_rand_items_k
                        ),
                        self.metrics_calculator.recall_at_k(
                            true_items, pred_rand_items_k
                        ),
                        self.metrics_calculator.ap_at_k(true_items, pred_rand_items_k),
                        self.metrics_calculator.ndcg_at_k(
                            true_items, pred_rand_items_k
                        ),
                    ]
                )
        metrics_df = pd.DataFrame(
            metrics_df_rows,
            columns=["model", "k", "precision", "recall", "map", "ndcg"],
        )
        metrics_df = (
            metrics_df.groupby(["model", "k"])
            .agg(
                precision=("precision", "mean"),
                recall=("recall", "mean"),
                map=("map", "mean"),
                ndcg=("ndcg", "mean"),
            )
            .reset_index(drop=False)
        )

        metrics_df.to_csv(result_dir.joinpath("ranker_metrics.csv"))
        for metric_name in ["precision", "recall", "map", "ndcg"]:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
            sns.lineplot(metrics_df, x="k", y=metric_name, hue="model", ax=ax)
            plt.tight_layout()
            plt.savefig(result_dir.joinpath(f"ranker_{metric_name}.png"))
            plt.close(fig)
