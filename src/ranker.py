from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

from src.dataset import Dataset
from src.logger import get_logger
from src.metrics_calculator import MetricsCalculator
from src.schema.config import Config

logger = get_logger(__file__)


class Ranker:
    def __init__(
        self,
        cfg: Config,
        metrics_calculator: MetricsCalculator,
    ) -> None:
        self.cfg = cfg
        self.metrics_calculator = metrics_calculator
        self.user_feature_cols = [
            "FN",
            "Active",
            "club_member_status",
            "fashion_news_frequency",
            "age",
        ]
        self.item_feature_cols = [
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

    def create_user_features(
        self, customer_df: pd.DataFrame, col_le: dict[str, LabelEncoder]
    ) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
        logger.info("create user features")
        user_features = customer_df[["customer_id"] + self.user_feature_cols].copy()
        for col in self.user_feature_cols:
            if col == "age":
                continue
            if col in col_le:
                le = col_le[col]
                user_features[col] = le.transform(user_features[col].astype(str))
            else:
                le = LabelEncoder()
                user_features[col] = le.fit_transform(user_features[col].astype(str))
                col_le[col] = le
        user_features = user_features.add_prefix("feature__").rename(
            columns={"feature__customer_id": "customer_id"}
        )
        return user_features, col_le

    def create_item_features(
        self, article_df: pd.DataFrame, col_le: dict[str, LabelEncoder]
    ) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
        logger.info("create item features")
        item_features = article_df[["article_id"] + self.item_feature_cols].copy()
        for col in self.item_feature_cols:
            if col in col_le:
                le = col_le[col]
                item_features[col] = le.transform(item_features[col].astype(str))
            else:
                le = LabelEncoder()
                item_features[col] = le.fit_transform(item_features[col].astype(str))
                col_le[col] = le
        item_features = item_features.add_prefix("feature__").rename(
            columns={"feature__article_id": "article_id"}
        )
        return item_features, col_le

    def create_item_user_trans_feature(
        self,
        trans_df: pd.DataFrame,
        ref_date: date,
        customer_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        logger.info("create item user trans features")
        trans_df["days_since_purchase"] = (ref_date - trans_df["t_dat"]).dt.days

        item_trans_features = (
            trans_df.merge(customer_df, on="customer_id", how="left")
            .groupby("article_id")
            .agg(
                feature__item_purchase_cnt=("customer_id", "size"),
                feature__item_purchase_nunique=("customer_id", "nunique"),
                feature__item_days_since_last_purchase=("days_since_purchase", "min"),
                feature__item_price_sum=("price", "sum"),
                feature__item_price_mean=("price", "mean"),
                feature__item_price_std=("price", "std"),
                feature__item_price_max=("price", "max"),
                feature__item_price_min=("price", "min"),
                feature__item_age_mean=("age", "mean"),
                feature__item_age_std=("age", "std"),
                feature__item_age_max=("age", "max"),
                feature__item_age_min=("age", "min"),
            )
        )

        user_trans_features = trans_df.groupby("customer_id").agg(
            feature__user_purchase_cnt=("article_id", "size"),
            feature__user_purchase_nunique=("article_id", "nunique"),
            feature__user_days_since_last_purchase=("days_since_purchase", "min"),
            feature__user_price_sum=("price", "sum"),
            feature__user_price_mean=("price", "mean"),
            feature__user_price_std=("price", "std"),
            feature__user_price_min=("price", "min"),
            feature__user_price_max=("price", "max"),
        )
        return item_trans_features, user_trans_features

    def create_time_aware_features(
        self,
        trans_df: pd.DataFrame,
        article_df: pd.DataFrame,
        ref_date: date,
    ) -> list[pd.DataFrame]:
        logger.info(f"Creating time-aware features with ref_date: {ref_date}")

        def time_decay_weight(days: np.ndarray, decay_rate: float) -> np.ndarray:
            return np.exp(-decay_rate * days)

        merged_df = trans_df.merge(article_df, on="article_id", how="left")
        merged_df["days_since_purchase"] = (ref_date - merged_df["t_dat"]).dt.days
        merged_df["time_weight"] = time_decay_weight(
            merged_df["days_since_purchase"], self.cfg.model.features.time_decay_rate
        )
        time_windows = {"1w": timedelta(weeks=1), "1m": timedelta(days=30)}
        all_features = []

        for name, delta in time_windows.items():
            logger.info(f"Processing window: {name}")
            if delta:
                window_df = merged_df[
                    merged_df["days_since_purchase"] < delta.days
                ].copy()
            else:
                window_df = merged_df.copy()

            if window_df.empty:
                continue

            user_item_features = window_df.groupby(["customer_id", "article_id"]).agg(
                **{f"feature__user_item_purchase_cnt_{name}": ("customer_id", "size")},
                **{
                    f"feature__user_item_time_weighted_purchase_cnt_{name}": (
                        "time_weight",
                        "sum",
                    )
                },
            )
            all_features.append(user_item_features)

            for cat_col in self.item_feature_cols:
                user_cat_features = window_df.groupby(["customer_id", cat_col]).agg(
                    **{
                        f"feature__user_{cat_col}_purchase_cnt_{name}": (
                            "customer_id",
                            "size",
                        )
                    },
                    **{
                        f"feature__user_{cat_col}_time_weighted_purchase_cnt_{name}": (
                            "time_weight",
                            "sum",
                        )
                    },
                )
                all_features.append(user_cat_features)

        return all_features

    def merge_item2vec_feature(
        self,
        article_item2vec_embs: dict[str, np.ndarray],
        trans_df: pd.DataFrame,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        logger.info("create item2vec features")
        emb_df = pd.DataFrame(article_item2vec_embs).T.reset_index()
        emb_df.columns = ["article_id"] + [
            f"item2vec_emb_{i}"
            for i in range(self.cfg.model.params.item2vec.vector_size)
        ]

        user_history_df = trans_df[["customer_id", "article_id"]].merge(
            emb_df, on="article_id", how="inner"
        )
        user_profile = (
            user_history_df.drop(columns="article_id").groupby("customer_id").mean()
        )
        user_profile = user_profile.add_prefix("user_profile_")

        df = df.merge(user_profile, on="customer_id", how="left").merge(
            emb_df, on="article_id", how="left"
        )

        user_vec_cols = [
            f"user_profile_item2vec_emb_{i}"
            for i in range(self.cfg.model.params.item2vec.vector_size)
        ]
        item_vec_cols = [
            f"item2vec_emb_{i}"
            for i in range(self.cfg.model.params.item2vec.vector_size)
        ]

        valid_rows = df[user_vec_cols[0]].notna() & df[item_vec_cols[0]].notna()
        user_vectors = df.loc[valid_rows, user_vec_cols].values
        item_vectors = df.loc[valid_rows, item_vec_cols].values
        dot_products = (user_vectors * item_vectors).sum(axis=1)

        df["feature__item2vec_affinity_score"] = 0.0
        df.loc[valid_rows, "feature__item2vec_affinity_score"] = dot_products
        df = df.drop(columns=user_vec_cols + item_vec_cols)
        return df

    def merge_ttm_feature(
        self,
        article_ttm_embs: dict[str, np.ndarray],
        customer_ttm_embs: dict[str, np.ndarray],
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        logger.info("create TwoTowerModel features")
        article_emb_df = pd.DataFrame(article_ttm_embs).T.reset_index()
        article_emb_df.columns = ["article_id"] + [
            f"article_emb_{i}" for i in range(self.cfg.model.params.ttm.emb_size)
        ]

        customer_emb_df = pd.DataFrame(customer_ttm_embs).T.reset_index()
        customer_emb_df.columns = ["customer_id"] + [
            f"customer_emb_{i}" for i in range(self.cfg.model.params.ttm.emb_size)
        ]

        df = df.merge(customer_emb_df, on="customer_id", how="left").merge(
            article_emb_df, on="article_id", how="left"
        )

        user_vec_cols = [
            f"customer_emb_{i}" for i in range(self.cfg.model.params.ttm.emb_size)
        ]
        item_vec_cols = [
            f"article_emb_{i}" for i in range(self.cfg.model.params.ttm.emb_size)
        ]

        valid_rows = df[user_vec_cols[0]].notna() & df[item_vec_cols[0]].notna()
        user_vectors = df.loc[valid_rows, user_vec_cols].values
        item_vectors = df.loc[valid_rows, item_vec_cols].values
        dot_products = (user_vectors * item_vectors).sum(axis=1)

        df["feature__ttm_affinity_score"] = 0.0
        df.loc[valid_rows, "feature__ttm_affinity_score"] = dot_products
        df = df.drop(columns=user_vec_cols + item_vec_cols)
        return df

    def create_ranking_features(
        self,
        df: pd.DataFrame,
        trans_df: pd.DataFrame,
        customer_df: pd.DataFrame,
        article_df: pd.DataFrame,
        ref_date: date,
        article_item2vec_embs: dict[str, np.ndarray],
        article_ttm_embs: dict[str, np.ndarray],
        customer_ttm_embs: dict[str, np.ndarray],
        col_le: dict[str, LabelEncoder] | dict,
    ) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
        logger.info("create ranking features")
        item_trans_features, user_trans_features = self.create_item_user_trans_feature(
            trans_df.copy(), ref_date, customer_df
        )
        user_features, _ = self.create_user_features(customer_df, col_le)
        item_features, _ = self.create_item_features(article_df, col_le)
        time_features_dfs = self.create_time_aware_features(
            trans_df.copy(), article_df, ref_date
        )

        df = (
            df.merge(item_trans_features, on="article_id", how="left")
            .merge(user_trans_features, on="customer_id", how="left")
            .merge(user_features, on="customer_id", how="left")
            .merge(item_features, on="article_id", how="left")
        )

        df = df.merge(article_df, on="article_id", how="left")
        for time_features in time_features_dfs:
            merge_key = time_features.index.names
            df = df.merge(time_features.reset_index(), on=merge_key, how="left")
        drop_cols = article_df.columns.tolist()
        drop_cols.remove("article_id")
        df = df.drop(columns=drop_cols)

        df = self.merge_item2vec_feature(article_item2vec_embs, trans_df, df)
        # df = self.merge_ttm_feature(article_ttm_embs, customer_ttm_embs, df)
        return df, col_le

    def preprocess(
        self,
        dataset: Dataset,
        candidates_df: pd.DataFrame,
        article_item2vec_embs: dict[str, np.ndarray],
        article_ttm_embs: dict[str, np.ndarray],
        customer_ttm_embs: dict[str, np.ndarray],
    ):
        logger.info("proprocess")
        train_trans_df = (
            dataset.train_trans_df.drop(columns=["price", "sales_channel_id"])
            .drop_duplicates(["t_dat", "customer_id", "article_id"])
            .copy()
        )
        val_trans_df = (
            dataset.val_trans_df.drop(columns=["price", "sales_channel_id"])
            .drop_duplicates(["t_dat", "customer_id", "article_id"])
            .copy()
        )
        test_trans_df = (
            dataset.test_trans_df.drop(columns=["price", "sales_channel_id"])
            .drop_duplicates(["t_dat", "customer_id", "article_id"])
            .copy()
        )

        train_trans_df["purchased"] = 1
        val_trans_df["purchased"] = 1
        test_trans_df["purchased"] = 1

        train_trans_df = (
            candidates_df[["customer_id", "article_id"]]
            .merge(
                train_trans_df[["t_dat", "customer_id"]].drop_duplicates(),
                on="customer_id",
                how="inner",
            )
            .merge(
                train_trans_df, on=["t_dat", "customer_id", "article_id"], how="left"
            )
            .fillna(0)
        )
        val_trans_df = (
            candidates_df[["customer_id", "article_id"]]
            .merge(
                val_trans_df[["t_dat", "customer_id"]].drop_duplicates(),
                on="customer_id",
                how="inner",
            )
            .merge(val_trans_df, on=["t_dat", "customer_id", "article_id"], how="left")
            .fillna(0)
        )
        test_trans_df = (
            candidates_df[["customer_id", "article_id"]]
            .merge(
                test_trans_df[["t_dat", "customer_id"]].drop_duplicates(),
                on="customer_id",
                how="inner",
            )
            .merge(test_trans_df, on=["t_dat", "customer_id", "article_id"], how="left")
            .fillna(0)
        )

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

        train_trans_df["group_id"] = train_trans_df.groupby(
            ["t_dat", "customer_id"]
        ).ngroup()
        val_trans_df["group_id"] = val_trans_df.groupby(
            ["t_dat", "customer_id"]
        ).ngroup()
        test_trans_df["group_id"] = test_trans_df.groupby(
            ["t_dat", "customer_id"]
        ).ngroup()

        col_le = {}
        train_trans_df, col_le = self.create_ranking_features(
            train_trans_df,
            dataset.past_trans_df,
            dataset.customer_df,
            dataset.article_df,
            dataset.train_start_date,
            article_item2vec_embs,
            article_ttm_embs,
            customer_ttm_embs,
            col_le,
        )
        val_trans_df, col_le = self.create_ranking_features(
            val_trans_df,
            pd.concat([dataset.past_trans_df, dataset.train_trans_df], axis=0),
            dataset.customer_df,
            dataset.article_df,
            dataset.val_start_date,
            article_item2vec_embs,
            article_ttm_embs,
            customer_ttm_embs,
            col_le,
        )
        test_trans_df, col_le = self.create_ranking_features(
            test_trans_df,
            pd.concat(
                [dataset.past_trans_df, dataset.train_trans_df, dataset.val_trans_df],
                axis=0,
            ),
            dataset.customer_df,
            dataset.article_df,
            dataset.test_start_date,
            article_item2vec_embs,
            article_ttm_embs,
            customer_ttm_embs,
            col_le,
        )

        print(
            "\nFeature Engineering Complete.",
            f"\nShape of the training data: {train_trans_df.shape}",
            f"\nShape of the validation data: {val_trans_df.shape}",
            f"\nShape of the test data: {test_trans_df.shape}",
        )

        self.train_trans_df = train_trans_df.sort_values("group_id")
        self.val_trans_df = val_trans_df.sort_values("group_id")
        self.test_trans_df = test_trans_df.sort_values("group_id")

    def evaluate(self, result_dir: Path, dataset: Dataset) -> None:
        logger.info("evaluate ranker")
        metrics_df_rows = []
        rec_results = []
        betas = np.arange(0.05, 1, 0.05)
        for customer_id, group_df in self.test_trans_df.groupby("customer_id"):
            y_pred = self.ranker.predict(
                group_df[self.cfg.model.features.cat + self.cfg.model.features.num]
            )
            group_df["pred"] = y_pred

            true_items = group_df.loc[
                group_df["purchased"] == 1, "article_id"
            ].values.tolist()

            group_df = group_df.sort_values("pred", ascending=False)
            pred_items = group_df.head(self.cfg.exp.num_rec)[
                "article_id"
            ].values.tolist()

            # pred_items_rarank_list = []
            # for beta in betas:
            #     pred_items_rerank = self.reranker.rerank(
            #         group_df.copy(),
            #         "interpolation",
            #         self.cfg.exp.num_rec,
            #         beta=beta,
            #     )
            #     pred_items_rarank_list.append([beta, pred_items_rerank])

            rec_result = {
                "customer_id": customer_id,
                "past_items": dataset.past_trans_df.query(
                    "customer_id == @customer_id"
                )["article_id"].tolist(),
                "true_items": true_items,
                "pred_items": pred_items,
            }
            # rec_result.update(
            #     {f"pred_items:{data[0]}": data[1] for data in pred_items_rarank_list}
            # )
            rec_results.append(rec_result)

            for k in self.cfg.exp.ks:
                metrics_df_rows.append(
                    [
                        self.__class__.__name__,
                        # 0,
                        k,
                        self.metrics_calculator.precision_at_k(
                            true_items, pred_items[:k]
                        ),
                        self.metrics_calculator.recall_at_k(true_items, pred_items[:k]),
                        self.metrics_calculator.ap_at_k(true_items, pred_items[:k]),
                        self.metrics_calculator.ndcg_at_k(true_items, pred_items[:k]),
                    ]
                )
                # for data in pred_items_rarank_list:
                #     metrics_df_rows.append(
                #         [
                #             self.__class__.__name__,
                #             data[0],
                #             k,
                #             self.metrics_calculator.precision_at_k(
                #                 true_items, data[1][:k]
                #             ),
                #             self.metrics_calculator.recall_at_k(
                #                 true_items, data[1][:k]
                #             ),
                #             self.metrics_calculator.ap_at_k(true_items, data[1][:k]),
                #             self.metrics_calculator.ndcg_at_k(true_items, data[1][:k]),
                #         ]
                #     )

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
        metrics_df.to_csv(result_dir.joinpath(f"{self.__class__.__name__}_metrics.csv"))

        for metric_name in ["precision", "recall", "map", "ndcg"]:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
            sns.lineplot(metrics_df, x="k", y=metric_name, hue="model", ax=ax)
            fig.tight_layout()
            fig.savefig(
                result_dir.joinpath(f"{self.__class__.__name__}_{metric_name}.png")
            )

        # rec_results_df = pd.DataFrame(rec_results).set_index("customer_id")
        # past_items = rec_results_df["past_items"].to_dict()
        # true_items = rec_results_df["true_items"].to_dict()
        # all_items_catalog = dataset.past_trans_df["article_id"].unique()
        # sub_metrics_list = []
        # for i, pred_items_col in enumerate(
        #     ["pred_items"] + [f"pred_items:{beta}" for beta in betas]
        # ):
        #     pred_items = rec_results_df[pred_items_col].to_dict()
        #     all_pred_items = [
        #         item for sublist in pred_items.values() for item in sublist
        #     ]
        #     coverage = self.metrics_calculator.coverage(
        #         set(all_pred_items), set(all_items_catalog)
        #     )
        #     gini = self.metrics_calculator.gini_index(all_pred_items)
        #     dissimilarity = self.metrics_calculator.dissimilarity_score(pred_items)
        #     novelty = self.metrics_calculator.novelty(
        #         past_items.values(), pred_items.values()
        #     )
        #     serendipity = self.metrics_calculator.serendipity(
        #         past_items, true_items, pred_items
        #     )
        #     if pred_items_col == "pred_items":
        #         sub_metrics_list.append(
        #             {
        #                 "beta": 0,
        #                 "coverage": coverage,
        #                 "gini": gini,
        #                 "dissimilarity": dissimilarity,
        #                 "novelty": novelty,
        #                 "serendipity": serendipity,
        #             }
        #         )
        #     else:
        #         sub_metrics_list.append(
        #             {
        #                 "beta": betas[i - 1],
        #                 "coverage": coverage,
        #                 "gini": gini,
        #                 "dissimilarity": dissimilarity,
        #                 "novelty": novelty,
        #                 "serendipity": serendipity,
        #             }
        #         )
        # sub_metrics_df = pd.DataFrame(sub_metrics_list)

        # for metric_name in [
        #     "coverage",
        #     "gini",
        #     "dissimilarity",
        #     "novelty",
        #     "serendipity",
        # ]:
        #     fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
        #     sns.lineplot(sub_metrics_df, x="beta", y=metric_name, ax=ax)
        #     fig.tight_layout()
        #     fig.savefig(
        #         result_dir.joinpath(f"{self.__class__.__name__}_{metric_name}.png")
        #     )

        # counts = list(Counter(all_pred_items).values())
        # counts_mmr_item2vec = list(Counter(all_pred_items_mmr_item2vec).values())
        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6))
        # ax.hist(
        #     counts,
        #     color="skyblue",
        #     edgecolor="black",
        #     alpha=0.5,
        #     label=self.__class__.__name__,
        # )
        # ax.hist(
        #     counts_mmr_item2vec,
        #     color="orange",
        #     edgecolor="black",
        #     alpha=0.5,
        #     label=f"{self.__class__.__name__}_mmr_item2vec",
        # )
        # ax.set_title("Histogram of Recommendation Counts per Item")
        # ax.set_xlabel("Number of Times an Item was Recommended")
        # ax.set_ylabel("Number of Items (Frequency)")
        # ax.legend(title="Model", bbox_to_anchor=(1, 1), loc="upper left")
        # fig.tight_layout()
        # fig.savefig(result_dir.joinpath(f"{self.__class__.__name__}_rec_counts.png"))
        # plt.close(fig)

        # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
        # counts = np.array(list(Counter(all_pred_items).values()))
        # sorted_counts = np.sort(counts)
        # cumulative_counts = np.cumsum(sorted_counts) / np.sum(sorted_counts)
        # cumulative_counts = np.insert(cumulative_counts, 0, 0)
        # ax.plot(
        #     np.linspace(0, 1, len(cumulative_counts)),
        #     cumulative_counts,
        #     label=self.__class__.__name__,
        # )
        # counts = np.array(list(Counter(all_pred_items_mmr_item2vec).values()))
        # sorted_counts = np.sort(counts)
        # cumulative_counts = np.cumsum(sorted_counts) / np.sum(sorted_counts)
        # cumulative_counts = np.insert(cumulative_counts, 0, 0)
        # ax.plot(
        #     np.linspace(0, 1, len(cumulative_counts)),
        #     cumulative_counts,
        #     label=f"{self.__class__.__name__}_mmr_item2vec",
        # )
        # ax.plot([0, 1], [0, 1], linestyle="--", label="Line of Perfect Equality")
        # ax.set_xlabel("Cumulative Share of Items")
        # ax.set_ylabel("Cumulative Share of Recommnedations")
        # ax.legend(title="Model", bbox_to_anchor=(1, 1), loc="upper left")
        # fig.tight_layout()
        # fig.savefig(result_dir.joinpath(f"{self.__class__.__name__}_lorenz_curve.png"))
        # plt.close(fig)
