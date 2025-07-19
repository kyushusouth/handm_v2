import random
from collections import defaultdict
from datetime import date, datetime
from itertools import combinations
from pathlib import Path

import gensim
import lightgbm as lgb
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from node2vec import Node2Vec
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

from candidates_generator import CandidatesGenerator
from embedding_generator import EmbeddingGenerator
from metrics_calculator import MetricsCalculator
from ranker import Ranker
from schema.config import Config

plt.rcParams["font.size"] = 18
sns.set_style("whitegrid")


def main():
    cfg = Config.load(Path(__file__).parent.parent.joinpath("conf", "config.yaml"))

    current_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")
    result_dir = Path(__file__).parent.parent.joinpath("result", current_datetime)
    result_dir.mkdir(parents=True, exist_ok=True)

    past_start_date = pd.to_datetime(cfg.data.past_start_date)
    past_end_date = pd.to_datetime(cfg.data.past_end_date)
    train_start_date = pd.to_datetime(cfg.data.train_start_date)
    train_end_date = pd.to_datetime(cfg.data.train_end_date)
    val_start_date = pd.to_datetime(cfg.data.val_start_date)
    val_end_date = pd.to_datetime(cfg.data.val_end_date)
    test_start_date = pd.to_datetime(cfg.data.test_start_date)
    test_end_date = pd.to_datetime(cfg.data.test_end_date)

    filtered_chunks = []
    transactions_path = Path(__file__).parent.parent.joinpath(
        cfg.data.transactions_path
    )
    for chunk in pd.read_csv(transactions_path, chunksize=cfg.data.chunksize):
        chunk["t_dat"] = pd.to_datetime(chunk["t_dat"])
        filtered_chunk = chunk.loc[
            (past_start_date <= chunk["t_dat"]) & (chunk["t_dat"] <= test_end_date)
        ]
        if not filtered_chunk.empty:
            filtered_chunks.append(filtered_chunk)
    trans_df = pd.concat(filtered_chunks)
    all_customers = trans_df["customer_id"].unique()
    all_articles = trans_df["article_id"].unique()

    filtered_chunks = []
    customers_path = Path(__file__).parent.parent.joinpath(cfg.data.customers_path)
    for chunk in pd.read_csv(customers_path, chunksize=cfg.data.chunksize):
        filtered_chunk = chunk.loc[chunk["customer_id"].isin(all_customers)]
        if not filtered_chunk.empty:
            filtered_chunks.append(filtered_chunk)
    customer_df = pd.concat(filtered_chunks)

    filtered_chunks = []
    articles_path = Path(__file__).parent.parent.joinpath(cfg.data.articles_path)
    for chunk in pd.read_csv(articles_path, chunksize=cfg.data.chunksize):
        filtered_chunk = chunk.loc[chunk["article_id"].isin(all_articles)]
        if not filtered_chunk.empty:
            filtered_chunks.append(filtered_chunk)
    article_df = pd.concat(filtered_chunks)

    past_trans_df = trans_df.loc[
        (past_start_date <= trans_df["t_dat"]) & (trans_df["t_dat"] <= past_end_date)
    ].copy()
    train_trans_df = trans_df.loc[
        (train_start_date <= trans_df["t_dat"]) & (trans_df["t_dat"] <= train_end_date)
    ].copy()
    val_trans_df = trans_df.loc[
        (val_start_date <= trans_df["t_dat"]) & (trans_df["t_dat"] <= val_end_date)
    ].copy()
    test_trans_df = trans_df.loc[
        (test_start_date <= trans_df["t_dat"]) & (trans_df["t_dat"] <= test_end_date)
    ].copy()

    candidate_generator = CandidatesGenerator(cfg)
    embedding_generator = EmbeddingGenerator(cfg)
    ranker = Ranker(cfg)
    metrics_calculator = MetricsCalculator(cfg)

    print("--- Calculate Embeddings ---")
    all_embeddings = {}
    # all_embeddings["item2vec"] = create_item2vec_embeddings(
    #     past_trans_df,
    #     vector_size=cfg.model.params.item2vec.vector_size,
    #     window=cfg.model.params.item2vec.window,
    #     min_count=cfg.model.params.item2vec.min_count,
    #     workers=cfg.model.params.item2vec.workers,
    #     sg=cfg.model.params.item2vec.sg,
    #     seed=cfg.seed,
    # )
    # all_embeddings["cooc"] = create_cooccurrence_embeddings(cfg, past_trans_df)
    # all_embeddings["graph"] = create_graph_embeddings(past_trans_df, cfg)

    print("--- Generate Candidates ---")
    candidates_popular = generate_popular_items(cfg, past_trans_df)
    candidates_age_popular, customer_age_map = generate_age_popular_items(
        cfg, past_trans_df, customer_df
    )
    candidates_user_past = generate_user_past_items(cfg, past_trans_df)
    customer_top_cat, cat_popular_items = generate_most_freq_cat_popular_items(
        cfg, past_trans_df, article_df
    )
    candidates_similar = generate_similar_items(cfg, article_df.copy())
    candidates_same_product = generate_same_product_items(cfg, article_df)
    candidates_transition = generate_transition_prob_candidates(
        cfg, past_trans_df, candidates_user_past
    )
    # candidates_item2vec = generate_embedding_based_candidates(
    #     past_trans_df, all_embeddings["item2vec"], cfg.candidates.item2vec.topk
    # )
    # candidates_cooc = generate_embedding_based_candidates(
    #     past_trans_df, all_embeddings["cooc"], cfg.candidates.cooc.topk
    # )
    # candidates_node2vec = generate_embedding_based_candidates(
    #     past_trans_df, all_embeddings["graph"], cfg.candidates.node2vec.topk
    # )

    print("--- Generating final candidates with meta features ---")
    all_customers = customer_df["customer_id"].unique()
    candidate_sources = []
    for cid in all_customers:
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
                    {"customer_id": cid, "article_id": aid_top_cat, "source": "top_cat"}
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
        # for aid in candidates_item2vec.get(cid, []):
        #     candidate_sources.append(
        #         {"customer_id": cid, "article_id": aid, "source": "item2vec"}
        #     )
        # for aid in candidates_cooc.get(cid, []):
        #     candidate_sources.append(
        #         {"customer_id": cid, "article_id": aid, "source": "cooc"}
        #     )
        # for aid in candidates_node2vec.get(cid, []):
        #     candidate_sources.append(
        #         {"customer_id": cid, "article_id": aid, "source": "node2vec"}
        #     )
    candidates_df = pd.DataFrame(candidate_sources).drop_duplicates()

    source_dummies = pd.get_dummies(candidates_df["source"], prefix="source", dtype=int)
    meta_features_df = pd.concat([candidates_df, source_dummies], axis=1)
    meta_features_df = (
        meta_features_df.groupby(["customer_id", "article_id"]).sum().reset_index()
    )

    print("Generating final candidates for all customers...")
    all_customers = customer_df["customer_id"].unique()
    final_candidates = defaultdict(list)

    for cid in all_customers:
        candidates = []

        candidates.extend(candidates_user_past.get(cid, []))
        for aid in candidates_user_past.get(cid, []):
            candidates.extend(candidates_similar.get(aid, []))
            candidates.extend(candidates_same_product.get(aid, []))
        if cid in customer_top_cat:
            candidates.extend(cat_popular_items.get(customer_top_cat[cid], []))
        if cid in customer_age_map:
            candidates.extend(candidates_age_popular[customer_age_map[cid]])
        candidates.extend(candidates_popular)
        candidates.extend(candidates_transition.get(cid, []))
        # candidates.extend(candidates_item2vec.get(cid, []))
        # candidates.extend(candidates_cooc.get(cid, []))
        # candidates.extend(candidates_node2vec.get(cid, []))

        seen = set()
        final_candidates[cid] = []
        for candidate in candidates:
            if candidate not in seen:
                final_candidates[cid].append(candidate)
                seen.add(candidate)

    all_candidate_sources = {
        "popular": {"all_users": candidates_popular},
        "age_popular": {
            cid: candidates_age_popular.get(customer_age_map.get(cid, ""), [])
            for cid in all_customers
        },
        "user_past": candidates_user_past,
        "similar": {
            cid: [
                sim_aid
                for past_aid in candidates_user_past.get(cid, [])
                for sim_aid in candidates_similar.get(past_aid, [])
            ]
            for cid in all_customers
        },
        "same_product": {
            cid: [
                same_aid
                for past_aid in candidates_user_past.get(cid, [])
                for same_aid in candidates_same_product.get(past_aid, [])
            ]
            for cid in all_customers
        },
        "top_cat": {
            cid: cat_popular_items.get(customer_top_cat.get(cid, ""), [])
            for cid in all_customers
        },
        "transition": candidates_transition,
        # "item2vec": candidates_item2vec,
        # "cooc": candidates_cooc,
        # "graph": candidates_node2vec,
    }

    evaluate_candidate_sources(
        train_trans_df, all_candidate_sources, metrics_calculator, cfg
    )
    evaluate_candidate_sources(
        val_trans_df, all_candidate_sources, metrics_calculator, cfg
    )
    evaluate_candidate_sources(
        test_trans_df, all_candidate_sources, metrics_calculator, cfg
    )

    print("\n--- Starting Feature Engineering for Ranking Model ---")
    candidates_df = []
    for cid, aids in final_candidates.items():
        for aid in aids:
            candidates_df.append({"customer_id": cid, "article_id": aid})
    candidates_df = pd.DataFrame(candidates_df)

    train_trans_df = train_trans_df.drop(
        columns=["t_dat", "price", "sales_channel_id"]
    ).drop_duplicates(["customer_id", "article_id"])
    val_trans_df = val_trans_df.drop(
        columns=["t_dat", "price", "sales_channel_id"]
    ).drop_duplicates(["customer_id", "article_id"])
    test_trans_df = test_trans_df.drop(
        columns=["t_dat", "price", "sales_channel_id"]
    ).drop_duplicates(["customer_id", "article_id"])

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
    train_trans_df, col_le = create_ranking_features(
        cfg,
        train_trans_df,
        past_trans_df,
        customer_df,
        article_df,
        train_start_date,
        all_embeddings,
        meta_features_df,
        col_le,
    )
    val_trans_df, col_le = create_ranking_features(
        cfg,
        val_trans_df,
        past_trans_df,
        customer_df,
        article_df,
        val_start_date,
        all_embeddings,
        meta_features_df,
        col_le,
    )
    test_trans_df, col_le = create_ranking_features(
        cfg,
        test_trans_df,
        past_trans_df,
        customer_df,
        article_df,
        test_start_date,
        all_embeddings,
        meta_features_df,
        col_le,
    )

    print(
        "\nFeature Engineering Complete.",
        f"\nShape of the training data: {train_trans_df.shape}",
        f"\nShape of the validation data: {val_trans_df.shape}",
        f"\nShape of the test data: {test_trans_df.shape}",
    )

    train_trans_df = train_trans_df.sort_values("customer_id")
    val_trans_df = val_trans_df.sort_values("customer_id")
    test_trans_df = test_trans_df.sort_values("customer_id")

    ranker = lgb.LGBMRanker(
        objective=cfg.model.params.lgb.objective,
        metric=cfg.model.params.lgb.eval_metric,
        max_depth=cfg.model.params.lgb.max_depth,
        learning_rate=cfg.model.params.lgb.learning_rate,
        n_estimators=cfg.model.params.lgb.n_estimators,
        importance_type=cfg.model.params.lgb.importance_type,
        random_state=cfg.seed,
    )

    ranker.fit(
        train_trans_df[cfg.model.features.cat + cfg.model.features.num],
        train_trans_df["purchased"],
        group=train_trans_df.groupby("customer_id").size().values,
        eval_set=[
            (
                val_trans_df[cfg.model.features.cat + cfg.model.features.num],
                val_trans_df["purchased"],
            )
        ],
        eval_group=[
            val_trans_df.groupby("customer_id").size().values,
        ],
        eval_metric=cfg.model.params.lgb.eval_metric,
        eval_at=cfg.model.params.lgb.eval_at,
        categorical_feature=cfg.model.features.cat,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=cfg.model.params.lgb.early_stopping_round
            )
        ],
    )

    metrics_df_rows = []
    for _, group_df in test_trans_df.groupby("customer_id"):
        y_pred = ranker.predict(
            group_df[cfg.model.features.cat + cfg.model.features.num]
        )
        group_df["pred"] = y_pred
        group_df = group_df.sort_values("pred", ascending=False)
        true_items = group_df.loc[
            group_df["purchased"] == 1, "article_id"
        ].values.tolist()
        pred_items = group_df.head(cfg.exp.num_rec)["article_id"].values.tolist()
        pred_rand_items = random.sample(
            group_df["article_id"].values.tolist(), cfg.exp.num_rec
        )
        for k in cfg.exp.ks:
            pred_items_k = pred_items[:k]
            pred_rand_items_k = pred_rand_items[:k]
            metrics_df_rows.append(
                [
                    "lightgbm",
                    k,
                    metrics_calculator.precision_at_k(true_items, pred_items_k),
                    metrics_calculator.recall_at_k(true_items, pred_items_k),
                    metrics_calculator.ap_at_k(true_items, pred_items_k),
                    metrics_calculator.ndcg_at_k(true_items, pred_items_k),
                ]
            )
            metrics_df_rows.append(
                [
                    "rand",
                    k,
                    metrics_calculator.precision_at_k(true_items, pred_rand_items_k),
                    metrics_calculator.recall_at_k(true_items, pred_rand_items_k),
                    metrics_calculator.ap_at_k(true_items, pred_rand_items_k),
                    metrics_calculator.ndcg_at_k(true_items, pred_rand_items_k),
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

    metrics_df.to_csv(result_dir.joinpath("metrics.csv"))
    for metric_name in ["precision", "recall", "map", "ndcg"]:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
        sns.lineplot(metrics_df, x="k", y=metric_name, hue="model", ax=ax)
        fig.tight_layout()
        fig.savefig(result_dir.joinpath(f"{metric_name}.png"))
        plt.close()


if __name__ == "__main__":
    main()
