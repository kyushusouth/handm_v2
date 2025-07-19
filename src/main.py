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

from metrics_calculator import MetricsCalculator
from schema.config import Config

plt.rcParams["font.size"] = 18
sns.set_style("whitegrid")


def generate_popular_items(cfg: Config, trans_df: pd.DataFrame) -> list[str]:
    if "popular" not in cfg.candidates.use:
        return []

    print("Generating popular items...")
    popular_items = (
        trans_df["article_id"]
        .value_counts()
        .nlargest(cfg.candidates.popular.topk)
        .index.tolist()
    )
    return popular_items


def generate_age_popular_items(
    cfg: Config, trans_df: pd.DataFrame, customer_df: pd.DataFrame
) -> tuple[dict[str, list[str]], dict[int, str]]:
    if "age_popular" not in cfg.candidates.use:
        return {}, {}

    print("Generating age-segmented popular items...")
    df = trans_df.merge(customer_df[["customer_id", "age"]], on="customer_id")
    bins = [0, 19, 29, 39, 49, 59, 69, 120]
    labels = ["<20", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
    df["age_bin"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)

    age_popular_items = df.groupby("age_bin", observed=True)["article_id"].apply(
        lambda x: x.value_counts()
        .nlargest(cfg.candidates.age_popular.topk)
        .index.tolist()
    )

    customer_age_map = customer_df.set_index("customer_id")["age"].copy()
    customer_age_map = pd.cut(
        customer_age_map, bins=bins, labels=labels, right=True
    ).to_dict()

    return age_popular_items.to_dict(), customer_age_map


def generate_user_past_items(
    cfg: Config, trans_df: pd.DataFrame
) -> dict[str, list[str]]:
    if "user_past" not in cfg.candidates.use:
        return {}

    print("Generating user past purchase items...")
    user_past_items = (
        trans_df.sort_values("t_dat", ascending=False)
        .groupby("customer_id")["article_id"]
        .apply(list)
    )
    return user_past_items.to_dict()


def generate_most_freq_cat_popular_items(
    cfg: Config, trans_df: pd.DataFrame, article_df: pd.DataFrame
) -> tuple[dict[str, str], dict[str, list[str]]]:
    if "most_freq_cat_popular" not in cfg.candidates.use:
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
            .nlargest(cfg.candidates.most_freq_cat_popular.topk)
            .index.tolist()
        )
        .to_dict()
    )

    return customer_top_cat, cat_popular_items


def generate_similar_items(
    cfg: Config,
    article_df: pd.DataFrame,
) -> dict[str, list[str]]:
    if "similar" not in cfg.candidates.use:
        return {}

    print("Generating similar items...")
    feature_cols = [
        "product_type_name",
        "graphical_appearance_name",
        "colour_group_name",
        "department_name",
        "index_name",
    ]
    article_df["features"] = article_df[feature_cols].astype(str).agg(" ".join, axis=1)

    tfidf = TfidfVectorizer(min_df=cfg.candidates.similar.min_df)
    tfidf_matrix = tfidf.fit_transform(article_df["features"])

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    cosine_sim *= -1
    cosine_sim_sorted_indices = np.argsort(cosine_sim, axis=1)

    similar_items = {}
    for idx, row in enumerate(article_df.itertuples()):
        article_id = row.article_id
        top_indices = cosine_sim_sorted_indices[idx][
            1 : cfg.candidates.similar.topk + 1
        ]
        similar_items[article_id] = article_df["article_id"].iloc[top_indices]

    return similar_items


def generate_same_product_items(
    cfg: Config, article_df: pd.DataFrame
) -> dict[str, list[str]]:
    if "same_product" not in cfg.candidates.use:
        return {}

    print("Generating same product code items...")
    product_to_articles = article_df.groupby("product_code")["article_id"].apply(list)

    same_product_items = {}
    for article_id, product_code in article_df[["article_id", "product_code"]].values:
        items = product_to_articles[product_code]
        if len(items) > 1:
            same_product_items[article_id] = [
                item for item in items if item != article_id
            ]
        else:
            same_product_items[article_id] = []

    return same_product_items


def generate_transition_prob_candidates(
    cfg: Config,
    past_trans_df: pd.DataFrame,
    user_past_items: dict[str, list[str]],
) -> dict[str, list[str]]:
    if "transition" not in cfg.candidates.use:
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
            -cfg.candidates.transition.topk * 2 :
        ][::-1]
        candidate_aids = [idx2id[i] for i in top_indices if next_item_probs[i] > 0]

        purchased_set = set(past_items)
        candidate_aids = [aid for aid in candidate_aids if aid not in purchased_set]
        candidates[user_id] = candidate_aids[: cfg.candidates.transition.topk]

    return candidates


def generate_embedding_based_candidates(
    past_trans_df: pd.DataFrame, item_embeddings: dict[str, np.ndarray], top_k: int
) -> dict[str, list[str]]:
    print("Generating embedding candidates...")
    emb_df = pd.DataFrame(item_embeddings).T.reset_index()
    emb_df.columns = ["article_id"] + [f"emb_{i}" for i in range(emb_df.shape[1] - 1)]

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
        top_indices = np.argsort(similarity_matrix[i, :])[-top_k * 2 :][::-1]
        candidate_aids = item_ids[top_indices]
        if user_id in user_past_items:
            purchased_set = user_past_items[user_id]
            candidate_aids = [aid for aid in candidate_aids if aid not in purchased_set]
        candidates[user_id] = candidate_aids[:top_k]

    return candidates


def create_item2vec_embeddings(
    trans_df: pd.DataFrame,
    vector_size: int,
    window: int,
    min_count: int,
    workers: int,
    sg: int,
    seed: int,
) -> dict[str, np.ndarray]:
    print("Creating item2vec embeddings...")
    purchase_histories = (
        trans_df.sort_values("t_dat", ascending=True)
        .groupby("customer_id")["article_id"]
        .apply(list)
    )

    model = gensim.models.Word2Vec(
        sentences=purchase_histories,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        seed=seed,
    )

    item_embeddings = {
        article_id: model.wv[article_id] for article_id in model.wv.index_to_key
    }
    return item_embeddings


def create_cooccurrence_embeddings(
    cfg: Config, trans_df: pd.DataFrame
) -> dict[str, np.ndarray]:
    print("Creating co-occurrence embeddings with SVD...")
    daily_purchases = (
        trans_df.groupby(["t_dat", "customer_id"])["article_id"]
        .apply(list)
        .reset_index()
    )

    all_articles = trans_df["article_id"].unique()
    id2idx = {aid: i for i, aid in enumerate(all_articles)}
    idx2id = {i: aid for i, aid in enumerate(all_articles)}
    n_items = len(all_articles)

    rows, cols, data = [], [], []
    for articles in daily_purchases["article_id"]:
        for i, j in combinations(articles, 2):
            idx_i, idx_j = id2idx[i], id2idx[j]
            rows.extend([idx_i, idx_j])
            cols.extend([idx_j, idx_i])
            data.extend([1, 1])

    cooc_matrix = coo_matrix(
        (data, (rows, cols)), shape=(n_items, n_items), dtype=np.float32
    )

    if cfg.model.params.cooc.use_ppmi:
        total_events = cooc_matrix.sum()
        row_sum = np.array(cooc_matrix.sum(axis=1)).flatten()
        col_sum = np.array(cooc_matrix.sum(axis=0)).flatten()

        row_sum[row_sum == 0] = 1
        col_sum[col_sum == 0] = 1

        cooc_matrix_csr = cooc_matrix.tocsr()
        ppmi_rows, ppmi_cols, ppmi_data = [], [], []

        for i in range(n_items):
            for j in range(cooc_matrix_csr.indptr[i], cooc_matrix_csr.indptr[i + 1]):
                col = cooc_matrix_csr.indices[j]
                val = cooc_matrix_csr.data[j]

                # p(i, j) = count(i, j) / total_events
                # p(i) = row_sum(i) / total_events
                # p(j) = col_sum(j) / total_events
                # p(i, j) / p(i) * p(j) = (count(i, j) * total_events) / (row_sum(i) * col_sum(j))
                pmi = np.log2(val * total_events / (row_sum[i] * col_sum[col]))
                if pmi > 0:
                    ppmi_rows.append(i)
                    ppmi_cols.append(col)
                    ppmi_data.append(pmi)

        cooc_matrix = coo_matrix(
            (ppmi_data, (ppmi_rows, ppmi_cols)),
            shape=(n_items, n_items),
            dtype=np.float32,
        )

    svd = TruncatedSVD(
        n_components=cfg.model.params.cooc.n_components, random_state=cfg.seed
    )
    item_vectors = svd.fit_transform(cooc_matrix)

    item_embeddings = {idx2id[i]: item_vectors[i] for i in range(n_items)}
    return item_embeddings


def create_graph_embeddings(
    trans_df: pd.DataFrame, cfg: Config
) -> dict[str, np.ndarray]:
    print("Creating graph embeddings with Node2Vec...")
    edges = trans_df[["customer_id", "article_id"]].drop_duplicates().values
    graph = nx.Graph()
    graph.add_edges_from(edges)

    node2vec = Node2Vec(
        graph,
        dimensions=cfg.model.params.node2vec.dimensions,
        walk_length=cfg.model.params.node2vec.walk_length,
        num_walks=cfg.model.params.node2vec.num_walks,
        p=cfg.model.params.node2vec.p,
        q=cfg.model.params.node2vec.q,
        workers=cfg.model.params.node2vec.workers,
    )
    model = node2vec.fit(
        window=cfg.model.params.node2vec.window,
        min_count=cfg.model.params.node2vec.min_count,
    )

    item_ids = trans_df["article_id"].unique()
    item_embeddings = {aid: model.wv[aid] for aid in item_ids if aid in model.wv}
    return item_embeddings


def evaluate_candidate_sources(
    trans_df: pd.DataFrame,
    candidate_sources: dict[str, dict[str, list[str]]],
    result_dir: Path,
):
    """
    複数の候補生成ロジックの結果をまとめて評価し、比較する
    """
    print("\n--- Evaluating Candidate Generation Recall ---")

    ground_truth = trans_df.groupby("customer_id")["article_id"].apply(set).to_dict()
    total_true_items = sum(len(s) for s in ground_truth.values())

    recall_scores = {}
    for source_name, candidates_dict in candidate_sources.items():
        hits = 0
        for cid, true_items in ground_truth.items():
            pred_items = set(
                candidates_dict.get(cid, candidates_dict.get("all_users", []))
            )
            hits += len(true_items.intersection(pred_items))

        recall = hits / total_true_items if total_true_items > 0 else 0
        recall_scores[source_name] = recall
        print(f"Recall for {source_name}: {recall:.4f}")

    recall_series = pd.Series(recall_scores).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    recall_series.plot(kind="barh", ax=ax)
    ax.set_title("Recall by Candidate Source")
    ax.set_xlabel("Recall")
    fig.tight_layout()
    fig.savefig(result_dir.joinpath("candidate_recall.png"))
    plt.close()

    recall_series.to_csv(result_dir.joinpath("candidate_recall.csv"))


def create_ranking_features(
    cfg: Config,
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
        .merge(item_user_trans_features, on=["article_id", "customer_id"], how="left")
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
