from itertools import combinations

import gensim
import networkx as nx
import numpy as np
import pandas as pd
from node2vec import Node2Vec
from scipy.sparse import coo_matrix
from sklearn.decomposition import TruncatedSVD

from src.schema.config import Config


class EmbeddingGenerator:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def create_item2vec_embeddings(
        self, trans_df: pd.DataFrame
    ) -> dict[str, np.ndarray]:
        print("Creating item2vec embeddings...")
        purchase_histories = (
            trans_df.sort_values("t_dat", ascending=True)
            .groupby("customer_id")["article_id"]
            .apply(list)
        )
        model = gensim.models.Word2Vec(
            sentences=purchase_histories,
            vector_size=self.cfg.model.params.item2vec.vector_size,
            window=self.cfg.model.params.item2vec.window,
            min_count=self.cfg.model.params.item2vec.min_count,
            workers=self.cfg.model.params.item2vec.workers,
            sg=self.cfg.model.params.item2vec.sg,
            seed=self.cfg.seed,
        )
        item_embeddings = {
            article_id: model.wv[article_id] for article_id in model.wv.index_to_key
        }
        return item_embeddings

    def create_cooccurrence_embeddings(
        self, trans_df: pd.DataFrame
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

        if self.cfg.model.params.cooc.use_ppmi:
            total_events = cooc_matrix.sum()
            row_sum = np.array(cooc_matrix.sum(axis=1)).flatten()
            col_sum = np.array(cooc_matrix.sum(axis=0)).flatten()

            row_sum[row_sum == 0] = 1
            col_sum[col_sum == 0] = 1

            cooc_matrix_csr = cooc_matrix.tocsr()
            ppmi_rows, ppmi_cols, ppmi_data = [], [], []

            for i in range(n_items):
                for j in range(
                    cooc_matrix_csr.indptr[i], cooc_matrix_csr.indptr[i + 1]
                ):
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
            n_components=self.cfg.model.params.cooc.n_components,
            random_state=self.cfg.seed,
        )
        item_vectors = svd.fit_transform(cooc_matrix)
        item_embeddings = {idx2id[i]: item_vectors[i] for i in range(n_items)}
        return item_embeddings

    def create_graph_embeddings(self, trans_df: pd.DataFrame) -> dict[str, np.ndarray]:
        print("Creating graph embeddings with Node2Vec...")
        edges = trans_df[["customer_id", "article_id"]].drop_duplicates().values
        graph = nx.Graph()
        graph.add_edges_from(edges)

        node2vec = Node2Vec(
            graph,
            dimensions=self.cfg.model.params.node2vec.dimensions,
            walk_length=self.cfg.model.params.node2vec.walk_length,
            num_walks=self.cfg.model.params.node2vec.num_walks,
            p=self.cfg.model.params.node2vec.p,
            q=self.cfg.model.params.node2vec.q,
            workers=self.cfg.model.params.node2vec.workers,
        )
        model = node2vec.fit(
            window=self.cfg.model.params.node2vec.window,
            min_count=self.cfg.model.params.node2vec.min_count,
        )

        item_ids = trans_df["article_id"].unique()
        item_embeddings = {aid: model.wv[aid] for aid in item_ids if aid in model.wv}
        return item_embeddings
