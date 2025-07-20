import numpy as np
import pandas as pd

from src.schema.config import Config


class MMRRerankerItemEmbeddingSimilarity:
    """
    MMR (Maximal Marginal Relevance) を使って、
    DataFrameに 'mmr_score' を追加し、リランキングを行うクラス。
    """

    def __init__(self, cfg: Config, item_embeddings: dict[str, np.ndarray]) -> None:
        self.cfg = cfg
        self.item_ids = list(item_embeddings.keys())
        self.item_vectors = np.array(list(item_embeddings.values())).astype("float32")
        self.item_vectors_norm = self.item_vectors / np.linalg.norm(
            self.item_vectors, axis=1, keepdims=True
        )
        self.id_to_idx = {aid: i for i, aid in enumerate(self.item_ids)}

    def _calculate_cosine_similarity(self, idx1: int, idx2: int) -> float:
        return np.dot(self.item_vectors_norm[idx1], self.item_vectors_norm[idx2])

    def rerank_dataframe(
        self, user_df: pd.DataFrame, w: float, col_name: str
    ) -> pd.DataFrame:
        df = user_df.copy()

        valid_cands = df[df["article_id"].isin(self.id_to_idx)].copy()
        if valid_cands.empty:
            df[col_name] = -float("inf")
            return df

        original_scores = valid_cands["pred"].values
        normalized_scores = (original_scores - original_scores.min()) / (
            original_scores.max() - original_scores.min() + 1e-9
        )

        cand_scores = dict(zip(valid_cands["article_id"], normalized_scores))
        remaining_candidates_ids = list(valid_cands["article_id"])

        reranked_ids = []
        mmr_scores_map = {}

        best_first_id = remaining_candidates_ids[0]
        reranked_ids.append(best_first_id)
        remaining_candidates_ids.remove(best_first_id)
        mmr_scores_map[best_first_id] = cand_scores[best_first_id]

        while len(reranked_ids) < self.cfg.exp.num_rec and remaining_candidates_ids:
            mmr_scores = {}
            for cand_id in remaining_candidates_ids:
                cand_idx = self.id_to_idx[cand_id]
                original_score = cand_scores[cand_id]

                max_similarity = max(
                    [
                        self._calculate_cosine_similarity(
                            cand_idx, self.id_to_idx[sel_id]
                        )
                        for sel_id in reranked_ids
                    ],
                    default=0,
                )

                mmr_score = w * original_score - (1 - w) * max_similarity
                mmr_scores[cand_id] = mmr_score

            best_next_id = max(mmr_scores, key=mmr_scores.get)

            reranked_ids.append(best_next_id)
            remaining_candidates_ids.remove(best_next_id)
            mmr_scores_map[best_next_id] = mmr_scores[best_next_id]

        df[col_name] = df["article_id"].map(mmr_scores_map).fillna(-float("inf"))

        return df
