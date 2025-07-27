import numpy as np
import pandas as pd

from src.schema.config import Config


class MMRReranker:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def rerank(
        self,
        item_purchase_cnt_map: dict[str, int],
        pred_items: list[str],
        pred_scores: np.ndarray,
        w: float,
    ) -> pd.DataFrame:
        pred_scores = (pred_scores - pred_scores.min()) / (
            pred_scores.max() - pred_scores.min()
        )
        seen = [False for _ in range(len(pred_items))]
        select_cnt = 0
        pred_items_rerank = []
        total = len(pred_items)
        while select_cnt < total:
            max_total_score = -float("inf")
            select_index = None
            for cand_i, pred_item in enumerate(pred_items):
                if seen[cand_i]:
                    continue
                pred_score = pred_scores[cand_i]
                popularity_score = item_purchase_cnt_map[pred_item]
                total_score = w * pred_score - (1 - w) * popularity_score
                if total_score > max_total_score:
                    max_total_score = total_score
                    select_index = cand_i
            pred_items_rerank.append(pred_items[select_index])
            seen[select_index] = True
            select_cnt += 1
        return pred_items_rerank
