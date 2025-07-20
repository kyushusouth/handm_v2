from collections import Counter
from itertools import combinations
from typing import Any

import numpy as np

from src.schema.config import Config


class MetricsCalculator:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def precision_at_k(self, true_items: list[Any], pred_items: list[Any]) -> float:
        if len(pred_items) == 0:
            return 0
        precision_at_k = (len(set(true_items) & set(pred_items))) / len(pred_items)
        return precision_at_k

    def recall_at_k(self, true_items: list[Any], pred_items: list[Any]) -> float:
        if len(true_items) == 0:
            return 0
        recall_at_k = (len(set(true_items) & set(pred_items))) / len(true_items)
        return recall_at_k

    def f1_at_k(self, true_items: list[Any], pred_items: list[Any]) -> float:
        precision = self.precision_at_k(true_items, pred_items)
        recall = self.recall_at_k(true_items, pred_items)
        if precision + recall == 0:
            return 0
        return 2 * precision * recall / (precision + recall)

    def rr_at_k(self, true_items: list[Any], pred_items: list[Any]) -> float:
        rr = 0
        for rank, pred_item in enumerate(pred_items):
            if pred_item in true_items:
                rr = 1 / (rank + 1)
        return rr

    def get_user_relevance(
        self, true_items: list[Any], pred_items: list[Any]
    ) -> list[int]:
        user_relevances = [0 for _ in range(len(pred_items))]
        for rank, pred_item in enumerate(pred_items):
            if pred_item in true_items:
                user_relevances[rank] = 1
        return user_relevances

    def ap_at_k(self, true_items: list[Any], pred_items: list[Any]) -> float:
        user_relevances = self.get_user_relevance(true_items, pred_items)
        if sum(user_relevances) == 0:
            return 0
        nonzero_indices = np.asarray(user_relevances).nonzero()[0]
        return sum(
            [sum(user_relevances[: idx + 1]) / (idx + 1) for idx in nonzero_indices]
        ) / sum(user_relevances)

    def dcg_at_k(self, user_relevances: list[int]) -> float:
        return user_relevances[0] + np.sum(
            user_relevances[1:] / np.log2(np.arange(2, len(user_relevances) + 1))
        )

    def ndcg_at_k(self, true_items: list[Any], pred_items: list[Any]) -> float:
        user_relevances = self.get_user_relevance(true_items, pred_items)
        if sum(user_relevances) == 0:
            return 0
        dcg_max = self.dcg_at_k(sorted(user_relevances, reverse=True))
        if not dcg_max:
            return 0
        return self.dcg_at_k(user_relevances) / dcg_max

    def coverage(
        self,
        all_pred_items: set[Any],
        all_items_catalog: set[Any],
    ) -> float:
        return len(all_pred_items) / len(all_items_catalog)

    def gini_index(self, all_pred_items: list[Any]) -> float:
        counts = np.array(list(Counter(all_pred_items).values()))
        sorted_counts = np.sort(counts)
        cumulative_counts = np.cumsum(sorted_counts) / np.sum(sorted_counts)
        cumulative_counts = np.insert(cumulative_counts, 0, 0)
        area_under_curve = np.trapz(cumulative_counts, dx=1 / len(cumulative_counts))
        return 1 - 2 * area_under_curve

    def dissimilarity_score(self, pred_items: dict[str, list[Any]]) -> float:
        dissimilarity_scores = []
        for user1, user2 in combinations(pred_items.keys(), 2):
            list1, list2 = (
                set(pred_items.get(user1, [])),
                set(pred_items.get(user2, [])),
            )
            if not list1 or not list2:
                continue
            intersection, union = (
                len(list1.intersection(list2)),
                len(list1.union(list2)),
            )
            dissimilarity_scores.append(1.0 - (intersection / union))
        return np.mean(dissimilarity_scores).item() if dissimilarity_scores else 0.0

    def novelty(self, past_items: list[Any], pred_items: list[Any]) -> float:
        all_interactions = [item for sublist in past_items for item in sublist]
        item_counts = Counter(all_interactions)
        total_interactions = len(all_interactions)
        item_popularity_prob = {
            item: count / total_interactions for item, count in item_counts.items()
        }
        mean_novelty_scores = []
        for pred_list in pred_items:
            user_novelty = [
                -np.log2(item_popularity_prob.get(item_id, 1e-9))
                for item_id in pred_list
            ]
            if user_novelty:
                mean_novelty_scores.append(np.mean(user_novelty))
        return np.mean(mean_novelty_scores).item() if mean_novelty_scores else 0.0

    def serendipity(
        self,
        past_items: dict[str, list[Any]],
        true_items: dict[str, list[Any]],
        pred_items: dict[str, list[Any]],
    ) -> float:
        serendipity_scores = []
        for user_id, pred_list in pred_items.items():
            past_set = set(past_items.get(user_id, []))
            true_set = set(true_items.get(user_id, []))
            if not pred_list or not true_set:
                continue
            serendipitous_count = sum(
                1 for item in pred_list if item not in past_set and item in true_set
            )
            serendipity_scores.append(serendipitous_count / len(pred_list))
        return np.mean(serendipity_scores).item() if serendipity_scores else 0.0
