from collections import Counter

import numpy as np
import pandas as pd


class PopularityBasedFairness:
    def __init__(
        self,
        all_items_df: pd.DataFrame,
        item_id_col: str,
        popularity_col: str,
        num_bins: int,
    ):
        self.item_to_bin = pd.Series(
            pd.qcut(
                all_items_df[popularity_col],
                q=num_bins,
                labels=False,
                duplicates="drop",
            ),
            index=all_items_df[item_id_col],
        ).to_dict()

    def calculate_fairness_score(self, item_list: list[str]) -> float:
        if not item_list:
            return 0.0
        bin_counts = Counter(
            [
                self.item_to_bin.get(item)
                for item in item_list
                if item in self.item_to_bin
            ]
        )
        fairness_score = sum(np.sqrt(count) for count in bin_counts.values())
        return fairness_score
