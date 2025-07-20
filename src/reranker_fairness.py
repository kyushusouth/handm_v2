from typing import Literal

import pandas as pd

from src.fairness_metrics import PopularityBasedFairness


class FairReranker:
    """
    論文で提案された公平性ポリシーに基づき、予測結果のDataFrameをリランキングするクラス。
    """

    def __init__(self, fairness_calculator: PopularityBasedFairness):
        """
        Args:
            fairness_calculator (PopularityBasedFairness): 初期化済みの公平性計算機
        """
        self.fairness_calculator = fairness_calculator

    def rerank(
        self,
        user_df: pd.DataFrame,
        policy: Literal["interpolation", "guaranteed_relevance"],
        top_k: int,
        **kwargs,
    ) -> pd.DataFrame:
        """
        指定されたポリシーに基づいてDataFrameをリランキングし、新しいスコアカラムを追加する。

        Args:
            user_df (pd.DataFrame): 'article_id'と'pred'カラムを持つDataFrame
            policy (str): "interpolation" または "guaranteed_relevance"
            top_k (int): 最終的に推薦するアイテム数
            **kwargs: 各ポリシーが必要とするパラメータ
                      - policy='interpolation' の場合: beta (float)
                      - policy='guaranteed_relevance' の場合: relevance_threshold (float)
        """
        if policy == "interpolation":
            beta = kwargs.get("beta")
            if beta is None:
                raise ValueError(
                    "`beta` parameter is required for 'interpolation' policy."
                )
            return self._policy_interpolation(user_df, beta, top_k)

        elif policy == "guaranteed_relevance":
            relevance_threshold = kwargs.get("relevance_threshold")
            if relevance_threshold is None:
                raise ValueError(
                    "`relevance_threshold` parameter is required for 'guaranteed_relevance' policy."
                )
            return self._policy_guaranteed_relevance(user_df, relevance_threshold)

        else:
            raise ValueError(
                f"Unknown policy: '{policy}'. Available policies are 'interpolation', 'guaranteed_relevance'."
            )

    def _policy_interpolation(
        self, user_df: pd.DataFrame, beta: float, top_k: int
    ) -> list[str]:
        """
        ポリシー① (修正版): 関連性と「公平性の marginal gain」の加重和でリランキング
        """
        df = user_df.copy()

        # 処理を高速化するため、一度辞書に変換
        relevance_scores = pd.Series(
            df["pred"].values, index=df["article_id"]
        ).to_dict()

        pred_items = []

        # 候補リストを元のスコア順にソートしておく
        remaining_candidates = list(
            df.sort_values("pred", ascending=False)["article_id"]
        )

        while len(pred_items) < top_k and remaining_candidates:
            step_scores = {}
            # 現在の推薦リストの公平性スコアを計算
            base_fairness = self.fairness_calculator.calculate_fairness_score(
                pred_items
            )

            for cand_id in remaining_candidates:
                # 候補を追加した場合の「公平性の増分 (marginal gain)」を計算
                temp_list = pred_items + [cand_id]
                marginal_fairness_gain = (
                    self.fairness_calculator.calculate_fairness_score(temp_list)
                    - base_fairness
                )

                relevance = relevance_scores[cand_id]

                # 新しいスコア = (1-β) * 関連性 + β * 公平性の増分
                # βが公平性の重要度をコントロールする
                combined_score = (1 - beta) * relevance + beta * marginal_fairness_gain
                step_scores[cand_id] = combined_score

            # このステップで最もスコアが高いアイテムを選択
            best_next_id = max(step_scores, key=step_scores.get)

            pred_items.append(best_next_id)
            remaining_candidates.remove(best_next_id)

        return pred_items

    def _policy_guaranteed_relevance(
        self, user_df: pd.DataFrame, relevance_threshold: float, top_k: int
    ) -> list[str]:
        """ポリシー②: 最低限の関連性を保証し、その中で公平性を最大化"""
        df = user_df.copy()

        relevant_items = df[df["pred"] >= relevance_threshold].copy()
        if relevant_items.empty:
            df["fairness_rerank_score"] = -1.0
            return df

        # 公平性（人気度ビンの番号が小さい＝ニッチなほど良い）でソート
        relevant_items["fairness_rank"] = relevant_items["article_id"].apply(
            lambda x: self.fairness_calculator.item_to_bin.get(x, 99)
        )
        reranked_df = relevant_items.sort_values("fairness_rank", ascending=False)
        pred_items = reranked_df["article_id"].values.tolist()[:top_k]

        return pred_items


# import numpy as np
# import pandas as pd
# from typing import Literal

# from src.fairness_metrics import PopularityBasedFairness, UserExploratoryScoreCalculator

# class FairReranker:
#     """
#     論文で提案された公平性ポリシーに基づき、予測結果のDataFrameをリランキングするクラス。
#     """
#     def __init__(self, fairness_calculator: PopularityBasedFairness):
#         self.fairness_calculator = fairness_calculator

#     def rerank(
#         self,
#         user_df: pd.DataFrame,
#         policy: Literal["interpolation", "guaranteed_relevance", "adaptive"],
#         top_k: int,
#         **kwargs,
#     ) -> pd.DataFrame:
#         """
#         指定されたポリシーに基づいてDataFrameをリランキングし、新しいスコアカラムを追加する。
#         """
#         if policy == "interpolation":
#             beta = kwargs.get("beta")
#             if beta is None: raise ValueError("`beta` is required for 'interpolation' policy.")
#             return self._policy_interpolation(user_df, beta, top_k)

#         elif policy == "guaranteed_relevance":
#             relevance_threshold = kwargs.get("relevance_threshold")
#             if relevance_threshold is None: raise ValueError("`relevance_threshold` is required for 'guaranteed_relevance' policy.")
#             return self._policy_guaranteed_relevance(user_df, relevance_threshold)

#         elif policy == "adaptive":
#             user_exploratory_score = kwargs.get("user_exploratory_score")
#             if user_exploratory_score is None: raise ValueError("`user_exploratory_score` is required for 'adaptive' policy.")
#             return self._policy_adaptive(user_df, user_exploratory_score, top_k)

#         else:
#             raise ValueError(f"Unknown policy: '{policy}'.")

#     def _policy_interpolation(self, user_df: pd.DataFrame, beta: float, top_k: int) -> pd.DataFrame:
#         """ポリシー①: 関連性と「公平性の marginal gain」の加重和でリランキング"""
#         return self._greedy_rerank_with_beta(user_df, beta, top_k)

#     def _policy_adaptive(self, user_df: pd.DataFrame, user_exploratory_score: float, top_k: int) -> pd.DataFrame:
#         """
#         ポリシー③ (修正版): ユーザーの探索性スコアをβとして利用し、貪欲法でリランキング
#         """
#         # ユーザーの探索性スコア(事前に0-1に正規化されている想定)をβとして使用
#         beta = user_exploratory_score
#         return self._greedy_rerank_with_beta(user_df, beta, top_k)

#     def _greedy_rerank_with_beta(self, user_df: pd.DataFrame, beta: float, top_k: int) -> pd.DataFrame:
#         """
#         関連性と公平性の増分に基づき、貪欲法でリランキングを行う共通ロジック。
#         """
#         df = user_df.copy()
#         relevance_scores = pd.Series(df["pred"].values, index=df["article_id"]).to_dict()
#         remaining_candidates = list(df.sort_values("pred", ascending=False)["article_id"])

#         reranked_ids = []
#         rerank_scores_map = {}

#         while len(reranked_ids) < top_k and remaining_candidates:
#             step_scores = {}
#             base_fairness = self.fairness_calculator.calculate_fairness_score(reranked_ids)

#             for cand_id in remaining_candidates:
#                 temp_list = reranked_ids + [cand_id]
#                 marginal_fairness_gain = self.fairness_calculator.calculate_fairness_score(temp_list) - base_fairness

#                 relevance = relevance_scores.get(cand_id, 0)

#                 # βを重みとして、関連性と公平性の増分を組み合わせる
#                 combined_score = (1 - beta) * relevance + beta * marginal_fairness_gain
#                 step_scores[cand_id] = combined_score

#             best_next_id = max(step_scores, key=step_scores.get)

#             reranked_ids.append(best_next_id)
#             remaining_candidates.remove(best_next_id)

#             # ランクをスコアとして保存 (上位ほど高いスコア)
#             rerank_scores_map[best_next_id] = top_k - len(reranked_ids) + 1

#         df['fairness_rerank_score'] = df['article_id'].map(rerank_scores_map).fillna(-1.0)
#         return df


#     def _policy_guaranteed_relevance(self, user_df: pd.DataFrame, relevance_threshold: float) -> pd.DataFrame:
#         """ポリシー②: 最低限の関連性を保証し、その中で公平性を最大化"""
#         # (このメソッドは変更なし)
#         df = user_df.copy()
#         relevant_items = df[df["pred"] >= relevance_threshold].copy()
#         if relevant_items.empty:
#             df["fairness_rerank_score"] = -1.0
#             return df
#         relevant_items["fairness_rank"] = relevant_items["article_id"].apply(
#             lambda x: self.fairness_calculator.item_to_bin.get(x, 99)
#         )
#         reranked_df = relevant_items.sort_values("fairness_rank", ascending=True)
#         reranked_scores = pd.Series(np.arange(len(reranked_df), 0, -1), index=reranked_df.index)
#         df["fairness_rerank_score"] = reranked_scores.fillna(-1.0)
#         return df
