from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import shap

from src.metrics_calculator import MetricsCalculator
from src.ranker import Ranker
from src.reranker_item2vec import MMRRerankerItemEmbeddingSimilarity
from src.schema.config import Config


class LGBRanker(Ranker):
    def __init__(
        self,
        cfg: Config,
        metrics_calculator: MetricsCalculator,
    ) -> None:
        super().__init__(cfg, metrics_calculator)

    def train(self, result_dir: Path) -> "LGBRanker":
        self.ranker = lgb.LGBMRanker(
            objective=self.cfg.model.params.lgb.objective,
            metric=self.cfg.model.params.lgb.eval_metric,
            max_depth=self.cfg.model.params.lgb.max_depth,
            learning_rate=self.cfg.model.params.lgb.learning_rate,
            n_estimators=self.cfg.model.params.lgb.n_estimators,
            importance_type=self.cfg.model.params.lgb.importance_type,
            random_state=self.cfg.seed,
            verbosity=-1,
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
                ),
                lgb.log_evaluation(),
            ],
        )

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
        metric_colormaps = {"ndcg": "Blues"}
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
        plt.savefig(
            result_dir.joinpath(f"{self.__class__.__name__}_learning_curve.png")
        )
        plt.close(fig)

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
        lgb.plot_importance(
            self.ranker, ax=ax, max_num_features=20, importance_type="gain"
        )
        plt.tight_layout()
        plt.savefig(
            result_dir.joinpath(f"{self.__class__.__name__}_feature_importance.png")
        )
        plt.close(fig)

        explainer = shap.TreeExplainer(self.ranker)
        shap_samples = self.val_trans_df.sample(
            self.cfg.exp.summary_plot_num_sample, random_state=self.cfg.seed
        )[self.cfg.model.features.cat + self.cfg.model.features.num]
        shap_values = explainer.shap_values(shap_samples)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
        shap.summary_plot(shap_values, shap_samples, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(
            result_dir.joinpath(f"{self.__class__.__name__}_shap_summary_plot.png")
        )
        plt.close(fig)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))
        shap.summary_plot(shap_values, shap_samples, plot_type="dot", show=False)
        plt.tight_layout()
        plt.savefig(
            result_dir.joinpath(f"{self.__class__.__name__}_shap_summary_plot_dot.png")
        )
        plt.close(fig)

        return self
